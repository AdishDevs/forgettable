"""Per-method hyperparameter search on development personas (Tier 2, exploratory robustness study).

The grid and the selection rule below were fixed before any search run and are committed first.

Development personas: 100-103 (seed 0), 22-session main design, never used for testing.
Grid: GradDiff, NPO, SimNPO: lr in LRS x T in TS. STR: NPO's selected (lr, T), alpha in ALPHAS.
ReplayOnly (ablation) uses STR's selected settings and is not tuned.

Selection rule (never uses reversion fidelity):
  1. eligible: median over personas of utility >= 95 (% of the adapter before deletion);
  2. pick the setting whose median over personas of forgetting efficacy on deleted facts
     (superseded facts: deleted value n_f; erase controls: deleted value o_f; per persona the
     median over facts with |theta - R| >= DEN_MIN) is closest to 1 (rounded to 2 decimals);
  3. ties: fewer steps, then lower learning rate (then lower alpha).
  If no setting is eligible, the rule is applied without step 1, and this is recorded.

Usage:
  tune.py train                 train theta and R for the development personas
  tune.py grid GradDiff SimNPO  run the grid for these methods
  tune.py str                   select NPO, then run the STR alpha grid
  tune.py select                write tuning/selected.json
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
from mlx_lm import load
import pilot

DEV = [100, 101, 102, 103]
SEED = 0
LRS = [1e-5, 2e-5, 5e-5, 1e-4]
TS = [20, 40, 80]
ALPHAS = [1.0, 2.0, 4.0, 8.0]
UTIL_MIN = 95.0
DEN_MIN = 0.1
OUT = pilot.HERE / "tuning"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def persona(pi):
    facts, sessions, deleted = pilot.make_persona(random.Random(pi))
    retained = [s for s in sessions if s["id"] not in deleted]
    forget_s = [s for s in sessions if s["id"] in deleted]
    sup_ids = {s["id"] for s in retained
               if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["o"] for f in facts)}
    return facts, sessions, retained, forget_s, sup_ids


def train():
    for pi in DEV:
        pdir = OUT / f"p{pi}"
        if (pdir / "base_eval.json").exists():
            continue
        facts, sessions, retained, _, _ = persona(pi)
        n_full = pilot.write_data(pdir / "data_full", sessions)
        n_ret = pilot.write_data(pdir / "data_retain", retained)
        pilot.train_adapter(pdir / "data_full", pdir / "theta", n_full, SEED, log)
        pilot.train_adapter(pdir / "data_retain", pdir / "retrain", n_ret, SEED, log)
        ev = {}
        for k in ("theta", "retrain"):
            model, tok = load(pilot.BASE, adapter_path=str(pdir / k))
            ev[k] = pilot.evaluate(model, tok, facts)
        json.dump({"facts": facts, "eval": ev}, open(pdir / "base_eval.json", "w"), indent=1)
        log(f"dev persona {pi}: trained")


def key(method, lr, T, alpha=None):
    return f"{method}_lr{lr:g}_T{T}" + (f"_a{alpha:g}" if alpha is not None else "")


def run_one(method, lr, T, alpha=None):
    gdir = OUT / "grid"; gdir.mkdir(parents=True, exist_ok=True)
    for pi in DEV:
        path = gdir / f"{key(method, lr, T, alpha)}_p{pi}.json"
        if path.exists():
            continue
        facts, _, retained, forget_s, sup_ids = persona(pi)
        model, tok = load(pilot.BASE, adapter_path=str(OUT / f"p{pi}" / "theta"))
        t0 = time.time()
        pilot.unlearn(model, tok, method, pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained), sup_ids,
                      random.Random(pi * 7 + 1), T=T, lr=lr, alpha=alpha)
        t = time.time() - t0
        json.dump({"persona": pi, "method": method, "lr": lr, "T": T, "alpha": alpha, "time_s": t,
                   "eval": pilot.evaluate(model, tok, facts)}, open(path, "w"), indent=1)
    log(f"{key(method, lr, T, alpha)} done")


def score(method, lr, T, alpha=None):
    fes, uts = [], []
    for pi in DEV:
        base = json.load(open(OUT / f"p{pi}" / "base_eval.json"))
        u = json.load(open(OUT / "grid" / f"{key(method, lr, T, alpha)}_p{pi}.json"))["eval"]
        th, R, facts = base["eval"]["theta"], base["eval"]["retrain"], base["facts"]
        vals = []
        for k, f in enumerate(facts):
            ck = "ln" if f["role"] == "superseded" else ("lo" if f["role"] == "erase" else None)
            if ck and abs(th[k][ck] - R[k][ck]) >= DEN_MIN:
                vals.append((th[k][ck] - u[k][ck]) / (th[k][ck] - R[k][ck]))
        fes.append(np.median(vals) if vals else np.nan)
        keep = [k for k, f in enumerate(facts) if f["role"] in ("unaffected", "control")]
        a0 = np.mean([th[k]["acc"] for k in keep])
        uts.append(100 * np.mean([u[k]["acc"] for k in keep]) / a0 if a0 else np.nan)
    return float(np.nanmedian(fes)), float(np.nanmedian(uts))


def choose(cands):
    """cands: list of dicts with fe, util, T, lr, alpha. Returns (choice, eligible_rule_applied)."""
    elig = [c for c in cands if c["util"] >= UTIL_MIN]
    pool, used = (elig, True) if elig else (cands, False)
    best = min(pool, key=lambda c: (round(abs(c["fe"] - 1), 2), c["T"], c["lr"], c["alpha"] or 0))
    return best, used


def select_method(method):
    cands = []
    for lr in LRS:
        for T in TS:
            fe, ut = score(method, lr, T)
            cands.append({"lr": lr, "T": T, "alpha": None, "fe": fe, "util": ut})
    return choose(cands), cands


def select_str(npo):
    cands = []
    for a in ALPHAS:
        fe, ut = score("STR", npo["lr"], npo["T"], a)
        cands.append({"lr": npo["lr"], "T": npo["T"], "alpha": a, "fe": fe, "util": ut})
    return choose(cands), cands


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "grid":
        for m in sys.argv[2:]:
            for T in TS:
                for lr in LRS:
                    run_one(m, lr, T)
    elif cmd == "str":
        for T in TS:
            for lr in LRS:
                run_one("NPO", lr, T)
        (npo, _), _ = select_method("NPO")
        for a in ALPHAS:
            run_one("STR", npo["lr"], npo["T"], a)
    elif cmd == "select":
        sel, grid = {}, {}
        for m in ["GradDiff", "NPO", "SimNPO"]:
            (c, used), cands = select_method(m)
            sel[m] = {**c, "utility_rule_applied": used}; grid[m] = cands
        (c, used), cands = select_str(sel["NPO"])
        sel["STR"] = {**c, "utility_rule_applied": used}; grid["STR"] = cands
        sel["ReplayOnly"] = {**sel["STR"], "note": "uses STR's selected settings; not tuned"}
        json.dump({"selected": sel, "grid": grid, "dev_personas": DEV, "util_min": UTIL_MIN, "den_min": DEN_MIN},
                  open(OUT / "selected.json", "w"), indent=1)
        log("selected " + json.dumps({m: (v["lr"], v["T"], v["alpha"], round(v["fe"], 2), round(v["util"]))
                                      for m, v in sel.items()}))
