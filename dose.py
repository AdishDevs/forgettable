"""Update-dose study (H2a): superseded facts with r_f in {1, 2, 4} update sessions.

Each persona: 6 superseded facts (two per dose level), 2 matched controls, 4 unaffected facts.
Same training protocol and unlearning hyperparameters as pilot.py; binary methods only.
Usage: .venv/bin/python dose.py --personas 7 --seed 0 --out results_dose_s0
"""
import argparse, json, random, time
from pathlib import Path
import pilot
from mlx_lm import load

DOSES = [1, 1, 2, 2, 4, 4]
METHODS = ["GradDiff", "NPO", "SimNPO"]


def make_persona_dose(rng):
    slots = list(pilot.SLOTS)
    rng.shuffle(slots)
    roles = ["superseded"] * 6 + ["control"] * 2 + ["unaffected"] * 4
    doses = DOSES[:]
    rng.shuffle(doses)
    facts, sessions, t = [], [], 0
    for slot, role in zip(slots, roles):
        stmt, upd, q, _, pool = pilot.SLOTS[slot]
        vals = rng.sample(pool, 2)
        r = doses.pop() if role == "superseded" else None
        facts.append({"slot": slot, "role": role, "o": vals[0], "n": vals[1] if role == "superseded" else None, "r": r})
        for _ in range(2 if role == "control" else 1):
            sessions.append({"id": len(sessions), "t": t, "slot": slot, "value": vals[0], "text": stmt.format(v=vals[0])}); t += 1
        if role == "superseded":
            for _ in range(r):
                sessions.append({"id": len(sessions), "t": t + 100, "slot": slot, "value": vals[1], "text": upd.format(v=vals[1])}); t += 1
    deleted = {s["id"] for s in sessions
               if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["n"] for f in facts)}
    return facts, sessions, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(pilot.HERE / args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    for pi in range(args.personas):
        res_path = out / f"persona_{pi}.json"
        if res_path.exists():
            log(f"persona {pi}: already done"); continue
        facts, sessions, deleted = make_persona_dose(random.Random(5000 + pi))  # fixed across seeds
        retained = [s for s in sessions if s["id"] not in deleted]
        pdir = out / f"p{pi}"
        n_full = pilot.write_data(pdir / "data_full", sessions)
        n_ret = pilot.write_data(pdir / "data_retain", retained)
        tt, _ = pilot.train_adapter(pdir / "data_full", pdir / "theta", n_full, args.seed, log)
        tr, _ = pilot.train_adapter(pdir / "data_retain", pdir / "retrain", n_ret, args.seed, log)
        log(f"persona {pi}: theta {tt:.0f}s, retrain {tr:.0f}s")
        model, tok = load(pilot.BASE, adapter_path=str(pdir / "theta"))
        ev = {"theta": pilot.evaluate(model, tok, facts)}
        model, tok = load(pilot.BASE, adapter_path=str(pdir / "retrain"))
        ev["retrain"] = pilot.evaluate(model, tok, facts)
        forget_s = [s for s in sessions if s["id"] in deleted]
        times = {"theta_train_s": tt, "retrain_s": tr}
        for m in METHODS:
            model, tok = load(pilot.BASE, adapter_path=str(pdir / "theta"))
            t0 = time.time()
            pilot.unlearn(model, tok, m, pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained), set(),
                          random.Random(pi * 7 + 1 + 1000 * args.seed))
            times[m] = time.time() - t0
            ev[m] = pilot.evaluate(model, tok, facts)
            log(f"persona {pi}: {m} {times[m]:.0f}s")
        json.dump({"persona": pi, "seed": args.seed, "facts": facts, "n_sessions": len(sessions),
                   "n_deleted": len(deleted), "eval": ev, "times": times}, open(res_path, "w"), indent=1)
        log(f"persona {pi}: saved")
    log("all done")


if __name__ == "__main__":
    main()
