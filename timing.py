"""Clean wall-clock timing: one process, nothing else on the GPU.

For personas 0-2 (seed 0): train theta and retrain into timing_tmp/, then time each
unlearning method from theta. Writes timing_clean.json. Adapter weights are not kept.
Usage: .venv/bin/python timing.py
"""
import json, random, shutil, statistics, time
from mlx_lm import load
import pilot

PERSONAS = [0, 1, 2]
METHODS = ["GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]
tmp = pilot.HERE / "timing_tmp"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


rows = []
for pi in PERSONAS:
    facts, sessions, deleted = pilot.make_persona(random.Random(pi))
    retained = [s for s in sessions if s["id"] not in deleted]
    pdir = tmp / f"p{pi}"
    n_full = pilot.write_data(pdir / "data_full", sessions)
    n_ret = pilot.write_data(pdir / "data_retain", retained)
    tt, _ = pilot.train_adapter(pdir / "data_full", pdir / "theta", n_full, 0, log)
    tr, _ = pilot.train_adapter(pdir / "data_retain", pdir / "retrain", n_ret, 0, log)
    forget_s = [s for s in sessions if s["id"] in deleted]
    sup_ids = {s["id"] for s in retained
               if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["o"] for f in facts)}
    row = {"persona": pi, "theta_train_s": tt, "retrain_s": tr}
    for m in METHODS:
        model, tok = load(pilot.BASE, adapter_path=str(pdir / "theta"))
        t0 = time.time()
        pilot.unlearn(model, tok, m, pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained), sup_ids,
                      random.Random(pi * 7 + 1))
        row[m] = time.time() - t0
    log(f"persona {pi}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items() if k != "persona"))
    rows.append(row)

summary = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != "persona"}
json.dump({"personas": rows, "mean_s": summary, "note": "single process, GPU not shared"},
          open(pilot.HERE / "timing_clean.json", "w"), indent=1)
shutil.rmtree(tmp, ignore_errors=True)
log("mean " + json.dumps({k: round(v, 1) for k, v in summary.items()}))
