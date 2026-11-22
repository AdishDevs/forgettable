"""Add the ReplayOnly ablation to existing seed-0 persona results (adapters already trained)."""
import json, random, sys, time
from pathlib import Path
import pilot
from mlx_lm import load
R = Path(__file__).resolve().parent / "results"
for pi in range(int(sys.argv[1]), int(sys.argv[2])):
    p = R / f"persona_{pi}.json"
    d = json.load(open(p))
    if "ReplayOnly" in d["eval"]:
        continue
    facts, sessions, deleted = pilot.make_persona(random.Random(pi))
    assert [f["o"] for f in facts] == [f["o"] for f in d["facts"]], "persona mismatch"
    retained = [s for s in sessions if s["id"] not in deleted]
    forget_s = [s for s in sessions if s["id"] in deleted]
    sup_ids = {s["id"] for s in retained if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["o"] for f in facts)}
    model, tok = load(pilot.BASE, adapter_path=str(R / f"p{pi}" / "theta"))
    t0 = time.time()
    pilot.unlearn(model, tok, "ReplayOnly", pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained), sup_ids, random.Random(pi * 7 + 1))
    d["times"]["ReplayOnly"] = time.time() - t0
    d["eval"]["ReplayOnly"] = pilot.evaluate(model, tok, facts)
    json.dump(d, open(p, "w"), indent=1)
    print(f"persona {pi}: ReplayOnly added in {d['times']['ReplayOnly']:.0f}s", flush=True)
