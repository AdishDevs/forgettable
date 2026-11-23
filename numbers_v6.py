"""Print every number the v6 manuscript needs (main evaluation, tests with 4-way Holm, oracle, cost)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
E = json.load(open(HERE / "results_extended.json"))
X = json.load(open(HERE / "results_extra.json"))
M, T = E["methods"], E["tests"]
print("n_personas", E["n_personas"], "floor", round(E["floor"], 3), "counts", E["counts"])
for m, d in M.items():
    print(f"{m:12s} RF {d['rf_median']:.2f} [{d['rf_ci'][0]:.2f}, {d['rf_ci'][1]:.2f}]  seeds {d['rf_seed_mean']:.2f}+-{d['rf_seed_sd']:.2f}  "
          f"FE {d['fe_median']:.2f}  FEe {d['fee_median']:.2f}  util {d['util_median']:.0f}  supp {d['supp']:.2f}/{d['supc']:.2f}  t {d['time_s']:.0f}")
h2a = X["dose"]["methods"][T["best_binary_rule"]]["page_p"]
ps = {"H1": T["H1"]["p"], "H2a": h2a, "H2b": T["H2b"]["p"], "H3": T["H3_STR_vs_best"]["p"]}
order = sorted(ps, key=ps.get)
holm, run = {}, 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, ps[k] * (len(order) - i)))
    holm[k] = run
print("best binary", T["best_binary_rule"])
for k in ["H1", "H1_GradDiff", "H2b", "H3_STR_vs_best", "ABL_STR_vs_NPO_uniform", "ABL_STR_vs_ReplayOnly", "H3_noninf_FE", "H3_noninf_util"]:
    print(k, T[k])
print("Holm4", {k: round(v, 3) for k, v in holm.items()}, "raw", {k: round(v, 3) for k, v in ps.items()})
o = X["oracle"]
print("oracle n", o["n_personas"], {m: (round(v["rf_median"], 2), [round(c, 2) for c in v["rf_ci"]], round(v["fe_median"], 2), round(v["util_median"]), round(v["time_s"]))
                                    for m, v in o["methods"].items()}, o["oracle_minus_STR"])
