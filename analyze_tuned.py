"""Exploratory analysis of the per-method tuned settings (Tier 2).

Runs exactly the analysis code of analyze2.py, with the evaluation of the five unlearning methods
replaced by the tuned reruns (persona_{pi}_tuned.json); theta, the retrains, the inclusion rule and
the noise floor are unchanged. Adds a greedy-decoding table on included superseded facts.
Writes results_tuned.json. All tests here are exploratory (not corrected, not pre-specified).
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
src = (HERE / "analyze2.py").read_text()
anchor = "runs = {k: v for k, v in runs.items() if k[0] in personas}\n"
assert anchor in src
patch = anchor + '''
TUNED = ["GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]
for (p_, s_), r_ in runs.items():
    t_ = json.load(open(DIRS[s_] / f"persona_{p_}_tuned.json"))
    assert t_["persona"] == p_ and t_["seed"] == s_
    for m_ in TUNED:
        r_["eval"][m_] = t_["eval"][m_]; r_["times"][m_] = t_["times"][m_]
    r_["greedy"] = t_["greedy"]; r_["settings"] = t_["settings"]
'''
src = src.replace(anchor, patch).replace('"results_extended.json"', '"results_tuned.json"')
G = {"__file__": str(HERE / "analyze2.py"), "__name__": "analyze_tuned_inner"}
exec(compile(src, "analyze2(tuned)", "exec"), G)
runs, personas, DIRS, fact_rows = G["runs"], G["personas"], G["DIRS"], G["fact_rows"]

# ---- greedy decoding on included superseded facts ----
GM = ["theta", "RetrainIndep", "retrain", "GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]


def cat(g):
    return "o" if g["o"] >= 2 else ("n" if g["n"] >= 2 else "other")


per = {m: {"o": {}, "n": {}, "agree": {}} for m in GM}
n_inc = 0
for p in personas:
    acc = {m: {"o": [], "n": [], "agree": []} for m in GM}
    for s in DIRS:
        s2 = (s + 1) % 3
        inc = [r for r in fact_rows(p, s) if r["role"] == "superseded" and r["included"]]
        n_inc += len(inc)
        if not inc:
            continue
        gen = runs[(p, s)]["greedy"]; gen2 = runs[(p, s2)]["greedy"]
        for m in GM:
            g = [gen2["retrain"][r["k"]] if m == "RetrainIndep" else gen[m][r["k"]] for r in inc]
            acc[m]["o"].append(np.mean([x["o"] / 3 for x in g]))
            acc[m]["n"].append(np.mean([x["n"] / 3 for x in g]))
            acc[m]["agree"].append(np.mean([cat(x) == cat(gen["retrain"][r["k"]]) for x, r in zip(g, inc)]))
    for m in GM:
        for k in ("o", "n", "agree"):
            if acc[m][k]:
                per[m][k][p] = float(np.mean(acc[m][k]))

greedy = {"n_included_instances": n_inc, "methods": {}, "tests_vs_retrain_indep": {}}
for m in GM:
    greedy["methods"][m] = {k: float(np.median(list(per[m][k].values()))) for k in ("o", "n", "agree")}
    greedy["methods"][m].update({k + "_mean": float(np.mean(list(per[m][k].values()))) for k in ("o", "n", "agree")})
for m in ["GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]:
    x = np.array([per[m]["o"][p] - per["RetrainIndep"]["o"][p] for p in personas if p in per[m]["o"]])
    nz = x[x != 0]
    w = stats.wilcoxon(nz, alternative="less") if len(nz) else None
    greedy["tests_vs_retrain_indep"][m] = {"median_diff": float(np.median(x)), "mean_diff": float(np.mean(x)),
                                           "W": float(w.statistic) if w else None, "p": float(w.pvalue) if w else None}

out = json.load(open(HERE / "results_tuned.json"))
out["settings"] = runs[(personas[0], 0)]["settings"]
out["greedy"] = greedy
json.dump(out, open(HERE / "results_tuned.json", "w"), indent=1)
print("\nGREEDY (included superseded facts; share of probes answering o / n; agreement with R_s category)")
for m in GM:
    d = greedy["methods"][m]
    print(f"{m:12s} o {d['o']:.2f} ({d['o_mean']:.2f})  n {d['n']:.2f} ({d['n_mean']:.2f})  agree {d['agree']:.2f} ({d['agree_mean']:.2f})")
for m, v in greedy["tests_vs_retrain_indep"].items():
    print("o-share vs RetrainIndep", m, v)
