"""Exploratory analyses added in v7 (post hoc, not corrected; no new training).

1. Seed-matched comparison. In the main analysis each unlearned adapter starts from theta of
   the same seed s as the reference R_s, whereas the comparator is the retrain of seed s+1.
   Here each method is also taken from seed s+1 and scored against R_s (with theta_s as the
   baseline), exactly as the comparator is, so method and comparator are exchangeable.
2. Equivalence (TOST) of each method with the independently seeded retrain, margin 0.1 in
   persona-level RF, with paired Wilcoxon signed-rank tests.
Writes results_sensitivity.json. Reuses the loading and inclusion rule of analyze2.py.
"""
import json
import numpy as np
from scipy import stats
import analyze2 as A

MARGIN = 0.1
METHODS = ["GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]


def rf_cross(row, m, p, s2):
    d = row["R"]["lo"] - row["th"]["lo"]
    u = A.runs[(p, s2)]["eval"][m][row["k"]]["lo"]
    return (u - row["th"]["lo"]) / d


cross = {m: {} for m in METHODS}
for p in A.personas:
    acc = {m: [] for m in METHODS}
    for s in A.DIRS:
        s2 = (s + 1) % 3
        inc = [r for r in A.fact_rows(p, s) if r["role"] == "superseded" and r["included"]]
        if inc:
            for m in METHODS:
                acc[m].append(np.median([rf_cross(r, m, p, s2) for r in inc]))
    for m in METHODS:
        if acc[m]:
            cross[m][p] = float(np.mean(acc[m]))

ref = A.per["RetrainIndep"]["rf"]


def paired(a, b):
    return np.array([a[p] - b[p] for p in A.personas if p in a and p in b])


def wil(x, alt):
    x = x[x != 0]
    return stats.wilcoxon(x, alternative=alt) if len(x) else None


def tost(d):
    lo, hi = wil(d + MARGIN, "greater"), wil(d - MARGIN, "less")
    p_lo = float(lo.pvalue) if lo else 0.0
    p_hi = float(hi.pvalue) if hi else 0.0
    return {"n": int(len(d)), "median_diff": float(np.median(d)), "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": max(p_lo, p_hi), "equivalent_at_0.05": max(p_lo, p_hi) < 0.05}


out = {"margin": MARGIN, "n_personas": len(A.personas), "seed_matched": {}, "tost_main": {}, "tost_seed_matched": {}}
for m in METHODS:
    v = list(cross[m].values())
    d_main = paired(A.per[m]["rf"], ref)
    d_x = paired(cross[m], ref)
    w = wil(d_x, "less")
    out["seed_matched"][m] = {"rf_median": float(np.median(v)), "rf_ci": A.boot(v),
                              "vs_retrain_less": {"W": float(w.statistic), "p": float(w.pvalue),
                                                  "median_diff": float(np.median(d_x))} if w else None}
    out["tost_main"][m] = tost(d_main)
    out["tost_seed_matched"][m] = tost(d_x)
out["retrain_indep_rf_median"] = float(np.median(list(ref.values())))
json.dump(out, open(A.HERE / "results_sensitivity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
