"""Extended-study analysis: personas x 3 seeds, persona-level tests with Holm correction."""
import glob, json, random
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DIRS = {0: HERE / "results", 1: HERE / "results_s1", 2: HERE / "results_s2"}
BIN = ["GradDiff", "NPO", "SimNPO"]
METHODS = ["NoUpdate", "GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]

runs = {}  # (persona, seed) -> json
for sd, d in DIRS.items():
    for p in glob.glob(str(d / "persona_*.json")):
        r = json.load(open(p))
        runs[(r["persona"], sd)] = r
personas = sorted({p for p, s in runs if all((p, k) in runs for k in DIRS)})  # complete personas only
runs = {k: v for k, v in runs.items() if k[0] in personas}
need = [m for m in METHODS if all(m in runs[(p, s)]["eval"] for p in personas for s in DIRS)]

# retrain noise floor: SD across seeds of l_R(o) per superseded fact, median over facts
sds = []
for p in personas:
    for k, f in enumerate(runs[(p, 0)]["facts"]):
        if f["role"] == "superseded":
            sds.append(np.std([runs[(p, s)]["eval"]["retrain"][k]["lo"] for s in DIRS], ddof=1))
floor = float(np.median(sds))


def fact_rows(p, s):
    r = runs[(p, s)]; ev = r["eval"]; s2 = (s + 1) % 3; ev2 = runs[(p, s2)]["eval"]
    out = []
    for k, f in enumerate(r["facts"]):
        th, R = ev["theta"][k], ev["retrain"][k]
        row = {"role": f["role"], "k": k, "th": th, "R": R, "R2": ev2["retrain"][k], "ev": ev}
        if f["role"] == "superseded":
            row["learned"] = th["ln"] > th["lo"]
            row["included"] = row["learned"] and (R["lo"] - th["lo"] > 2 * floor)
        out.append(row)
    return out


def rf(row, m):
    d = row["R"]["lo"] - row["th"]["lo"]
    u = row["R2"]["lo"] if m == "RetrainIndep" else row["ev"][m][row["k"]]["lo"]
    return (u - row["th"]["lo"]) / d


def fe(row, m, key):
    d = row["th"][key] - row["R"][key]
    if abs(d) < 1e-6:
        return np.nan
    u = row["R2"][key] if m == "RetrainIndep" else row["ev"][m][row["k"]][key]
    return (row["th"][key] - u) / d


ALL = ["RetrainIndep"] + need
per = {m: {"rf": {}, "fe": {}, "fee": {}, "util": {}, "supp": {}, "supc": {}, "rf_seed": {}} for m in ALL}
counts = {"superseded": 0, "learned": 0, "included": 0, "retrain_top1": 0}
for p in personas:
    acc = {m: {k: [] for k in per[m]} for m in ALL}
    for s in DIRS:
        rows = fact_rows(p, s)
        sup = [r for r in rows if r["role"] == "superseded"]
        inc = [r for r in sup if r["included"]]
        ctl = [r for r in rows if r["role"] == "control"]
        era = [r for r in rows if r["role"] == "erase"]
        keep = [r for r in rows if r["role"] in ("unaffected", "control")]
        counts["superseded"] += len(sup); counts["learned"] += sum(r["learned"] for r in sup)
        counts["included"] += len(inc); counts["retrain_top1"] += sum(r["R"]["lo"] > r["R"]["ln"] for r in sup)
        th_util = np.mean([r["th"]["acc"] for r in keep])
        for m in ALL:
            if inc:
                acc[m]["rf"].append(np.median([rf(r, m) for r in inc]))
                acc[m]["fe"].append(np.nanmedian([fe(r, m, "ln") for r in inc]))
                acc[m]["supp"].append(np.mean([(r["R2"]["lo"] if m == "RetrainIndep" else r["ev"][m][r["k"]]["lo"]) < r["th"]["lo"] for r in inc]))
            acc[m]["fee"].append(np.nanmedian([fe(r, m, "lo") for r in era]))
            u = np.mean([(r["R2"]["acc"] if m == "RetrainIndep" else r["ev"][m][r["k"]]["acc"]) for r in keep])
            acc[m]["util"].append(100 * u / th_util if th_util > 0 else np.nan)
            acc[m]["supc"].append(np.mean([(r["R2"]["lo"] if m == "RetrainIndep" else r["ev"][m][r["k"]]["lo"]) < r["th"]["lo"] for r in ctl]))
    for m in ALL:
        for k, v in acc[m].items():
            if k != "rf_seed" and v:
                per[m][k][p] = float(np.nanmean(v))
        per[m]["rf_seed"][p] = [float(x) for x in acc[m]["rf"]]


def boot(x, n=5000):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if len(x) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(0)
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def seed_sd(m):
    """SD across seeds of the persona-median RF (each seed: median over personas)."""
    by_seed = []
    for s in range(3):
        vals = [per[m]["rf_seed"][p][s] for p in personas if len(per[m]["rf_seed"].get(p, [])) > s]
        if vals:
            by_seed.append(np.median(vals))
    return float(np.mean(by_seed)), float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else float("nan")


summary = {"n_personas": len(personas), "personas": personas, "seeds": 3, "floor": floor, "counts": counts, "methods": {}}
for m in ALL:
    rfv = [per[m]["rf"][p] for p in personas if p in per[m]["rf"]]
    mean_s, sd_s = seed_sd(m)
    summary["methods"][m] = {
        "rf_median": float(np.median(rfv)), "rf_ci": boot(rfv), "rf_seed_mean": mean_s, "rf_seed_sd": sd_s,
        "fe_median": float(np.nanmedian(list(per[m]["fe"].values()))), "fee_median": float(np.nanmedian(list(per[m]["fee"].values()))),
        "util_median": float(np.nanmedian(list(per[m]["util"].values()))), "supp": float(np.nanmean(list(per[m]["supp"].values()))),
        "supc": float(np.nanmean(list(per[m]["supc"].values()))),
        "time_s": float(np.mean([runs[(p, s)]["times"].get(m, 0.0) for p in personas for s in DIRS])) if m != "RetrainIndep"
        else float(np.mean([runs[(p, s)]["times"]["retrain_s"] for p in personas for s in DIRS])),
        "rf_persona": {str(p): per[m]["rf"].get(p) for p in personas}}


def wil(a, b, alt):
    x = np.array([per[a]["rf"][p] - per[b]["rf"][p] for p in personas if p in per[a]["rf"] and p in per[b]["rf"]])
    w = stats.wilcoxon(x, alternative=alt, zero_method="wilcox")
    n = len(x); rbc = 1 - 2 * w.statistic / (n * (n + 1) / 2) if alt == "greater" else 2 * w.statistic / (n * (n + 1) / 2) - 1
    return {"W": float(w.statistic), "p": float(w.pvalue), "n": n, "median_diff": float(np.median(x)), "r_rb": float(-rbc if alt == "less" else rbc)}


best = min(BIN, key=lambda m: abs(summary["methods"][m]["fe_median"] - 1))
tests = {"best_binary_rule": best,
         "H1": wil(best, "RetrainIndep", "less"),
         "H1_GradDiff": wil("GradDiff", "RetrainIndep", "less"),
         "H3_STR_vs_best": wil("STR", best, "greater"),
         "ABL_STR_vs_NPO_uniform": wil("STR", "NPO", "greater"),
         "ABL_STR_vs_ReplayOnly": wil("STR", "ReplayOnly", "greater") if "ReplayOnly" in need else None}
sb = np.array([per[best]["supp"][p] - per[best]["supc"][p] for p in personas if p in per[best]["supp"]])
w = stats.wilcoxon(sb, alternative="greater") if np.any(sb != 0) else None
tests["H2b"] = {"W": float(w.statistic) if w else float("nan"), "p": float(w.pvalue) if w else 1.0, "n": len(sb), "median_diff": float(np.median(sb))}
conf = ["H1", "H2b", "H3_STR_vs_best"]
ps = sorted((tests[k]["p"], k) for k in conf)
for i, (p, k) in enumerate(ps):  # Holm step-down
    tests[k]["p_holm"] = float(min(1.0, max(p * (len(ps) - j) for j, (pp, kk) in enumerate(ps[:i + 1]))))
# noninferiority of STR vs best on FE closeness and utility (margin 0.1 in |FE-1|, 5 pp utility)
fe_gap = np.array([abs(per["STR"]["fe"][p] - 1) - abs(per[best]["fe"][p] - 1) for p in personas if p in per["STR"]["fe"] and p in per[best]["fe"]])
ut_gap = np.array([per[best]["util"][p] - per["STR"]["util"][p] for p in personas])
tests["H3_noninf_FE"] = {"p": float(stats.wilcoxon(fe_gap - 0.1, alternative="less").pvalue), "median": float(np.median(fe_gap))}
tests["H3_noninf_util"] = {"p": float(stats.wilcoxon(ut_gap - 5, alternative="less").pvalue) if np.any(ut_gap - 5 != 0) else 0.0, "median": float(np.median(ut_gap))}
summary["tests"] = tests
json.dump(summary, open(HERE / "results_extended.json", "w"), indent=1)

print(f"personas {personas}  floor {floor:.3f}  counts {counts}  best(rule) {best}")
for m in ALL:
    d = summary["methods"][m]
    print(f"{m:12s} RF {d['rf_median']:.2f} [{d['rf_ci'][0]:.2f},{d['rf_ci'][1]:.2f}] seeds {d['rf_seed_mean']:.2f}±{d['rf_seed_sd']:.2f}  "
          f"FE {d['fe_median']:.2f} FEe {d['fee_median']:.2f} util {d['util_median']:.0f} supp {d['supp']:.2f}/{d['supc']:.2f} t {d['time_s']:.0f}")
for k, v in tests.items():
    print(k, v)
