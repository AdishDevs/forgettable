"""Oracle diagnostic (main personas x 3 seeds) and update-dose study (H2a). Writes results_extra.json."""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def boot(x, n=5000):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)])
    if len(x) < 2:
        return [float("nan")] * 2
    m = np.median(np.random.default_rng(0).choice(x, (n, len(x))), axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(dirs, oracle=False):
    runs = {}
    for sd, d in dirs.items():
        for p in glob.glob(str(HERE / d / "persona_*.json")):
            if p.endswith("_oracle.json"):
                continue
            r = json.load(open(p))
            if oracle:
                op = Path(p.replace(".json", "_oracle.json"))
                if not op.exists():
                    continue
                o = json.load(open(op)); r["eval"]["Oracle"] = o["Oracle"]; r["times"]["Oracle"] = o["time_s"]
            runs[(r["persona"], sd)] = r
    personas = sorted({p for p, s in runs if all((p, k) in runs for k in dirs)})
    return {k: v for k, v in runs.items() if k[0] in personas}, personas


def floor_of(runs, personas, seeds):
    sds = [np.std([runs[(p, s)]["eval"]["retrain"][k]["lo"] for s in seeds], ddof=1)
           for p in personas for k, f in enumerate(runs[(p, seeds[0])]["facts"]) if f["role"] == "superseded"]
    return float(np.median(sds))


def included(r, k, floor):
    th, R = r["eval"]["theta"][k], r["eval"]["retrain"][k]
    return th["ln"] > th["lo"] and (R["lo"] - th["lo"] > 2 * floor)


def rf(r, k, m):
    th, R = r["eval"]["theta"][k]["lo"], r["eval"]["retrain"][k]["lo"]
    return (r["eval"][m][k]["lo"] - th) / (R - th)


def fe(r, k, m):
    th, R = r["eval"]["theta"][k]["ln"], r["eval"]["retrain"][k]["ln"]
    return (th - r["eval"][m][k]["ln"]) / (th - R) if abs(th - R) > 1e-6 else np.nan


out = {}
# ---------------- oracle
DIRS = {0: "results", 1: "results_s1", 2: "results_s2"}
runs, personas = load(DIRS, oracle=True)
if personas:
    floor = floor_of(runs, personas, list(DIRS))
    res = {}
    for m in ["Oracle", "STR", "NPO", "SimNPO"]:
        rfp, fep, utp = [], [], []
        for p in personas:
            a, b, c = [], [], []
            for s in DIRS:
                r = runs[(p, s)]
                inc = [k for k, f in enumerate(r["facts"]) if f["role"] == "superseded" and included(r, k, floor)]
                keep = [k for k, f in enumerate(r["facts"]) if f["role"] in ("unaffected", "control")]
                if inc:
                    a.append(np.median([rf(r, k, m) for k in inc])); b.append(np.nanmedian([fe(r, k, m) for k in inc]))
                th_u = np.mean([r["eval"]["theta"][k]["acc"] for k in keep])
                c.append(100 * np.mean([r["eval"][m][k]["acc"] for k in keep]) / th_u if th_u else np.nan)
            rfp.append(np.mean(a) if a else np.nan); fep.append(np.nanmean(b) if b else np.nan); utp.append(np.nanmean(c))
        res[m] = {"rf_median": float(np.nanmedian(rfp)), "rf_ci": boot(rfp), "fe_median": float(np.nanmedian(fep)),
                  "util_median": float(np.nanmedian(utp)), "rf_persona": [float(x) for x in rfp],
                  "time_s": float(np.mean([runs[(p, s)]["times"][m] for p in personas for s in DIRS]))}
    d = np.array(res["Oracle"]["rf_persona"]) - np.array(res["STR"]["rf_persona"])
    d = d[~np.isnan(d)]
    w = stats.wilcoxon(d, alternative="two-sided") if np.any(d != 0) else None
    out["oracle"] = {"n_personas": len(personas), "floor": floor, "methods": res,
                     "oracle_minus_STR": {"median": float(np.median(d)), "W": float(w.statistic) if w else None,
                                          "p_two_sided": float(w.pvalue) if w else None}}

# ---------------- dose (H2a)
DD = {0: "results_dose_s0", 1: "results_dose_s1", 2: "results_dose_s2"}
runs, personas = load(DD)
if personas:
    floor = floor_of(runs, personas, list(DD))
    levels = [1, 2, 4]
    counts = {str(L): {"facts": 0, "retrain_top1": 0, "learned": 0, "included": 0} for L in levels}
    for (p, s), r in runs.items():
        for k, f in enumerate(r["facts"]):
            if f["role"] != "superseded":
                continue
            c = counts[str(f["r"])]; th, R = r["eval"]["theta"][k], r["eval"]["retrain"][k]
            c["facts"] += 1; c["retrain_top1"] += int(R["lo"] > R["ln"]); c["learned"] += int(th["ln"] > th["lo"])
            c["included"] += int(included(r, k, floor))
    res = {}
    for m in ["GradDiff", "NPO", "SimNPO"]:
        mat = []
        for p in personas:
            row = []
            for L in levels:
                vals = []
                for s in DD:
                    r = runs[(p, s)]
                    inc = [k for k, f in enumerate(r["facts"]) if f["role"] == "superseded" and f["r"] == L and included(r, k, floor)]
                    if inc:
                        vals.append(np.median([rf(r, k, m) for k in inc]))
                row.append(np.mean(vals) if vals else np.nan)
            mat.append(row)
        mat = np.array(mat); full = mat[~np.isnan(mat).any(axis=1)]
        pt = stats.page_trend_test(full, ranked=False, predicted_ranks=[3, 2, 1]) if len(full) >= 2 else None
        res[m] = {"rf_median_by_dose": {str(L): float(np.nanmedian(mat[:, i])) for i, L in enumerate(levels)},
                  "n_complete": int(len(full)), "page_L": float(pt.statistic) if pt else None,
                  "page_p": float(pt.pvalue) if pt else None}
    out["dose"] = {"n_personas": len(personas), "floor": floor, "counts": counts, "methods": res}

json.dump(out, open(HERE / "results_extra.json", "w"), indent=1)
print(json.dumps(out, indent=1, default=str)[:4000])
