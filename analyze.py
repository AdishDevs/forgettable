"""Aggregate pilot results into metrics, exploratory tests and a figure (SVG)."""
import glob, json, random, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE / "results"
METHODS = ["NoUpdate", "GradDiff", "NPO", "SimNPO", "STR"]
runs = [json.load(open(p)) for p in sorted(glob.glob(str(R / "persona_*.json")))]
if not runs:
    sys.exit("no results yet")


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


facts_all = []  # one row per (persona, fact)
for r in runs:
    ev = r["eval"]
    for k, f in enumerate(r["facts"]):
        row = {"persona": r["persona"], "slot": f["slot"], "role": f["role"]}
        for m in ["theta", "retrain"] + METHODS:
            row[m] = ev[m][k]
        facts_all.append(row)

sup = [f for f in facts_all if f["role"] == "superseded"]
learned = [f for f in sup if f["theta"]["ln"] > f["theta"]["lo"]]
reverts = [f for f in learned if f["retrain"]["lo"] > f["theta"]["lo"]]
included = reverts  # Eq. (4) with sigma_floor unavailable (one seed): positive retrain recovery
top1_retrain = sum(f["retrain"]["lo"] > f["retrain"]["ln"] for f in sup)


def rf(f, m):
    return (f[m]["lo"] - f["theta"]["lo"]) / (f["retrain"]["lo"] - f["theta"]["lo"])


def fe_sup(f, m):
    d = f["theta"]["ln"] - f["retrain"]["ln"]
    return (f["theta"]["ln"] - f[m]["ln"]) / d if abs(d) > 1e-6 else float("nan")


erase = [f for f in facts_all if f["role"] == "erase"]


def fe_erase(f, m):
    d = f["theta"]["lo"] - f["retrain"]["lo"]
    return (f["theta"]["lo"] - f[m]["lo"]) / d if abs(d) > 1e-6 else float("nan")


keep = [f for f in facts_all if f["role"] in ("unaffected", "control")]
controls = [f for f in facts_all if f["role"] == "control"]
summary = {"n_personas": len(runs), "n_superseded": len(sup), "n_learned": len(learned),
           "n_included": len(included), "retrain_top1_old": top1_retrain,
           "theta_recall": float(np.mean([f["theta"]["acc"] for f in facts_all if f["role"] != "superseded"] +
                                         [int(f["theta"]["ln"] > f["theta"]["lo"]) for f in sup])),
           "methods": {}}
theta_util = np.mean([f["theta"]["acc"] for f in keep])
for m in METHODS + ["retrain"]:
    rfs = [rf(f, m) for f in included]
    fes = [x for x in (fe_sup(f, m) for f in included) if not np.isnan(x)]
    fee = [x for x in (fe_erase(f, m) for f in erase) if not np.isnan(x)]
    util = np.mean([f[m]["acc"] for f in keep])
    top1 = np.mean([f[m]["lo"] > f[m]["ln"] for f in included]) if included else float("nan")
    supp_sup = np.mean([f[m]["lo"] < f["theta"]["lo"] for f in included]) if included else float("nan")
    supp_ctl = np.mean([f[m]["lo"] < f["theta"]["lo"] for f in controls]) if controls else float("nan")
    summary["methods"][m] = {
        "rf_median": float(np.median(rfs)) if rfs else float("nan"), "rf_ci": boot_ci(rfs),
        "fe_sup_median": float(np.median(fes)) if fes else float("nan"), "fe_sup_ci": boot_ci(fes),
        "fe_erase_median": float(np.median(fee)) if fee else float("nan"), "fe_erase_ci": boot_ci(fee),
        "utility_pct_of_theta": float(100 * util / theta_util) if theta_util > 0 else float("nan"),
        "top1_reversion": float(top1), "share_old_suppressed": float(supp_sup), "share_control_suppressed": float(supp_ctl),
        "time_s": float(np.mean([r["times"].get(m, r["times"]["retrain_s"] if m == "retrain" else 0.0) for r in runs])),
        "rf_values": [float(x) for x in rfs]}

# exploratory fact-level tests (not confirmatory: one seed, few personas)
tests = {}
binary = ["GradDiff", "NPO", "SimNPO"]
if len(included) >= 5:
    best = max(binary, key=lambda m: summary["methods"][m]["fe_sup_median"])
    x = np.array([rf(f, best) for f in included])
    w = stats.wilcoxon(x - 1.0, alternative="less")
    tests["H1_best_binary_vs_1"] = {"method": best, "stat": float(w.statistic), "p": float(w.pvalue), "n": len(x)}
    s = np.array([rf(f, "STR") for f in included]); n_ = np.array([rf(f, "NPO") for f in included])
    w2 = stats.wilcoxon(s - n_, alternative="greater")
    tests["H3_STR_vs_NPO"] = {"stat": float(w2.statistic), "p": float(w2.pvalue), "n": len(s),
                              "median_diff": float(np.median(s - n_))}
summary["tests"] = tests
summary["config"] = runs[0]["config"]
summary["mean_theta_train_s"] = float(np.mean([r["times"]["theta_train_s"] for r in runs]))
summary["mean_retrain_s"] = float(np.mean([r["times"]["retrain_s"] for r in runs]))
json.dump(summary, open(R / "summary.json", "w"), indent=1)

# ---------------------------------------------------------------- figure: RF per fact, per method
W, H = 1000, 560
x0, x1, y0, y1 = 150, 970, 40, 470
allv = [v for m in METHODS for v in summary["methods"][m]["rf_values"]] + [0, 1]
lo_, hi_ = min(-0.5, min(allv) - 0.1), max(1.5, max(allv) + 0.1)
Y = lambda v: y1 - (v - lo_) / (hi_ - lo_) * (y1 - y0)
F = "Helvetica, Arial, sans-serif"
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="7.16in" height="{7.16*H/W:.3f}in" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#fff"/>']
for t in np.arange(np.ceil(lo_ * 2) / 2, hi_ + 1e-9, 0.5):
    svg.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" stroke="#ddd" stroke-width="1"/>')
    svg.append(f'<text x="{x0-10}" y="{Y(t)+6:.1f}" text-anchor="end" font-family="{F}" font-size="17">{t:g}</text>')
svg.append(f'<line x1="{x0}" y1="{Y(1):.1f}" x2="{x1}" y2="{Y(1):.1f}" stroke="#1b5aa6" stroke-width="2" stroke-dasharray="8,6"/>')
svg.append(f'<text x="{x1}" y="{Y(1)-8:.1f}" text-anchor="end" font-family="{F}" font-size="16" fill="#1b5aa6">retraining (RF = 1)</text>')
svg.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="#555" stroke-width="1.5"/>')
step = (x1 - x0) / len(METHODS)
rng = random.Random(0)
for i, m in enumerate(METHODS):
    cx = x0 + step * (i + 0.5)
    for v in summary["methods"][m]["rf_values"]:
        svg.append(f'<circle cx="{cx + rng.uniform(-28, 28):.1f}" cy="{Y(v):.1f}" r="5" fill="#888" fill-opacity="0.65"/>')
    med = summary["methods"][m]["rf_median"]
    if not np.isnan(med):
        svg.append(f'<line x1="{cx-45:.1f}" y1="{Y(med):.1f}" x2="{cx+45:.1f}" y2="{Y(med):.1f}" stroke="#000" stroke-width="4"/>')
    svg.append(f'<text x="{cx:.1f}" y="{y1+30}" text-anchor="middle" font-family="{F}" font-size="18">{m}</text>')
svg.append(f'<text x="30" y="{(y0+y1)/2:.0f}" transform="rotate(-90 30 {(y0+y1)/2:.0f})" text-anchor="middle" font-family="{F}" font-size="18">Reversion fidelity (RF)</text>')
svg.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-12}" text-anchor="middle" font-family="{F}" font-size="18">Unlearning method</text>')
svg.append('</svg>')
open(R / "fig_rf.svg", "w").write("\n".join(svg))
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: 7.16in {7.16*H/W:.3f}in; margin: 0; }} html,body{{margin:0}} svg{{display:block}}</style></head><body>{''.join(svg)}</body></html>"""
open(R / "fig_rf_print.html", "w").write(html)

print(json.dumps({k: v for k, v in summary.items() if k != "methods"}, indent=1))
for m, d in summary["methods"].items():
    print(f"{m:9s} RF {d['rf_median']:.2f} [{d['rf_ci'][0]:.2f},{d['rf_ci'][1]:.2f}]  FEsup {d['fe_sup_median']:.2f}  FEerase {d['fe_erase_median']:.2f}  "
          f"util {d['utility_pct_of_theta']:.0f}%  top1rev {d['top1_reversion']:.2f}  supp_old {d['share_old_suppressed']:.2f} supp_ctl {d['share_control_suppressed']:.2f}  t {d['time_s']:.0f}s")
