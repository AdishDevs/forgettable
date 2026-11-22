"""Print LaTeX rows for the pilot results table from results/summary.json."""
import json
from pathlib import Path
S = json.load(open(Path(__file__).resolve().parent / "results" / "summary.json"))
M = S["methods"]


def f(x, d=2):
    return "--" if x != x else f"{x:.{d}f}"


def ci(c):
    return "" if (c[0] != c[0]) else f" [{c[0]:.2f}, {c[1]:.2f}]"


names = {"retrain": "Retrain (reference)", "NoUpdate": "No update", "GradDiff": "GradDiff", "NPO": "NPO",
         "SimNPO": "SimNPO", "STR": "STR (proposed)"}
for m in ["retrain", "NoUpdate", "GradDiff", "NPO", "SimNPO", "STR"]:
    d = M[m]
    print(f"{names[m]} & {f(d['rf_median'])}{ci(d['rf_ci'])} & {f(d['top1_reversion'])} & {f(d['fe_sup_median'])} & "
          f"{f(d['fe_erase_median'])} & {f(d['utility_pct_of_theta'],0)} & {f(d['share_old_suppressed'])} / {f(d['share_control_suppressed'])} & {f(d['time_s'],0)}\\\\")
