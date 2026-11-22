"""Fig. 2 (extended study): persona-level RF (mean over seeds) per method, axis clipped with counts."""
import json, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "results_extended.json"))
METHODS = [m for m in ["RetrainIndep", "GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"] if m in S["methods"]]
LABEL = {"RetrainIndep": "Retrain (other seed)", "GradDiff": "GradDiff", "NPO": "NPO", "SimNPO": "SimNPO",
         "STR": "STR", "ReplayOnly": "Replay only"}
W, H = 1100, 540
x0, x1, y0, y1 = 120, 1080, 70, 440
LO, HI = -2.0, 2.5
Y = lambda v: y1 - (v - LO) / (HI - LO) * (y1 - y0)
F = "Helvetica, Arial, sans-serif"
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="7.16in" height="{7.16*H/W:.3f}in" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#fff"/>']
for t in np.arange(LO, HI + 1e-9, 0.5):
    svg.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" stroke="#e2e2e2" stroke-width="1"/>')
    svg.append(f'<text x="{x0-12}" y="{Y(t)+7:.1f}" text-anchor="end" font-family="{F}" font-size="22">{t:g}</text>')
svg.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="#666" stroke-width="2"/>')
svg.append(f'<line x1="{x0}" y1="{Y(1):.1f}" x2="{x1}" y2="{Y(1):.1f}" stroke="#1b5aa6" stroke-width="2.5" stroke-dasharray="9,7"/>')
step = (x1 - x0) / len(METHODS)
rng = random.Random(0)
for i, m in enumerate(METHODS):
    cx = x0 + step * (i + 0.5)
    vals = [v for v in S["methods"][m]["rf_persona"].values() if v is not None]
    below = sum(v < LO for v in vals); above = sum(v > HI for v in vals)
    for v in vals:
        if LO <= v <= HI:
            svg.append(f'<circle cx="{cx + rng.uniform(-26, 26):.1f}" cy="{Y(v):.1f}" r="7" fill="#777" fill-opacity="0.75"/>')
    med = S["methods"][m]["rf_median"]
    svg.append(f'<line x1="{cx-44:.1f}" y1="{Y(min(max(med, LO), HI)):.1f}" x2="{cx+44:.1f}" y2="{Y(min(max(med, LO), HI)):.1f}" stroke="#000" stroke-width="5"/>')
    if med < LO:
        svg.append(f'<text x="{cx:.1f}" y="{y1-14:.1f}" text-anchor="middle" font-family="{F}" font-size="19">median −{abs(med):.1f}</text>')
    if below:
        svg.append(f'<text x="{cx:.1f}" y="{y1+26:.1f}" text-anchor="middle" font-family="{F}" font-size="19">{below} below −2 ↓</text>')
    if above:
        svg.append(f'<text x="{cx:.1f}" y="{y0-14:.1f}" text-anchor="middle" font-family="{F}" font-size="19">{above} above 2.5 ↑</text>')
    svg.append(f'<text x="{cx:.1f}" y="{y1+58:.1f}" text-anchor="middle" font-family="{F}" font-size="21" font-weight="bold">{LABEL[m]}</text>')
svg.append(f'<text x="34" y="{(y0+y1)/2:.0f}" transform="rotate(-90 34 {(y0+y1)/2:.0f})" text-anchor="middle" font-family="{F}" font-size="22">Persona-level reversion fidelity (RF)</text>')
svg.append('</svg>')
s = "\n".join(svg)
open(HERE / "results_extended_fig.svg", "w").write(s)
open(HERE / "results_extended_fig_print.html", "w").write(
    f'<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: 7.16in {7.16*H/W:.3f}in; margin: 0; }} html,body{{margin:0}} svg{{display:block}}</style></head><body>{s}</body></html>')
print("ok")
