"""Fig. 2: per-fact reversion fidelity, axis clipped to [-2, 2.5] with out-of-range counts."""
import json, random
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
S = json.load(open(R / "summary.json"))
METHODS = ["NoUpdate", "GradDiff", "NPO", "SimNPO", "STR"]
LABEL = {"NoUpdate": "No update", "GradDiff": "GradDiff", "NPO": "NPO", "SimNPO": "SimNPO", "STR": "STR"}
W, H = 1000, 520
x0, x1, y0, y1 = 120, 980, 70, 430
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
svg.append(f'<text x="{x1-4}" y="{Y(1)-9:.1f}" text-anchor="end" font-family="{F}" font-size="21" fill="#1b5aa6">retraining (RF = 1)</text>')
step = (x1 - x0) / len(METHODS)
rng = random.Random(0)
for i, m in enumerate(METHODS):
    cx = x0 + step * (i + 0.5)
    vals = S["methods"][m]["rf_values"]
    below = sum(v < LO for v in vals); above = sum(v > HI for v in vals)
    for v in vals:
        if LO <= v <= HI:
            svg.append(f'<circle cx="{cx + rng.uniform(-30, 30):.1f}" cy="{Y(v):.1f}" r="6" fill="#777" fill-opacity="0.7"/>')
    med = S["methods"][m]["rf_median"]
    my = Y(min(max(med, LO), HI))
    svg.append(f'<line x1="{cx-48:.1f}" y1="{my:.1f}" x2="{cx+48:.1f}" y2="{my:.1f}" stroke="#000" stroke-width="5"/>')
    if med < LO:
        svg.append(f'<text x="{cx:.1f}" y="{y1-14:.1f}" text-anchor="middle" font-family="{F}" font-size="19">median \u2212{abs(med):.1f}</text>')
    if below:
        svg.append(f'<text x="{cx:.1f}" y="{y1+26:.1f}" text-anchor="middle" font-family="{F}" font-size="19">{below} below −2 ↓</text>')
    if above:
        svg.append(f'<text x="{cx:.1f}" y="{y0-14:.1f}" text-anchor="middle" font-family="{F}" font-size="19">{above} above 2.5 ↑</text>')
    svg.append(f'<text x="{cx:.1f}" y="{y1+58:.1f}" text-anchor="middle" font-family="{F}" font-size="23" font-weight="bold">{LABEL[m]}</text>')
svg.append(f'<text x="34" y="{(y0+y1)/2:.0f}" transform="rotate(-90 34 {(y0+y1)/2:.0f})" text-anchor="middle" font-family="{F}" font-size="23">Reversion fidelity (RF)</text>')
svg.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" text-anchor="middle" font-family="{F}" font-size="23">Unlearning method</text>')
svg.append('</svg>')
s = "\n".join(svg)
open(R / "nimsa2.svg", "w").write(s)
open(R / "nimsa2_print.html", "w").write(f'<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: 7.16in {7.16*H/W:.3f}in; margin: 0; }} html,body{{margin:0}} svg{{display:block}}</style></head><body>{s}</body></html>')
print("ok")
