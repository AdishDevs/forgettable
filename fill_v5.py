"""Fill forgettable_access_v5.tex from results_extended.json (extended evaluation)."""
import json, re
from pathlib import Path
FYP = Path(__file__).resolve().parent.parent
S = json.load(open(FYP / "pilot/results_extended.json"))
M, T, C = S["methods"], S["tests"], S["counts"]
NP = S["n_personas"]
WORDS = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
NPw = WORDS.get(NP, str(NP))
NEG = "\\textminus{}"


def n(x, d=2):
    t = f"{x:.{d}f}"
    return (NEG + t[1:]) if t.startswith("-") else t


def ci(c):
    return f"[{n(c[0])}, {n(c[1])}]"


def pv(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


best = T["best_binary_rule"]
names = {"RetrainIndep": "Retrain (other seed)", "NoUpdate": "No update", "GradDiff": "GradDiff", "NPO": "NPO (uniform replay)",
         "SimNPO": "SimNPO", "STR": "STR (proposed)", "ReplayOnly": "Replay only (ablation)"}
order = [m for m in ["RetrainIndep", "NoUpdate", "GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"] if m in M]
rows = []
for m in order:
    d = M[m]
    rows.append(f"{names[m]} & {n(d['rf_median'])} {ci(d['rf_ci'])} & {n(d['rf_seed_mean'])} $\\pm$ {n(d['rf_seed_sd'])} & "
                f"{n(d['fe_median'])} & {n(d['fee_median'])} & {d['util_median']:.0f} & {n(d['supp'])} / {n(d['supc'])} & {d['time_s']:.0f}\\\\")


def sigrow(label, t, holm=True):
    ph = pv(t["p_holm"]) if holm and "p_holm" in t else "--"
    return f"{label} & {t['W']:.0f} & {pv(t['p'])} & {ph} & {n(t['median_diff'])}\\\\"


sig = [sigrow(f"H1: {best} $<$ retrain (other seed)", T["H1"]),
       f"H2b: {best} suppresses superseded $>$ controls & {T['H2b']['W']:.0f} & {pv(T['H2b']['p'])} & {pv(T['H2b']['p_holm'])} & {n(T['H2b']['median_diff'])}\\\\",
       sigrow(f"H3: STR $>$ {best}", T["H3_STR_vs_best"]),
       sigrow("Exploratory: GradDiff $<$ retrain (other seed)", T["H1_GradDiff"], holm=False)]
g, npo, sn, st, ri = M["GradDiff"], M["NPO"], M["SimNPO"], M["STR"], M["RetrainIndep"]
ro = M.get("ReplayOnly")
sup_txt = lambda h: "supported" if T[h]["p_holm"] < 0.05 else "not supported"
gd_word = "failed to restore it" if g["rf_median"] < 0.5 else "largely restored it"

vals = {}
vals["NP"] = NPw
vals["FLOOR"] = f"{S['floor']:.3f}"
vals["ABSRES"] = (f"Retraining made the earlier value the most probable answer for {C['retrain_top1']} of {C['superseded']} superseded-fact instances. "
                  f"Gradient difference {gd_word}, with a median persona-level reversion fidelity of {n(g['rf_median'])}, whereas negative preference "
                  f"optimization, its simplified variant, and supersession-targeted replay reached {n(npo['rf_median'])}, {n(sn['rf_median'])}, and {n(st['rf_median'])}.")
vals["SANITY"] = (f"Across {NPw} personas and three seeds ({C['superseded']} superseded-fact instances), the trained adapters preferred the update "
                  f"for {C['learned']} instances, and retraining made the earlier value the most probable answer for {C['retrain_top1']}, so the benchmark "
                  f"produces reversion as designed. With $\\sigma_{{\\mathrm{{floor}}}} = {S['floor']:.3f}$, {C['included']} instances met the inclusion rule. "
                  f"An independently seeded retrain scored a median RF of {n(ri['rf_median'])} {ci(ri['rf_ci'])} against the reference, which shows the "
                  f"run-to-run variation that any method must be compared with.")
vals["MAINTEXT"] = (f"Gradient difference reached a median RF of {n(g['rf_median'])} {ci(g['rf_ci'])} and lowered the earlier value below its "
                    f"pre-deletion probability for {100*g['supp']:.0f}\\% of included facts on average, against {100*g['supc']:.0f}\\% of matched controls. "
                    f"NPO reached {n(npo['rf_median'])} {ci(npo['rf_ci'])}, SimNPO {n(sn['rf_median'])} {ci(sn['rf_ci'])}, and STR {n(st['rf_median'])} "
                    f"{ci(st['rf_ci'])}. Across seeds, the per-seed medians varied by an SD of {n(g['rf_seed_sd'])} for gradient difference and "
                    f"{n(npo['rf_seed_sd'])}, {n(sn['rf_seed_sd'])} and {n(st['rf_seed_sd'])} for NPO, SimNPO and STR. Forgetting efficacy was "
                    f"{n(g['fe_median'])} for gradient difference, {n(npo['fe_median'])} for NPO, {n(sn['fe_median'])} for SimNPO and {n(st['fe_median'])} "
                    f"for STR, where values above 1 mean the deleted value was pushed further down than retraining does. Utility, measured on unaffected "
                    f"and control facts, ranged from {min(M[m]['util_median'] for m in ['GradDiff','NPO','SimNPO','STR']):.0f}\\% to "
                    f"{max(M[m]['util_median'] for m in ['GradDiff','NPO','SimNPO','STR']):.0f}\\% of the adapter before deletion, so it did not "
                    f"separate the methods clearly.")
vals["TESTS"] = (f"Table~\\ref{{tab:sig}} lists the tests. In an earlier single-seed pilot on personas 0--5, selecting the binary method by the highest "
                 f"forgetting efficacy chose gradient difference, the method that over-forgets most; the selection rule was therefore amended to "
                 f"choose the binary method whose forgetting efficacy is closest to 1 (Section~\\ref{{sec:setup}}). Under the amended rule the selected "
                 f"method is {best}. H1 ({best} has lower reversion fidelity than an independently seeded retrain) is {sup_txt('H1')} "
                 f"($p_{{\\mathrm{{Holm}}}} = {T['H1']['p_holm']:.3f}$). H2b is {sup_txt('H2b')} ($p_{{\\mathrm{{Holm}}}} = {T['H2b']['p_holm']:.3f}$), and "
                 f"H3 (STR has higher reversion fidelity than {best}) is {sup_txt('H3_STR_vs_best')} ($p_{{\\mathrm{{Holm}}}} = {T['H3_STR_vs_best']['p_holm']:.3f}$). "
                 f"For gradient difference, the exploratory comparison with an independently seeded retrain gives $p = {T['H1_GradDiff']['p']:.3f}$ "
                 f"(median paired difference {n(T['H1_GradDiff']['median_diff'])}). With {NPw} personas the tests have limited power, so a result that "
                 f"is not supported is not evidence of no effect.")
abl = T["ABL_STR_vs_NPO_uniform"]
abl_txt = (f"Because NPO and STR use the same number of steps and differ only in how retained sessions are sampled, NPO is the uniform-replay "
           f"ablation of STR at an equal step budget. STR versus NPO: median paired difference in persona-level RF {n(abl['median_diff'])} "
           f"($W = {abl['W']:.0f}$, $p = {pv(abl['p'])}$, exploratory).")
if T.get("ABL_STR_vs_ReplayOnly"):
    a2 = T["ABL_STR_vs_ReplayOnly"]
    abl_txt += (f" STR versus replay only (no forget loss): median paired difference {n(a2['median_diff'])} ($W = {a2['W']:.0f}$, $p = {pv(a2['p'])}$); "
                f"replay only reached a median RF of {n(ro['rf_median'])} and a forgetting efficacy of {n(ro['fe_median'])}, which shows how much of "
                f"the behavior comes from replay alone.")
vals["ABLTEXT"] = abl_txt
vals["COSTTEXT"] = (f"Averaged over all runs, retraining took {M['RetrainIndep']['time_s']:.0f}~s and an unlearning run took between "
                    f"{min(M[m]['time_s'] for m in ['GradDiff','NPO','SimNPO','STR']):.0f}~s and {max(M[m]['time_s'] for m in ['GradDiff','NPO','SimNPO','STR']):.0f}~s. "
                    f"Most runs shared the GPU with two other runs, so absolute times are inflated, but the comparison holds: at 22 sessions per persona, "
                    f"approximate unlearning with $T = 40$ steps is not cheaper than retraining. Its cost is fixed by $T$, whereas the cost of retraining "
                    f"grows with the retained history, so the crossover point remains to be measured on longer histories.")
vals["DISCUSSION"] = (f"The results suggest that whether reversion is reproduced depends on the objective. Gradient difference lowers the probability "
                      f"of deleted answers without a reference point; because the earlier and the deleted value answer the same question, the earlier "
                      f"value was often pushed down as well, as predicted from correlated-knowledge damage \\cite{{wei}}. The preference-based objectives "
                      f"bound the update relative to the original adapter, and replaying retained sessions, which contain the earlier value, restores it. "
                      f"Forgetting efficacy shows the trade-off: a method can restore the earlier value yet push the deleted value far below where "
                      f"retraining leaves it, which a forget-quality score alone would reward. The independently seeded retrain shows how much reversion "
                      f"fidelity varies between training runs, which is why every comparison is made against it rather than against the ideal value of 1.")
vals["CONCLRES"] = (f"Over {NPw} personas and three seeds, retraining restored the earlier value for {C['retrain_top1']} of {C['superseded']} superseded-fact "
                    f"instances; gradient difference reached a median reversion fidelity of {n(g['rf_median'])}, and NPO, SimNPO and STR reached "
                    f"{n(npo['rf_median'])}, {n(sn['rf_median'])} and {n(st['rf_median'])}. Under Holm correction, H1 was {sup_txt('H1')}, H2b was "
                    f"{sup_txt('H2b')} and H3 was {sup_txt('H3_STR_vs_best')}.")
vals["TABLEROWS"] = "\n".join(rows)
vals["SIGROWS"] = "\n".join(sig)

p = FYP / "forgettable_access_v5.tex"
s = open(p, encoding="utf-8").read()
tmpl = open(FYP / "pilot/experiments_section_template.tex", encoding="utf-8").read()
i = s.index("% =====================================================================\n\\section{Pilot Study}")
j = s.index("% =====================================================================\n\\section{Discussion}")
s = s[:i] + tmpl + "\n" + s[j:]


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


rep("We specify an evaluation protocol and report a pilot study on six personas with a 0.5-billion-parameter instruction-tuned model and low-rank adaptation. The pilot compares",
    "We specify an evaluation protocol and apply it to @@NP@@ personas with three training seeds each, using a 0.5-billion-parameter instruction-tuned model and low-rank adaptation. The evaluation compares")
a0 = s.index("against full retraining."[:0] + "had superseded.", s.index("\\begin{abstract}"))
abs_end = s.index("The benchmark, metric, and code run on one laptop-class computer", a0)
s = s[:a0 + len("had superseded.")] + " @@ABSRES@@ " + s[abs_end:]
rep("\\item A pilot study on six personas that exercises the complete pipeline, checks that the benchmark produces reversion under retraining, and gives first estimates of reversion fidelity, forgetting efficacy and cost for every method (Section~\\ref{sec:results}).",
    "\\item An evaluation on @@NP@@ personas with three seeds that checks that the benchmark produces reversion under retraining, compares every method with an independently seeded retrain using persona-level tests, and ablates STR (Section~\\ref{sec:results}).")
rep("Section~\\ref{sec:results} reports the pilot study,", "Section~\\ref{sec:results} reports the experiments,")
k = s.index("\\subsection{Interpretation of the Pilot}"); k2 = s.index("\\subsection{Limitations}")
s = s[:k] + "\\subsection{Interpretation}\n@@DISCUSSION@@\n\n" + s[k2:]
rep("The pilot covers only six personas with one seed and fixed hyperparameters, so its numbers are estimates for planning, not confirmatory results.",
    "The evaluation covers only @@NP@@ personas with three seeds and fixed hyperparameters, and the hypotheses were specified in advance but not formally registered, so its tests have limited power and are not fully confirmatory.")
c0 = s.index("(4) we ran a pilot on six personas.")
c1 = s.index("The next step is the registered full study", c0)
s = s[:c0] + "(4) we evaluated them on @@NP@@ personas with three seeds. @@CONCLRES@@ " + s[c1:]
for a_, b_ in [
    ("reports a pilot that runs the same pipeline at reduced scale and states every deviation from this protocol.", "reports an evaluation that runs this protocol at reduced scale and states every deviation from it."),
    ("the pilot in Section~\\ref{sec:results} uses a reduced generator", "the evaluation in Section~\\ref{sec:results} uses a reduced generator"),
    ("those used in the pilot (Section~\\ref{sec:results})", "those used in the evaluation of Section~\\ref{sec:results}"),
    ("the pilot (Section~\\ref{sec:pilot-tests}) showed", "an earlier single-seed pilot (Section~\\ref{sec:pilot-tests}) showed"),
    ("The pilot uses the fixed values in Table~\\ref{tab:hyper} without a search;", "The evaluation in Section~\\ref{sec:results} uses the fixed values in Table~\\ref{tab:hyper} without a search;"),
    ("(the pilot in Section~\\ref{sec:results} provides the first variance estimate)", "(the evaluation in Section~\\ref{sec:results} provides the first variance estimate)"),
    ("The pilot measured 36~s per adapter training run and 36~s per unlearning run on the hardware above (Section~\\ref{sec:pilot-cost}).", "A single-process pilot run measured 36~s per adapter training run and 36~s per unlearning run on the hardware above."),
    ("the generated pilot personas", "the generated personas"),
    ("The pilot code (", "The code ("),
]:
    assert s.count(a_) == 1, (s.count(a_), a_[:60])
    s = s.replace(a_, b_)
for key, v in vals.items():
    s = s.replace(f"@@{key}@@", v)
left = re.findall(r"@@[A-Z_]+@@", s)
assert not left, left
s = s.replace(" six personas", f" {NPw} personas") if NPw != "six" else s
open(p, "w", encoding="utf-8").write(s)
body = re.sub(r"(?<!\\)%.*", "", s)
from collections import Counter
b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", body)); e = Counter(re.findall(r"\\end\{(\w+\*?)\}", body))
labels = set(re.findall(r"\\label\{([^}]+)\}", body)); refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
cites = [x.strip() for c in re.findall(r"\\cite\{([^}]+)\}", body) for x in c.split(",")]
order2 = []; [order2.append(x) for x in cites if x not in order2]
a = s.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
print("braces", body.count("{") - body.count("}"), "| env mismatch", {x: (b[x], e[x]) for x in set(b) | set(e) if b[x] != e[x]},
      "| undefined refs", refs - labels, "| bib order", order2 == re.findall(r"\\bibitem\{([^}]+)\}", body),
      "| abstract words", len(a.replace(NEG, "-").split()), "| 'pilot' mentions", len(re.findall(r"\bpilot\b", s)))
