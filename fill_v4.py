"""Insert pilot results from results/summary.json into forgettable_access_v4.tex (one-time fill)."""
import json, re
from pathlib import Path
FYP = Path(__file__).resolve().parent.parent
S = json.load(open(FYP / "pilot/results/summary.json"))
M = S["methods"]
NEG = "\\textminus{}"


def num(x, d=2):
    t = f"{x:.{d}f}"
    return (NEG + t[1:]) if t.startswith("-") else t


def ci(c):
    return f" [{num(c[0])}, {num(c[1])}]"


names = {"retrain": "Retrain (reference)", "NoUpdate": "No update", "GradDiff": "GradDiff", "NPO": "NPO",
         "SimNPO": "SimNPO", "STR": "STR (proposed)"}
rows = []
for m in ["retrain", "NoUpdate", "GradDiff", "NPO", "SimNPO", "STR"]:
    d = M[m]
    rows.append(f"{names[m]} & {num(d['rf_median'])}{ci(d['rf_ci'])} & {num(d['top1_reversion'])} & {num(d['fe_sup_median'])} & "
                f"{num(d['fe_erase_median'])} & {d['utility_pct_of_theta']:.0f} & {num(d['share_old_suppressed'])} / "
                f"{num(d['share_control_suppressed'])} & {d['time_s']:.0f}\\\\")
T = S["tests"]
assert S["n_personas"] == 6 and S["n_included"] == 14 and S["n_superseded"] == 24 and S["n_learned"] == 15 and S["retrain_top1_old"] == 24
g, n, sn, st = M["GradDiff"], M["NPO"], M["SimNPO"], M["STR"]
neg = {m: sum(v < 0 for v in M[m]["rf_values"]) for m in M}
supp_g = round(g["share_old_suppressed"] * 14)
ttrain, tret, tunl = round(S["mean_theta_train_s"]), round(S["mean_retrain_s"]), round(M["NPO"]["time_s"])
gmed = f"{NEG}{abs(g['rf_median']):.1f}"
h1, h3 = T["H1_best_binary_vs_1"], T["H3_STR_vs_NPO"]
vals = {
    "NP": "6", "NINC": "14", "TTRAIN": str(ttrain), "TRETRAIN": str(tret), "TUNL": str(tunl),
    "ABSRES": (f"In the pilot, retraining made the earlier value the most probable answer for all 24 superseded facts. "
               f"Gradient difference instead suppressed it, with a median reversion fidelity of {gmed} over 14 included facts, "
               f"whereas the preference-based methods, including supersession-targeted replay, stayed close to retraining, "
               f"with medians from {n['rf_median']:.2f} to {st['rf_median']:.2f}."),
    "SANITY": (f"Before deletion, the adapters recalled {100*S['theta_recall']:.0f}\\% of facts (for superseded facts, recall means "
               f"preferring the update), and they preferred the update for {S['n_learned']} of the 24 superseded facts. Retraining "
               f"without the deleted sessions made the earlier value the most probable answer for all {S['retrain_top1_old']} "
               f"superseded facts, so the benchmark produces reversion as designed. Fourteen superseded facts met the inclusion "
               f"rule; nine were excluded because the update had not been learned and one because retraining did not raise the earlier value."),
    "MAINTEXT": (f"Gradient difference did not reproduce reversion: its median RF was {gmed}, and it pushed the earlier value below "
                 f"its pre-deletion probability for {supp_g} of the 14 included facts ({100*g['share_old_suppressed']:.0f}\\%). "
                 f"NPO reached a median RF of {n['rf_median']:.2f}, but with a wide interval and {neg['NPO']} facts below zero, whereas "
                 f"SimNPO ({sn['rf_median']:.2f}) and STR ({st['rf_median']:.2f}) had no fact below zero. Every method that updated "
                 f"the adapter made the earlier value more probable than the deleted one (top-1 reversion of 1.00). Forgetting "
                 f"efficacy shows the other side of this behavior: gradient difference, NPO and STR pushed the deleted value far "
                 f"below the level that retraining leaves (FE of {g['fe_sup_median']:.2f}, {n['fe_sup_median']:.2f} and "
                 f"{st['fe_sup_median']:.2f}), whereas SimNPO stayed close to retraining ({sn['fe_sup_median']:.2f}). Utility did not "
                 f"discriminate between methods: every updating method except gradient difference reached the utility of retraining, "
                 f"{M['retrain']['utility_pct_of_theta']:.0f}\\% of the adapter before deletion, because replaying retained sessions "
                 f"also reinforces unaffected facts."),
    "TESTS": (f"Fact-level one-sided Wilcoxon signed-rank tests on the 14 included facts are exploratory, because facts from the "
              f"same persona are not independent and only one seed was run. Under the selection rule as first written, which picks "
              f"the binary method with the highest forgetting efficacy, the selected method was gradient difference, and its RF was "
              f"below 1 ($W = {h1['stat']:.0f}$, $p = {h1['p']:.3f}$). Because that rule rewards over-forgetting, the protocol now "
              f"selects the binary method whose forgetting efficacy is closest to 1 (Section~\\ref{{sec:setup}}); in the pilot this "
              f"would be SimNPO. STR had higher RF than NPO ($W = {h3['stat']:.0f}$, $p = {h3['p']:.3f}$), but the median paired "
              f"difference was only {h3['median_diff']:.3f}, which is not practically meaningful. The comparison with matched "
              f"controls (H2b) is not yet interpretable, because retraining itself lowered the earlier value for half of the controls "
              f"(Table~\\ref{{tab:pilot}}), a level of run-to-run variation that the seed floor of the full study is designed to absorb."),
    "DISCUSSION": (f"The pilot suggests that the failure to reproduce reversion depends on the objective rather than being universal. "
                   f"Gradient difference lowers the probability of deleted answers without any reference point, and because the "
                   f"earlier and the deleted value answer the same question, pushing one down also pushed the other down in most "
                   f"included facts, as predicted from correlated-knowledge damage \\cite{{wei}}; its worst fact reached an RF of "
                   f"{NEG}{abs(min(g['rf_values'])):.0f}. The preference-based objectives bound the update relative to the original "
                   f"adapter, and replaying retained sessions, which contain the earlier value, restores it. The trade-off appears in "
                   f"forgetting efficacy: NPO and STR restore the earlier value but over-forget the deleted one, which a forget-quality "
                   f"score alone would reward, whereas SimNPO matched retraining on both criteria in this pilot. Targeting the replay "
                   f"toward superseded sessions removed the negative RF values that NPO produced, but its median gain was negligible at "
                   f"this scale. These observations come from 14 facts and one seed, so they motivate rather than establish the "
                   f"hypotheses of the full study."),
    "CONCLRES": (f"Retraining restored the earlier value for all 24 superseded facts. Gradient difference suppressed it (median "
                 f"reversion fidelity {gmed} over 14 included facts), whereas the preference-based methods, including STR, stayed "
                 f"close to retraining, and SimNPO also matched retraining in forgetting efficacy. At pilot scale, approximate "
                 f"unlearning was not cheaper than retraining, and the pilot led us to amend the method-selection rule."),
    "TABLEROWS": "\n".join(rows),
}
tmpl = open(FYP / "pilot/pilot_section_template.tex", encoding="utf-8").read()
i = tmpl.index("Training one persona adapter took")
tmpl = tmpl[:i] + (f"Training one persona adapter took {ttrain}~s on average and retraining the reference took {tret}~s, whereas one "
                   f"unlearning run took {tunl}~s. At the pilot's size of 22 sessions per persona, approximate unlearning was therefore "
                   f"about twice as slow as retraining: its cost is set by the fixed number of steps $T$, whereas the cost of retraining "
                   f"grows with the length of the retained history. The pilot does not show at what history length approximate "
                   f"unlearning becomes cheaper than retraining; the full study measures this.\n")
p = FYP / "forgettable_access_v4.tex"
s = open(p, encoding="utf-8").read()
s = s.replace("@@PILOT_SECTION@@", tmpl)


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


rep("We specify a registered evaluation protocol and report a pilot study on @@NP@@ personas with a 0.5-billion-parameter instruction-tuned model and low-rank adaptation, comparing gradient difference, negative preference optimization, its simplified variant, and supersession-targeted replay, a minimal change that up-weights the retained conversations that a deleted conversation had superseded, against full retraining.",
    "We specify an evaluation protocol and report a pilot study on @@NP@@ personas with a 0.5-billion-parameter instruction-tuned model and low-rank adaptation. The pilot compares full retraining with gradient difference, negative preference optimization, its simplified variant, and supersession-targeted replay, a minimal change that up-weights the retained conversations that a deleted conversation had superseded.")
rep("with a method-independent inclusion rule (Section~\\ref{sec:problem}), used to test whether standard unlearning objectives reproduce reversion (H1, H2).",
    "with a method-independent inclusion rule (Section~\\ref{sec:problem}), which defines the test of whether standard unlearning objectives reproduce reversion (H1, H2).")
rep("The selection rule, identical for all methods, maximizes forgetting efficacy \\eqref{eq:fe} subject to utility of at least 95\\% of the pre-deletion level.",
    "The selection rule, identical for all methods, chooses the configuration whose forgetting efficacy \\eqref{eq:fe} is closest to 1, the value of retraining, subject to utility of at least 95\\% of the pre-deletion level. An earlier version of this rule maximized forgetting efficacy; the pilot (Section~\\ref{sec:pilot-tests}) showed that this favors methods that over-forget, so the rule was amended before registration.")
rep("The best binary method, pre-specified as the method with the highest development forgetting efficacy,",
    "The best binary method, pre-specified as the method whose development forgetting efficacy is closest to 1,")
rep("Selection rule & maximum forgetting efficacy with utility $\\geq 95\\%$ of pre-deletion (fixed by design)\\\\",
    "Selection rule & forgetting efficacy closest to 1 with utility $\\geq 95\\%$ of pre-deletion (fixed by design)\\\\")
rep("The estimated budget is about 340 adapter runs of about three minutes each (original adapters, references with four seeds, S1 adapters, and the LongMemEval check) and about 3,900 unlearning runs of about 30 seconds each (tuning on development and S1 personas, plus test runs), roughly 50 hours on one computer.",
    "Scaling the pilot measurements to the full persona size of about 40 sessions gives about 340 adapter runs of about 70~s each (original adapters, references with four seeds, S1 adapters, and the LongMemEval check) and about 3,900 unlearning runs of about 36~s each (tuning on development and S1 personas, plus test runs), roughly 46 hours on one computer.")
for k, v in vals.items():
    s = s.replace(f"@@{k}@@", v)
left = re.findall(r"@@[A-Z_]+@@", s)
assert not left, left
open(p, "w", encoding="utf-8").write(s)
body = re.sub(r"(?<!\\)%.*", "", s)
print("braces", body.count("{") - body.count("}"))
from collections import Counter
b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", body)); e = Counter(re.findall(r"\\end\{(\w+\*?)\}", body))
print("env mismatch", {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]})
labels = set(re.findall(r"\\label\{([^}]+)\}", body)); refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
print("undefined refs", refs - labels)
cites = [k.strip() for c in re.findall(r"\\cite\{([^}]+)\}", body) for k in c.split(",")]
order = []; [order.append(k) for k in cites if k not in order]
print("bib order ok", order == re.findall(r"\\bibitem\{([^}]+)\}", body))
a = s.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
print("abstract words", len(a.replace(NEG, "-").split()))
print("tags", sorted(set(re.findall(r"⟦AUTHOR-INPUT-\d+⟧", s)), key=lambda x: int(re.findall(r"\d+", x)[0])))
