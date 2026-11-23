# Forgettable: reversion-fidelity pilot

Code, data and results for the pilot study in *"Does Unlearning Restore What a Deleted Conversation Overwrote? Reversion Fidelity in Personalized Language Models"* (A. Nimsara and M. Edirisooriya, manuscript in preparation).

**Question:** when a user deletes a conversation that had *updated* an earlier fact ("I moved to Kandy" after "I live in Colombo"), a model retrained without that conversation recalls the earlier value again. Do approximate unlearning methods reproduce this *reversion*?

## What the pilot does
For each of 6 synthetic personas (22 sessions, 12 facts each):

1. **Generate** the persona's sessions with ground-truth provenance (`pilot.py`, `make_persona`).
2. **Train** a LoRA adapter on Qwen2.5-0.5B-Instruct with MLX-LM (12 epochs, rank 8).
3. **Retrain** a reference adapter without the deleted sessions (the update sessions and the erase-control sessions).
4. **Unlearn** the deleted sessions from the original adapter with GradDiff, NPO, SimNPO and STR (supersession-targeted replay).
5. **Score** each adapter:
   - reversion fidelity (RF), Eq. (2) in the paper;
   - forgetting efficacy (FE), Eq. (3);
   - utility on unaffected facts.

## Main results (`results/summary.json`)
| Method | Median RF [95% CI] | FE (superseded) | Earlier value suppressed |
|---|---|---|---|
| Retrain (reference) | 1.00 | 1.00 | 0 / 14 |
| GradDiff | −13.29 [−39.89, 1.00] | 13.25 | 9 / 14 |
| NPO | 1.00 [0.51, 1.04] | 4.24 | 2 / 14 |
| SimNPO | 1.00 [0.98, 1.02] | 0.93 | 0 / 14 |
| STR | 1.01 [1.00, 1.04] | 4.70 | 0 / 14 |

Retraining made the earlier value the most probable answer for all 24 superseded facts; 14 facts passed the inclusion rule. The results are exploratory: one seed and six personas.

## Extended evaluation (20 personas × 3 seeds), `results_extended.json`
This is the evaluation reported in the paper (Section VI, Table 6, Fig. 2). The same personas are used across seeds (`results/`, `results_s1/`, `results_s2/`). Scores are persona-level, compared with an independently seeded retrain, with Holm correction across H1, H2a, H2b and H3 (`analyze2.py`, `numbers_v6.py`).

| Method | Median RF [95% CI] | RF over seeds (mean ± SD) | FE (superseded) |
|---|---|---|---|
| RetrainIndep | 0.89 [0.80, 0.99] | 0.98 ± 0.04 | 1.02 |
| GradDiff | −21.83 [−34.48, −18.16] | −21.05 ± 7.22 | 11.34 |
| NPO | 0.98 [0.72, 1.01] | 1.00 ± 0.00 | 3.70 |
| SimNPO | 1.00 [0.94, 1.01] | 1.00 ± 0.00 | 0.98 |
| STR | 0.95 [0.66, 1.01] | 1.00 ± 0.00 | 3.67 |
| ReplayOnly | 1.00 [0.96, 1.03] | 1.00 ± 0.00 | 1.03 |

Retraining restored the earlier value for 240/240 superseded-fact instances; 133 passed the inclusion rule. H1, H2a, H2b and H3 were not supported after Holm correction. Exploratory: gradient difference vs. independent retrain, p < 0.001.

## Update-dose study (H2a), `results_dose_s*/`
Seven personas with 28 sessions; six superseded facts at r_f ∈ {1, 2, 4}; three seeds; binary methods only (`dose.py`). Page's trend test: SimNPO L = 90, p = 0.072; GradDiff L = 94, p = 0.004 (exploratory).

## Oracle diagnostic, `results*/persona_*_oracle.json`
KL matching of the unlearned adapter to the retrained reference (`oracle.py`), on all 20 personas × 3 seeds: median RF 1.01 [1.00, 1.02], FE 0.94. It needs the retrain, so it is a diagnostic bound, not a method. `analyze_extra.py` writes `results_extra.json` for the dose and oracle analyses.

Reproduce the paper's numbers:
```bash
.venv/bin/python pilot.py --personas 20 --seed 0 --out results      # and --seed 1 --out results_s1, --seed 2 --out results_s2
.venv/bin/python dose.py --personas 7 --seed 0 --out results_dose_s0 # and seeds 1, 2
.venv/bin/python oracle.py results 0 20 0                            # and results_s1 ... 1, results_s2 ... 2
.venv/bin/python analyze2.py && .venv/bin/python analyze_extra.py && .venv/bin/python numbers_v6.py
.venv/bin/python make_fig_ext.py
```

## Reproduce
Requirements: Apple Silicon Mac, Python 3.10+, and Qwen2.5-0.5B-Instruct (downloaded automatically from Hugging Face on first run).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
HF_HUB_OFFLINE=0 .venv/bin/python pilot.py --personas 6 --seed 0   # about 22 min on an M3 Pro
.venv/bin/python analyze.py                                         # writes results/summary.json
.venv/bin/python make_fig.py                                        # writes results/nimsa2.svg
```

## Files
| Path | Content |
|---|---|
| `pilot.py` | Persona generator, LoRA training, unlearning methods, evaluation |
| `analyze.py` | Metrics, bootstrap CIs, exploratory Wilcoxon tests |
| `make_fig.py`, `make_tables.py` | Fig. 2 and the LaTeX rows of Table VI |
| `results/persona_*.json` | Per-persona facts, log-probabilities, timings and configuration |
| `results/p*/data_*` | Generated training data (full and retained) for each persona |
| `results/summary.json` | Aggregated metrics and tests |
| `results/log.txt` | Run log with timings |
| `analyze2.py`, `numbers_v6.py` | Main evaluation statistics (Table 6, Table 7) |
| `dose.py`, `oracle.py`, `analyze_extra.py` | Update-dose study and oracle diagnostic |
| `make_fig_ext.py` | Fig. 2 of the paper |

Trained adapter weights (about 135 MB) are not included; `pilot.py` regenerates them.

## Licence
Code: MIT (see `LICENSE`). Generated synthetic data and results: CC BY 4.0.
