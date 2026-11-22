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

## Extended evaluation (7 personas × 3 seeds), `results_extended.json`
Same personas across seeds (`results/`, `results_s1/`, `results_s2/`). Persona-level scores; independent-retrain comparator; Holm correction across H1, H2b and H3 (`analyze2.py`).

| Method | Median RF [95% CI] | RF over seeds (mean ± SD) | FE (superseded) |
|---|---|---|---|
| RetrainIndep | 0.85 [0.80, 0.99] | 0.96 ± 0.11 | 1.02 |
| GradDiff | −21.07 [−42.83, −8.38] | −21.70 ± 9.30 | 11.26 |
| NPO | 0.96 [0.71, 0.99] | 0.97 ± 0.05 | 5.46 |
| SimNPO | 1.01 [0.86, 1.02] | 1.00 ± 0.01 | 1.10 |
| STR | 0.98 [0.44, 1.01] | 1.01 ± 0.01 | 4.95 |
| ReplayOnly | 1.02 [1.00, 1.06] | 1.01 ± 0.02 | 1.16 |

Retraining restored the earlier value for 84/84 superseded-fact instances. H1, H2b and H3 were not supported after Holm correction. Exploratory: gradient difference vs. independent retrain, p = 0.008.

Reproduce: `pilot.py --personas 8 --seed 1 --out results_s1`, `--seed 2 --out results_s2`, `add_replay_only.py 0 6`, then `analyze2.py` and `make_fig_ext.py`.

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

Trained adapter weights (about 135 MB) are not included; `pilot.py` regenerates them.

## Licence
Code: MIT (see `LICENSE`). Generated synthetic data and results: CC BY 4.0.
