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
