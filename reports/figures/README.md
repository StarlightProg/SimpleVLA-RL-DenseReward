# Paper figure set

The `paper/` directory contains publication-ready figures generated from the audited current results. Every figure is available as vector SVG, vector PDF, and 2400-pixel PNG.

## Recommended use

| Figure | Role | Supported takeaway |
|---|---|---|
| `fig1_primary_reward_comparison` | Main results | Dense-labelled checkpoints are above terminal-only at all three aligned measured steps and have the highest final observed result. |
| `fig2_all_reward_ablations` | Ablation results | Strong-terminal dense variants have the best observed values; weaker-terminal variants are less consistent. |
| `fig3_best_checkpoint_intervals` | Compact summary | Best observed success is 65.3% for the dense-labelled run versus 61.7% for terminal-only, with strongly overlapping evaluation intervals. |
| `fig4_per_task_comparison` | Error analysis | Gains are task-dependent, while tasks 1 and 5 remain low-success bottlenecks. |
| `figS1_auxiliary_dynamic_runs` | Supplement only | Auxiliary dynamic/filtering checkpoints do not show global collapse, but they are not a matched experiment. |

## Paper-ready captions

**Figure 1 — Primary reward comparison.** LIBERO-Spatial validation success for terminal-only reward with coefficient 5 and the run labelled `Dense 0.05/0.05 + terminal 5`. Each checkpoint was evaluated over 300 episodes, balanced as 30 trials for each of ten tasks. Error bars show the 95% confidence intervals logged by the validation code. The horizontal line marks the 58.0% initial checkpoint. The dense-labelled checkpoints exceed terminal-only at steps 40, 80, and 110 by 1.3, 2.3, and 2.4 percentage points, respectively. Intervals capture evaluation uncertainty and do not include training-seed variability.

**Figure 2 — Reward ablations.** Validation success across five reward configurations on LIBERO-Spatial. Points are measured checkpoints; connecting lines are visual guides rather than interpolated measurements. Every checkpoint contains 300 validation episodes, and error bars show logged 95% confidence intervals. Reward names follow the archive labels; exact training-time configurations must be confirmed from original Hydra records.

**Figure 3 — Best observed checkpoints.** Best observed validation checkpoint for each reward configuration. Labels give checkpoint success and training step. Error bars show logged 95% confidence intervals from 300 episodes. Best-of-run selection is post hoc and the intervals do not describe variability across independent training runs.

**Figure 4 — Per-task behavior.** Per-task validation success for the initial checkpoint, the best terminal-only checkpoint at step 170, and the best dense-labelled checkpoint at step 160. Each task estimate uses 30 episodes. Tasks 1 and 5 remain the lowest-success tasks across checkpoints. Task-level uncertainty bars are omitted for readability; the estimate resolution is 1/30, or 3.33 percentage points.

**Figure S1 — Auxiliary dynamic/filtering runs.** Validation success for seven auxiliary checkpoints. Error bars show logged 95% confidence intervals; labels report the denominator because six checkpoints use 500 episodes and one uses 300. These runs differ in checkpoint step and configuration documentation and are shown only as diagnostics, not as a matched reward comparison.

## Visual and statistical conventions

- The current initial checkpoint is 58.0%; no older 50.0% or alternate 58.5% baseline is used.
- Focused axes are identified in the figure notes.
- Color is reinforced with marker shape and line style.
- Aggregate error bars reproduce the values logged by the validation code.
- The main curves contain only measured checkpoints.
- SVG or PDF should be used for LaTeX; PNG is intended for preview or systems without vector support.

## Reproduction

```bash
python3 reports/figures/generate_paper_figures.py
```

The source table is `paper_figure_data.csv`. It contains the 16 main checkpoint evaluations reconciled against the raw logs. Auxiliary values are read from `../validation_audit.json`.
