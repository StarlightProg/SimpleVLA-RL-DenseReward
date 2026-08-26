# Dense Reward Validation Report

## LIBERO-Spatial: dense rewards vs. the original terminal-only reward

**Project revision:** `SimpleVLA-RL-DenseReward` @ `86111de` (`validation optimization`)  
**Evidence:** 16 validation logs, a fixed 300-rollout evaluation per checkpoint, and the three supplied figures.  
**Starting checkpoint:** approximately **50.0%** validation success before RL (run-owner note and figures).

## Executive summary

The main dense-reward variant, **dense 0.05/0.05 + terminal 5**, is the strongest tested reward under the supplied validation protocol. It reached **65.3%** success at checkpoint 160, exceeding the best terminal-only result, **61.7%** at checkpoint 170, by **3.6 percentage points (pp)**.

From the reported 50.0% pre-RL checkpoint:

| Condition | Best success | Gain from start |
|---|---:|---:|
| Original terminal-only reward, coefficient 5 | 61.7% | +11.7 pp |
| Dense 0.05/0.05 + terminal 5 | **65.3%** | **+15.3 pp** |

The dense reward has a **30.8% larger absolute learning gain** than terminal-only: `(15.3 / 11.7) - 1`.

## Scope and validation protocol

- Benchmark: **LIBERO-Spatial**.
- Metric: mean success rate on a fixed validation set.
- Denominator: **300 rollouts per evaluated checkpoint** = 10 tasks × 30 trials.
- Model/settings visible in logs: OpenVLA-OFT, LoRA rank 16 on `llm-projector`, validation key `libero_spatial_no_noops`.
- This report compares reward variants only within the supplied runs. It does not claim a reproduction of the published SimpleVLA-RL results because the starting checkpoint and training recipe differ.

## Reward variants

| Reward variant | Description | Best success | Checkpoint | Gain from 50.0% |
|---|---|---:|---:|---:|
| Terminal only (coef=5) | Original sparse terminal reward; successful episode completion is weighted by 5. | 61.7% | 170 | +11.7 pp |
| Dense 0.05/0.05 + terminal 5 | Run-owner description: 0.05 for within-phase progress and 0.05 for phase completion, added to terminal reward 5. | **65.3%** | 160 | **+15.3 pp** |
| Dense clipped + terminal 5 | Same dense formulation, but training retains only trajectories that are neither fully successful nor fully failed. | 64.7% | 50 | +14.7 pp |
| Dense 0.05/0.05 + terminal 1 | Dense formulation with terminal reward coefficient reduced to 1. | 61.0% | 110 | +11.0 pp |
| Phase-only + terminal 1 | Run-owner description: reward only for phase completion; dense reward clipping remained at 0.05. | 63.7% | 90 | +13.7 pp |

> **Configuration provenance.** The reward descriptions for `dense_clipped` and `phase_02` are run-owner notes. The attached validation logs contain evaluation-time overrides and should not be treated as the only record of the original training command.

## Primary comparison: terminal-only vs. dense 0.05/0.05 + terminal 5

| Approx. checkpoint | Terminal only (coef=5) | Dense 0.05/0.05 + terminal 5 | Dense − terminal |
|---:|---:|---:|---:|
| 40 | 61.0% | 61.8% | +0.8 pp |
| 80 | 60.7% | 63.0% | +2.3 pp |
| 110 | 61.3% | 63.7% | +2.4 pp |
| 160 / 170 | 61.7% (170) | 65.3% (160) | +3.6 pp |

### Interpretation

1. **Terminal-only baseline (coefficient 5).** The original sparse terminal reward rises from approximately 50.0% to 60.7–61.7%, then shows a near-flat plateau after step 40 in the sampled checkpoints.
2. **Dense 0.05/0.05 + terminal 5.** This is above terminal-only at every directly aligned measured checkpoint. It improves from 61.8% at step 40 to 65.3% at step 160 without a sampled regression.
3. **Dense clipped + terminal 5.** It reaches 64.7% at step 50, close to the best unfiltered dense run. However, the trajectory-selection rule differs and the run owner reports approximately four days of training despite only 50 saved steps. Its checkpoint number is therefore not comparable as a measure of training speed or compute efficiency.
4. **Terminal coefficient 1 ablations.** Dense 0.05/0.05 + terminal 1 peaks at 61.0%; phase-only + terminal 1 peaks at 63.7% and falls to 62.3% at step 110. These variants are exploratory because the terminal scale differs from the coefficient-5 reference, and phase-only shaping was still clipped at 0.05.

## What the evidence supports

- The dense 0.05/0.05 + terminal 5 series is higher than terminal-only at every directly aligned sampled checkpoint.
- Dense 0.05/0.05 + terminal 5 has the best observed validation result: **65.3%**.
- Dense-clipped is promising, but its filtered trajectory pool and different wall-clock profile prevent a direct “faster learning” conclusion.

## What remains unproven

- **Statistical significance:** each evaluation has 300 binary rollouts; logs report individual 95% half-widths of roughly ±5.4–5.6 pp. A 3.6 pp best-checkpoint gap needs repeated training seeds and a paired or independent statistical test.
- **Causality of terminal scaling:** the data are consistent with the hypothesis that lowering terminal coefficient from 5 to 1 reduces stability, but terminal scale and dense-reward composition changed at the same time.
- **Reproduction of published SimpleVLA-RL:** these runs start from an approximately 50% checkpoint and use a different recipe, so they should not be compared directly to the original paper’s final headline results.

## Recommendation

Use **dense 0.05/0.05 + terminal 5** as the next candidate configuration.

1. Repeat terminal-only (coefficient 5) and dense 0.05/0.05 + terminal 5 from the same ≈50% checkpoint with at least three matched training seeds.
2. Evaluate selected checkpoints on the fixed 300-rollout set and on an additional held-out seed set. Retain per-episode success IDs for paired analysis.
3. Match dense-clipped and unfiltered experiments by wall-clock budget; record collected, retained, fully successful, and fully failed trajectory counts.
4. Test terminal coefficient 1 versus 5 while holding dense weights, clip, and batch composition fixed.
5. Re-run phase-only shaping with the intended clip set explicitly and recorded in the launch configuration.

## Figures

### Figure 1 — terminal-only vs. dense reward runs

![Validation success: terminal vs dense reward runs](/Users/daeron/Downloads/photo_2026-08-07%2017.18.25.jpeg)

### Figure 2 — all ablations

![Validation success across all ablations](/Users/daeron/Downloads/photo_2026-08-07%2017.18.27.jpeg)

### Figure 3 — best checkpoint by ablation

![Best validation checkpoint per ablation](/Users/daeron/Downloads/photo_2026-08-07%2017.18.29.jpeg)

## Sources

- `/Users/daeron/Downloads/vaidation_runs/plots/validation_metrics_summary.csv`
- The 16 corresponding validation logs in `/Users/daeron/Downloads/vaidation_runs/`
- Run-owner notes supplied with this report request
- Reward implementation: `verl/utils/subgoal_reward/dense_reward.py`, `verl/utils/subgoal_reward/engine.py`
- Trajectory filtering implementation: `verl/trainer/ppo/ray_trainer.py`
