# Dense Subgoal Rewards for Compute-Constrained SimpleVLA-RL

## Technical report and paper-preparation source

**Benchmark:** LIBERO-Spatial  
**Model family:** OpenVLA-OFT with LoRA  
**Primary comparison:** terminal-only outcome reward versus dense subgoal reward  
**Current starting checkpoint:** 0.580 validation success  
**Implementation snapshot reviewed:** `SimpleVLA-RL-DenseReward`, `origin/main` commit `1b5fc6c3791db1d010c10b334bb408c39a7e7ada`  
**Result archive reviewed:** `validation_results.zip`  
**Analysis date:** 2026-08-26

## Abstract

This study examines whether dense, phase-aware reward shaping improves online reinforcement learning of an OpenVLA-OFT policy in a compute-constrained SimpleVLA-RL setup. The original SimpleVLA-RL method uses a sparse binary outcome reward; the present extension augments that signal with positive within-phase progress and phase-completion rewards derived from simulator state. The current result archive evaluates five reward configurations on LIBERO-Spatial from a checkpoint with 58.0% initial validation success. Each main checkpoint evaluation contains 300 rollouts, balanced as 30 trials for each of ten tasks. The strongest observed checkpoint labelled `Dense 0.05/0.05 + terminal 5` reaches 65.3%, compared with 61.7% for the terminal-only control, an observed difference of 3.6 percentage points. At the checkpoints shared by both series (40, 80, and 110), the dense-labelled checkpoints are higher by 1.3, 2.3, and 2.4 points, respectively. The dense-labelled series is also monotonic across the four reported checkpoints, whereas the terminal-only series initially falls slightly and then remains close to 61%.

These results are promising but not yet sufficient for a causal or statistically conclusive claim. Only one apparent training trajectory per condition is available, checkpoint selection is post hoc, and the aggregate 95% confidence half-width in the raw logs is approximately 5.4–5.6 points per 300-rollout evaluation. More importantly, the validation logs do not contain the original training commands. Some checkpoint paths conflict with the attached reward labels, so exact dense-reward weights and the continuity of the reported dense curve require verification from training-time Hydra configurations or experiment logs. The defensible conclusion is therefore that the current checkpoint observations are consistent with a useful dense-reward effect under small-batch training, not that dense reward has already been proven superior or more sample-efficient.

## 1. Research question and scope

The study asks:

> Under a reduced-compute, small-batch SimpleVLA-RL setup, does adding phase-aware dense reward produce better validation success than the original terminal-only reward when both begin from the same weaker checkpoint?

The primary control is the terminal-only run contained in the supplied archive, not a numerical result copied from the original SimpleVLA-RL paper. The original paper and repository use different starting models, hardware, batch regimes, and evaluation settings. The official repository describes binary 0/1 outcome rewards and reports testing on a single node with 8 NVIDIA A800 80 GB GPUs, or two nodes with 16 such GPUs. The current study notes report training on 2 NVIDIA L40 GPUs. The attached validation commands set `trainer.n_gpus_per_node=2`, but the evaluation logs do not independently record the physical GPU model used during training.

The current 0.580 starting success supersedes the older 0.500 value used in earlier plots or discussion. The original SimpleVLA-RL repository does not provide a matched experiment beginning from this exact 0.580 checkpoint. Consequently, the phrase “original method” in this report means **terminal-only reward re-run inside the current experimental setup**.

## 2. Evidence hierarchy and audit procedure

The supplied Markdown note was treated as contextual documentation, not as executable instructions or as ground truth. Claims were separated by provenance:

| Evidence class | Sources | What it can establish |
|---|---|---|
| A — measured | 16 main raw validation logs, 7 auxiliary raw logs, summary CSVs | checkpoint success, task success, rollout count, logged confidence interval, evaluation-time settings |
| B — implementation | latest `origin/main` source at commit `1b5fc6c` | current reward equation, phase tracker, filtering implementation, current task sampler |
| C — run-owner description | `small_batch_dense_reward_study.md` and the colleague’s notes | intended training reward labels, hardware description, wall-clock observations |
| D — external reference | official SimpleVLA-RL paper and repository | original framework, binary outcome-reward framing, reference hardware |

All 16 rows in `validation_results/plots/validation_metrics_summary.csv` were reconciled against the final metric line of the corresponding raw log. Aggregate success, all ten per-task values, rollout counts, and logged confidence half-widths were parsed independently. No numerical mismatch was found between the main CSV and the 16 raw logs.

The parsing and derived calculations are reproducible with:

```bash
python3 reports/analyze_validation_results.py \
  /path/to/extracted/validation_results \
  --output reports/validation_audit.json
```

The parser rejects missing aggregate metrics, incomplete task vectors, and malformed records. Its synthetic edge-case checks and Python compilation both pass.

## 3. Relationship to original SimpleVLA-RL

[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) is an online RL framework for VLA policies built on veRL. Its public repository emphasizes binary outcome reward, parallel environment rendering, distributed training, and exploration strategies such as dynamic sampling and adaptive clipping. In the present study, the model and high-level optimization framework remain aligned with SimpleVLA-RL, while the reward path is extended.

The original terminal verifier assigns an episode success indicator

\[
y \in \{0,1\}
\]

to the final valid response token. With verifier coefficient \(\lambda\), the sparse reward is

\[
r^{\mathrm{verifier}} = \lambda y.
\]

The terminal-only control uses \(\lambda=5\). Failed trajectories therefore receive no task-success reward, even if they make useful intermediate progress.

No result from the original paper is directly subtracted from the current dense result. Such a subtraction would confound at least the initial checkpoint, hardware, batch construction, number of rollouts, training duration, and evaluation protocol.

## 4. Current dense-reward method

### 4.1 Phase decomposition

For supported LIBERO pick-and-place instructions, the current repository infers a five-phase task:

1. reach the object;
2. grasp the object;
3. lift the object;
4. move the object to the target;
5. place the object or reach environment success.

The default geometric thresholds are 0.05 m for reaching, 0.06 m for target proximity, and 0.08 m for lifting. Task support is inferred from instruction text and the availability of gripper, object, and target positions. Unsupported tasks fall back to terminal-only behavior unless the configuration requests an error.

### 4.2 Monotonic progress tracking

Let \(p_t\in[0,1]\) denote progress within the current phase. With `use_best_progress=True`, only improvement over the best previous progress is rewarded:

\[
\Delta p_t^+ = \max\!\left(p_t-\max_{\tau<t}p_\tau,0\right).
\]

This prevents repeated reward from moving backward and then returning to an already reached state. The tracker also records the number \(c_t\) of phases completed at a step. This can be greater than one when a state satisfies multiple phase transitions or when final success auto-completes remaining phases.

### 4.3 Dense reward equation

The implementation computes

\[
r_t^{\mathrm{progress}} = w_p\Delta p_t^+,
\]

\[
r_t^{\mathrm{phase}} = w_c c_t,
\]

\[
r_t^{\mathrm{smooth}} = -w_s\lVert a_t-a_{t-1}\rVert_2,
\]

and clips their sum:

\[
r_t^{\mathrm{shape}}=
\operatorname{clip}\!\left(
r_t^{\mathrm{progress}}+r_t^{\mathrm{phase}}+r_t^{\mathrm{smooth}},
-C,C
\right).
\]

An internal terminal term is then added outside that clip:

\[
r_t^{\mathrm{dense}}=r_t^{\mathrm{shape}}+w_Ty_t.
\]

In `mode=add`, the reward passed to optimization is

\[
r_t = r_t^{\mathrm{verifier}} + r_t^{\mathrm{dense}}.
\]

This ordering matters. `clip_dense_reward` bounds only progress, phase, and smoothness shaping; it does not clip either the dense module’s terminal term or the external verifier reward.

The current repository defaults are `C=0.05`, \(w_p=0.2\), \(w_c=0.05\), \(w_T=1\), and \(w_s=0\), with dense reward disabled in the base configuration. The current dense launcher enables `mode=add` and defaults to \(w_p=0.2\), \(w_c=0.3\). Because the launcher does not override \(w_T\), a successful terminal step under the current defaults can contain both \(5y\) from the verifier and \(1y\) from the dense module. Any paper table that calls this simply “terminal 5” must verify whether `terminal_success` was set to zero or one in the actual training run.

### 4.4 Mixed-outcome trajectory filtering

The run-owner describes `dense_clipped` as retaining only trajectory groups that are neither fully successful nor fully failed. In the current trainer, `n_samples` responses belonging to one prompt are reshaped into a group, their mean binary accuracy is computed, and the whole group is retained when

\[
L \leq \frac{1}{n}\sum_{i=1}^{n} y_i \leq U.
\]

With \(n=4\), excluding homogeneous groups requires bounds that reject group accuracies 0 and 1. This is **trajectory-group filtering**, not the same operation as clipping the numerical dense reward to \([-C,C]\). It deliberately focuses GRPO updates on groups containing outcome variation, but may require many more environment rollouts to fill one retained batch.

The run-owner reports that the 50-step filtered run required about four days, similar to unfiltered runs reaching approximately 160 steps. That observation explains why checkpoint index alone cannot be interpreted as wall-clock or environment-sample efficiency. The training logs needed to verify acceptance rates and total generated rollouts were not supplied.

### 4.5 Current task-aware sampling extension

The latest repository revision includes an optional `TaskBalancedHardBatchSampler`. It reserves a configurable fraction of every batch for rotating task coverage and samples remaining slots according to difficulty:

\[
q_k \propto 1-\operatorname{EMA}(s_k),
\]

mixed with a uniform distribution and a probability floor. Validation success updates the per-task exponential moving average. This mechanism directly targets the persistent low-success tasks observed below. It was added after the supplied primary runs and must be described as a current implementation or proposed follow-up, not as a method already evaluated by the main ablation table.

## 5. Experimental protocol recoverable from the archive

### 5.1 Confirmed evaluation settings

The raw checkpoint evaluations consistently show:

| Item | Value |
|---|---|
| benchmark | LIBERO-Spatial |
| model | OpenVLA-OFT |
| adapter | LoRA rank 16, alpha 16, target `llm-projector` |
| validation action normalization key | `libero_spatial_no_noops` |
| main evaluation size | 300 rollouts |
| task balance | 10 tasks × 30 trials |
| validation start index | 0 |
| validation passes | 1 |
| validation mode | checkpoint-only; no policy update |
| aggregate metric | mean binary episode success |

The main metric is both the micro-average over 300 episodes and, because every task contributes 30 trials, the macro-average over ten task success rates.

### 5.2 Training context requiring provenance qualification

The attached notes state that training used 2 L40 GPUs and small batches. Evaluation command templates contain `train_batch_size=6`, `n_samples=4`, PPO mini-batch 6, global PPO micro-batch 2, learning rate \(10^{-5}\), GRPO, KL coefficient 0, rollout temperature 1, and two GPUs. These fields were printed by a launcher that was then switched to `val_only=True`; they do not prove that every loaded checkpoint was trained with exactly those values. A publication-ready methods section should recover the original training-time Hydra config stored with each checkpoint or the corresponding Weights & Biases run.

The original repository reference should be stated as 8×A800 80 GB, not 8×A100.

## 6. Main validation results

The current baseline is 0.580. This value appears in three equivalent CSVs, including `initial_default_checkpoint_distribution.csv`, but the archive does not contain the raw initial-checkpoint validation log. At 300 rollouts, its implied count is 174 successes. The per-task values occur in increments of 1/30, consistent with 30 trials per task.

| Ablation label in archive | Observed checkpoints | Best success | Best step | Gain over 0.580 | Last success |
|---|---|---:|---:|---:|---:|
| Terminal only, coefficient 5 | 40, 80, 110, 170 | 0.617 | 170 | +0.037 | 0.617 |
| Dense 0.05/0.05 + terminal 5 | 40, 80, 110, 160 | **0.653** | 160 | **+0.073** | **0.653** |
| Dense clipped + terminal 5 | 30, 50 | 0.647 | 50 | +0.067 | 0.647 |
| Dense 0.05/0.05 + terminal 1 | 60, 110 | 0.610 | 110 | +0.030 | 0.610 |
| Phase 0.2 + terminal 1 | 30, 60, 90, 110 | 0.637 | 90 | +0.057 | 0.623 |

### 6.1 Primary terminal-versus-dense comparison

| Step | Terminal only, coefficient 5 | Dense-labelled, terminal 5 | Dense minus terminal |
|---:|---:|---:|---:|
| 40 | 0.610 | 0.623 | +0.013 |
| 80 | 0.607 | 0.630 | +0.023 |
| 110 | 0.613 | 0.637 | +0.024 |
| final measured, 170 vs 160 | 0.617 | 0.653 | +0.036 |

Across the three exactly aligned checkpoint indices, the dense-labelled result is higher every time, with a mean observed advantage of 2.0 percentage points. The final observed advantage is 3.6 points, or 5.8% relative to the terminal-only success rate. Relative to the common initial checkpoint, the dense-labelled gain is 7.3 points, compared with 3.7 points for terminal-only. Thus, the **observed improvement over baseline** is 97% larger for the dense-labelled series:

\[
\frac{0.073}{0.037}-1 = 0.973.
\]

This ratio describes the supplied checkpoint observations; it is not an estimate of generalizable treatment effect.

### 6.2 Shape of the measured trajectories

Terminal only:

```text
0.610 -> 0.607 -> 0.613 -> 0.617
```

Dense-labelled, terminal 5:

```text
0.623 -> 0.630 -> 0.637 -> 0.653
```

The dense-labelled series is monotonic across its four observed checkpoints. The terminal series has a 0.3-point early decrease and then changes by less than one point between successive evaluations. It is reasonable to write that the dense-labelled curve is **more consistently increasing at the measured checkpoints**. It is not yet reasonable to claim lower training variance or formal stability, because there is one run, only four measurements, and no repeated seeds.

### 6.3 Strong versus weak terminal coefficient

The two best main results both retain terminal coefficient 5: 0.653 for the dense-labelled series and 0.647 for the filtered dense series. Dense 0.05/0.05 with terminal coefficient 1 reaches 0.610. The phase-focused coefficient-1 run reaches 0.637 at step 90 but falls to 0.623 at step 110.

These observations are consistent with a useful anchoring role for a strong terminal objective. They do not isolate a terminal-coefficient effect, because the reward composition and, potentially, training selection also differ. A clean coefficient ablation must hold dense weights, dense terminal weight, clipping, batch construction, rollout budget, and seed fixed.

### 6.4 Mixed-outcome filtered run

The filtered dense run reaches 0.647 by checkpoint 50. This is 3.0 points above the best terminal checkpoint and only 0.6 points below the best dense-labelled checkpoint. However, it has only two evaluations, uses a different trajectory-selection rule, and reportedly consumed roughly the same wall time as the much longer unfiltered runs. The valid conclusion is that mixed-outcome filtering did not cause an obvious validation collapse. The data do not show that it is more sample-efficient or compute-efficient.

## 7. Per-task analysis

At the best overall checkpoint of each principal condition:

| Task | Initial | Terminal best | Dense-labelled best | Filtered dense best | Dense minus terminal |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.367 | 0.533 | 0.600 | 0.633 | +0.067 |
| 1 | 0.100 | 0.067 | 0.100 | 0.133 | +0.033 |
| 2 | 0.867 | 0.900 | 0.933 | 0.967 | +0.033 |
| 3 | 0.800 | 0.867 | 0.900 | 0.867 | +0.033 |
| 4 | 0.767 | 0.733 | 0.900 | 0.800 | **+0.167** |
| 5 | 0.033 | 0.033 | 0.067 | 0.067 | +0.034 |
| 6 | 0.767 | 0.867 | 0.833 | 0.900 | −0.034 |
| 7 | 0.633 | 0.667 | 0.767 | 0.600 | **+0.100** |
| 8 | 0.767 | 0.800 | 0.800 | 0.867 | 0.000 |
| 9 | 0.700 | 0.700 | 0.633 | 0.633 | **−0.067** |

The best dense-labelled checkpoint exceeds the best terminal checkpoint on tasks 0–5 and 7, ties on task 8, and is lower on tasks 6 and 9. The largest positive differences are task 4 (+16.7 points) and task 7 (+10.0 points). The principal regressions are task 9 (−6.7 points) and task 6 (−3.4 points).

Tasks 1 and 5 remain the main bottlenecks. Their initial success is 0.100 and 0.033; at the best dense-labelled checkpoint it is 0.100 and 0.067. Dense reward therefore improves the aggregate primarily through other tasks and does not solve the low-success tail.

Each task estimate is based on only 30 binary episodes, giving a resolution of 3.33 points and large uncertainty. The per-task table is valuable for diagnosis but should not be used to claim task-specific superiority without more trials or repeated evaluation seeds.

## 8. Additional batch-32 / dynamic-filtering checkpoints

The `trainings_with_dynamic_clip/` directory contains seven additional checkpoint evaluations. They are analyzed separately because they have different provenance and mostly 500-rollout validation.

| Checkpoint label | Step | Rollouts | Success | Logged 95% half-width | Task 1 | Task 5 |
|---|---:|---:|---:|---:|---:|---:|
| terminal | 19 | 500 | 0.598 | 0.043 | 0.120 | 0.040 |
| terminal | 59 | 500 | 0.604 | 0.043 | 0.120 | 0.040 |
| dense 0.1/0.1 | 19 | 500 | 0.602 | 0.043 | 0.080 | 0.020 |
| dense 0.1 | 79 | 500 | 0.630 | 0.042 | 0.120 | 0.080 |
| phase 0.2 | 19 | 500 | 0.594 | 0.043 | 0.080 | 0.040 |
| phase 0.2 | 79 | 500 | 0.600 | 0.043 | 0.080 | 0.020 |
| dense clip 0.3/0.5 | 79 | 300 | 0.637 | 0.054 | 0.133 | 0.133 |

At the shared step 19, dense 0.1/0.1 is only 0.4 point above terminal, while phase 0.2 is 0.4 point below terminal. At later but non-matched steps, dense reaches 0.630 and filtered dense reaches 0.637. These runs reinforce two qualitative observations: dense or filtered training does not show global collapse, and tasks 1 and 5 remain much weaker than the overall average. They do not provide a controlled comparison because checkpoint steps, rollout denominators, and configuration documentation differ.

The directory’s `summary.tsv` is malformed: values such as 500 were written in the `accuracy` column while `num_rollouts` is empty. Its `commands.txt` documents only three of seven launches. The table above therefore comes from raw metric lines, not from `summary.tsv`.

## 9. Uncertainty and robustness

### 9.1 Logged confidence intervals

For the 300-rollout main evaluations, the logs compute a normal-approximation 95% half-width of approximately 0.054–0.056. The best dense-labelled and terminal-only intervals therefore overlap substantially.

As a sensitivity calculation only, treating the best dense and terminal evaluations as independent binomial samples gives:

\[
\hat\Delta = 0.653-0.617=0.036,
\]

with approximate 95% interval

\[
[-0.041, 0.113]
\]

and two-sided \(p\approx0.36\). Comparing the best dense result with the 0.580 initial CSV gives an approximate interval of \([-0.005,0.151]\) and \(p\approx0.065\). These are not definitive tests: the same validation episodes may have been reused, but paired episode outcomes were not supplied, and the “best” checkpoints were selected after inspecting multiple evaluations. A paired McNemar analysis cannot be reconstructed from aggregate rates.

### 9.2 Major threats to validity

1. **No repeated training seeds.** Variation from initialization, rollout sampling, or environment interaction is unknown.
2. **Post hoc best-checkpoint selection.** Conditions have two to four evaluated checkpoints and unequal endpoints, biasing best-of-run comparisons upward.
3. **Sparse checkpoint sampling.** Four points do not characterize optimization variance or convergence.
4. **Training configuration is missing.** Validation-only commands do not prove the reward and optimizer settings used to produce checkpoints.
5. **Dense-series provenance conflict.** Logs labelled `Dense 0.05/0.05 + terminal 5` load adapters from directories named `dense_02_03` and `dense_03_02`, across different parent directories. The exact weights and whether all four points form one training trajectory are not verified.
6. **Terminal double-counting is unresolved.** Current code adds the dense module’s default terminal weight 1 on top of verifier coefficient 5 unless explicitly overridden.
7. **Initial baseline has no raw log.** The 0.580 value is internally consistent across CSVs but lacks the corresponding raw evaluation trace.
8. **Compute accounting is absent.** Checkpoint step does not encode generated rollouts, retained rollouts, environment transitions, GPU-hours, or wall-clock time.
9. **Single benchmark suite.** Evidence is limited to LIBERO-Spatial and should not be generalized to other LIBERO suites or real robots.
10. **Reward invariance is not established.** The implemented dense reward has not been shown to be potential-based; it may change the optimal policy and can, in principle, reward intermediate behavior that does not improve completion.

### 9.3 Plot audit

Use the following files from the archive for the current 0.580 study:

- `validation_results/plots/overall_selected_terminal_dense_denseclipped.svg`;
- `validation_results/plots/overall_all_ablations.svg`;
- `validation_results/plots/best_checkpoint_by_ablation.svg`;
- `validation_results/plots/per_task_all_ablations_grid.svg`.

Prefer SVG for paper preparation. The alternative `plots/starting_585/` figures should not be used: their caption says the axis starts at 58.5%, while the plotted initial observation is 58.0%, and no 0.585 baseline record is present in the supplied CSVs. New publication figures should add confidence intervals and distinguish measured points from connecting lines.

## 10. Defensible conclusions

The evidence supports the following statements:

- Under the supplied 300-rollout validation protocol, the strongest dense-labelled checkpoint achieves 65.3% success versus 61.7% for the terminal-only control.
- At the three aligned checkpoint indices, the dense-labelled checkpoints exceed terminal-only by 1.3–2.4 percentage points.
- The dense-labelled series increases monotonically across the four measured checkpoints; terminal-only remains near 61% after its first evaluation.
- Mixed-outcome trajectory filtering does not cause an obvious global collapse in the supplied runs, but its rollout cost is not recorded.
- Tasks 1 and 5 remain low-success bottlenecks across reward variants.
- A strong terminal objective appears useful, but its independent causal effect has not been isolated.

The following statements are not supported yet:

- dense reward is statistically significantly better;
- dense reward is required for training to work;
- dense reward is proven to be more environment-sample-efficient or compute-efficient;
- dynamic clipping ruins training;
- the current experiments reproduce the original paper’s headline results;
- the exact `0.05/0.05` training weights are verified by the supplied raw logs.

## 11. Recommended paper wording

### Result statement

> In a compute-constrained LIBERO-Spatial study initialized from a checkpoint with 58.0% validation success, the best checkpoint labelled with dense progress and phase rewards reached 65.3%, compared with 61.7% for a terminal-only coefficient-5 control. At steps 40, 80, and 110, the dense-labelled checkpoints were higher by 1.3, 2.3, and 2.4 percentage points. These single-run results suggest that intermediate phase-aware feedback may improve small-batch optimization, but repeated matched seeds are required to establish statistical reliability.

### Method statement

> We augment the SimpleVLA-RL terminal success verifier with simulator-derived subgoal shaping. Supported pick-and-place tasks are decomposed into reach, grasp, lift, move, and place phases. At each environment step, the method rewards positive improvement over the best previously observed progress within the current phase and discrete phase completion. The sum of progress, transition, and optional action-smoothness components is clipped before an optional dense terminal term is added. In additive mode, the resulting reward is combined with the external terminal verifier reward.

### Compute statement

> The run-owner reports training on two NVIDIA L40 GPUs, compared with the original repository’s tested 8×A800 80 GB single-node configuration. The supplied checkpoint-evaluation commands use two GPUs, although training-time hardware telemetry and complete Hydra configurations were not included in the archive.

## 12. Experiments required before submission

### Highest priority

1. Recover and archive the exact training Hydra configuration for every checkpoint, including `verifier.reward_coef`, all dense weights, `clip_dense_reward`, filter bounds, `n_samples`, batch sizes, seed, and commit hash.
2. Confirm whether the four dense-labelled points belong to one training trajectory and resolve the `0.05/0.05` versus `0.2/0.3` / `0.3/0.2` naming conflict.
3. Repeat terminal-only coefficient 5 and the selected dense configuration with at least three matched training seeds.
4. Store episode-level validation records containing task ID, trial/seed ID, checkpoint, and success. Use paired tests on the same validation episodes.
5. Report three x-axes: optimizer updates, total environment episodes/transitions generated, and GPU-hours or wall-clock time.

### Controlled ablations

6. Compare terminal coefficient 1 and 5 while holding every dense parameter fixed.
7. Set the dense internal terminal weight explicitly to 0 or 1 and report it, avoiding ambiguous effective terminal scale.
8. Compare unfiltered and mixed-outcome filtered training under matched generated-rollout and wall-clock budgets; report filter acceptance rate.
9. Evaluate progress-only, phase-only, and progress-plus-phase rewards with the same clip. Because clipping is applied after summation, document how often multiple components saturate the clip.
10. Test the latest task-balanced hard sampler, especially on tasks 1 and 5, with a uniform-sampling control.

### Robustness

11. Add a second held-out validation seed set to reduce repeated-set selection bias.
12. Evaluate at least one additional LIBERO suite.
13. Log dense reward components by task and phase to check for saturation, reward hacking, and unsupported-task fallback.
14. Pre-register one primary checkpoint-selection rule, such as area under the validation curve up to a fixed environment-rollout budget or final success at a fixed budget.

## 13. Suggested manuscript structure

1. **Introduction:** sparse terminal feedback is simple and aligned but difficult for small-batch credit assignment.
2. **Background:** SimpleVLA-RL, OpenVLA-OFT, GRPO, and binary outcome rewards.
3. **Method:** simulator-state extraction, five-phase task representation, monotonic progress, clipped shaping, terminal anchoring, and optional mixed-outcome filtering.
4. **Experimental setup:** weaker 0.580 checkpoint, LIBERO-Spatial, 2×L40 reported compute, exact batch and rollout accounting.
5. **Main results:** terminal versus dense under matched seeds and budgets.
6. **Ablations:** terminal scale, phase versus progress, clipping, filtering, task-aware sampling.
7. **Per-task analysis:** gains, regressions, and persistent hard tasks.
8. **Limitations:** provenance, uncertainty, single suite, simulator dependence, and reward-alignment risk.
9. **Conclusion:** dense shaping is promising for low-resource VLA RL but requires controlled replication.

## 14. Source map

### Supplied artifacts

- `small_batch_dense_reward_study.md` — run-owner narrative and intended labels;
- `validation_results/plots/validation_metrics_summary.csv` — main checkpoint table;
- `validation_results/plots/initial_default_checkpoint_distribution.csv` — current 0.580 baseline;
- 16 top-level `.log` files — raw 300-rollout main validation records;
- `validation_results/trainings_with_dynamic_clip/*.log` — seven auxiliary records;
- `validation_results/plots/*.svg` — paper figure sources.

### Current implementation files at `1b5fc6c`

- `verl/utils/subgoal_reward/dense_reward.py` — component weights, clipping, and total dense reward;
- `verl/utils/subgoal_reward/tracker.py` — monotonic progress and phase-transition tracking;
- `verl/utils/subgoal_reward/task_specs.py` — supported pick-and-place phase sequence;
- `verl/utils/subgoal_reward/phases.py` — geometric progress functions and thresholds;
- `verl/utils/subgoal_reward/engine.py` — state extraction, configuration, fallback behavior;
- `verl/trainer/main_ppo.py` — verifier reward and dense `add` / `replace` integration;
- `verl/trainer/ppo/ray_trainer.py` — group filtering, validation metrics, and sampler updates;
- `verl/utils/dataset/rob_dataset.py` — task-balanced hard sampler;
- `examples/run_openvla_oft_rl_libero_lora_dense.sh` — current dense launcher defaults.

### External references

- Li et al., “SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning,” [arXiv:2509.09674](https://arxiv.org/abs/2509.09674).
- [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL), official repository and hardware documentation.
- [StarlightProg/SimpleVLA-RL-DenseReward](https://github.com/StarlightProg/SimpleVLA-RL-DenseReward), dense-reward implementation reviewed here.
- Ng, Harada, and Russell, “Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping,” [ICML 1999](https://dl.acm.org/doi/10.5555/645528.657613).

## 15. Concise paper-ready conclusion

The current validation archive provides consistent descriptive evidence that phase-aware dense reward is a promising extension of terminal-only SimpleVLA-RL in a weaker-checkpoint, small-batch regime. The best dense-labelled checkpoint reaches 65.3%, 3.6 points above the terminal-only best and 7.3 points above the 58.0% starting checkpoint. The advantage is present at every aligned measured step, while low-success tasks 1 and 5 remain largely unresolved. The result should presently be framed as a single-run empirical trend. Exact training configuration recovery, matched repeated seeds, episode-level paired evaluation, and rollout/GPU-hour accounting are necessary before making claims of statistical superiority, stability, or sample efficiency.
