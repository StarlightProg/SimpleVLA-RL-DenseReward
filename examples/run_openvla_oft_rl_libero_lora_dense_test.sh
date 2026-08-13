#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-libero-lora-dense-test-${SLURM_JOB_ID:-manual}}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export N_SAMPLES="${N_SAMPLES:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-10}"
export TARGET_ROLLOUTS="${TARGET_ROLLOUTS:-10}"
export SAVE_VIDEO="${SAVE_VIDEO:-True}"
export VIDEO_MAX_EPISODES="${VIDEO_MAX_EPISODES:-10}"
export VIDEO_PER_TASK_LIMIT="${VIDEO_PER_TASK_LIMIT:-1}"
export VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-1}"
export VIDEO_EVERY_N_CALLS="${VIDEO_EVERY_N_CALLS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-1}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export DENSE_REWARD_WEIGHT="${DENSE_REWARD_WEIGHT:-0.2}"
export PHASE_TRANSITION_WEIGHT="${PHASE_TRANSITION_WEIGHT:-0.3}"

echo "Starting dense LoRA test run:"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  train steps: ${TRAIN_MAX_STEPS}"
echo "  validation rollouts: ${TARGET_ROLLOUTS}"
echo "  validation videos: ${SAVE_VIDEO}, max=${VIDEO_MAX_EPISODES}, per-task=${VIDEO_PER_TASK_LIMIT}"

exec bash "${SCRIPT_DIR}/run_openvla_oft_rl_libero_lora_dense.sh" "$@"
