#!/usr/bin/env bash
set -euo pipefail

# 8GB-GPU SmolVLA LoRA launcher for LIBERO.
#
# Tested on RTX 4060 8GB:
# - Works: LoRA training, dense rewards, one-image rollouts, GRPO with 4 samples,
#   and a LoRA checkpoint from a one-step training job.
# - Validation is safe before training in this same process.
# - Continuing to a second rollout after an actor update can OOM on 8GB because
#   the FSDP actor has to reload/offload parameters around rollout/update phases.
#   Override TOTAL_TRAINING_STEPS only if you have enough free GPU memory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-/home/tamamonomaepc/liboft/LIBERO:${REPO_ROOT}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_NODES="${NUM_NODES:-1}"
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-HuggingFaceVLA/smolvla_libero}"
export CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints_smolvla_8gb}"
export DATASET_NAME="${DATASET_NAME:-libero_spatial}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-smolvla-libero-lora-8gb}"

cd "$REPO_ROOT"

bash "$SCRIPT_DIR/run_smolvla_rl_libero_lora_dense.sh" \
    data.num_trials_per_task="${NUM_TRIALS_PER_TASK:-1}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-1}" \
    data.n_samples="${N_SAMPLES:-4}" \
    data.val_batch_size=1 \
    data.filter_accuracy=False \
    data.oversample_factor=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.num_images_in_input="${NUM_IMAGES_IN_INPUT:-1}" \
    actor_rollout_ref.rollout.num_images_in_input="${NUM_IMAGES_IN_INPUT:-1}" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-4}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-4}" \
    actor_rollout_ref.rollout.max_episode_steps="${MAX_EPISODE_STEPS:-8}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1000}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-1}" \
    trainer.save_freq="${SAVE_FREQ:-1}" \
    trainer.test_freq="${TEST_FREQ:--1}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.val_only=False \
    trainer.final_val_after_train=False \
    trainer.final_save_after_train="${FINAL_SAVE_AFTER_TRAIN:-False}" \
    trainer.validation.target_rollouts="${VAL_TARGET_ROLLOUTS:-1}" \
    trainer.validation.max_passes="${VAL_MAX_PASSES:-1}" \
    trainer.validation.save_video="${VAL_SAVE_VIDEO:-False}" \
    trainer.validation.video_max_episodes=0 \
    trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
    trainer.wandb_mode="$WANDB_MODE" \
    "$@"
