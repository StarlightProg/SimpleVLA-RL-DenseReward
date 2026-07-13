#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  validate_openvla_oft_rl_libero_lora_checkpoint.sh /path/to/lora_adapter [experiment_name] [hydra_overrides...]

Required:
  /path/to/lora_adapter   PEFT adapter directory with adapter_config.json and adapter_model.*

Common environment knobs:
  SFT_MODEL_PATH          Base OpenVLA-OFT checkpoint. If unset, inferred from checkpoint metadata or ../openvla_model.
  DATASET_NAME            LIBERO suite, e.g. libero_spatial/libero_object/libero_goal/libero_10.
  LIBERO_ROOT             LIBERO checkout path. Defaults to ../LIBERO.
  NUM_GPUS                GPUs per node. Defaults to 2.
  VAL_BATCH_SIZE          Validation batch size. Defaults to NUM_GPUS; keep >= NUM_GPUS for FSDP.
  TARGET_ROLLOUTS         Number of validation episodes to run. Defaults to 40.
  START_INDEX             Validation dataset offset for chunked eval. Defaults to 0.
  NUM_TRIALS_PER_TASK     Trials per LIBERO task used to build validation dataset. Defaults to launcher value.
  SAVE_VIDEO              true/false. Defaults to true.
  VIDEO_MAX_EPISODES      Max saved validation videos. Defaults to 10.
  VIDEO_PER_TASK_LIMIT    Max saved videos per task. Defaults to 1.
  VIDEO_FRAME_STRIDE      Save every Nth env frame. Defaults to 8.
  WANDB_MODE              disabled/offline/online. Defaults to disabled for this wrapper.
  CKPT_PATH               Output/checkpoint root. Defaults to ../checkpoints.
  ALIGN_PATH              Ray runtime env JSON. Defaults to ./align.json.
  PYTHONNOUSERSITE        Defaults to 1 to avoid ~/.local package conflicts.

Examples:
  NUM_GPUS=2 TARGET_ROLLOUTS=2 SAVE_VIDEO=false \
    bash examples/validate_openvla_oft_rl_libero_lora_checkpoint.sh /path/to/lora_adapter smoke

  NUM_GPUS=2 VAL_BATCH_SIZE=2 TARGET_ROLLOUTS=100 START_INDEX=200 NUM_TRIALS_PER_TASK=50 \
    bash examples/validate_openvla_oft_rl_libero_lora_checkpoint.sh /path/to/lora_adapter chunk-200

Hydra overrides at the end still work and take final precedence.
EOF
}

if [ "$#" -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 1
fi

LORA_ADAPTER_PATH="$1"
DEFAULT_EVAL_EXPERIMENT_NAME="$(basename "$(dirname "$LORA_ADAPTER_PATH")")-libero-spatial-videos"
NUM_GPUS="${NUM_GPUS:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$NUM_GPUS}"
TARGET_ROLLOUTS="${TARGET_ROLLOUTS:-40}"
START_INDEX="${START_INDEX:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
VIDEO_MAX_EPISODES="${VIDEO_MAX_EPISODES:-10}"
VIDEO_PER_TASK_LIMIT="${VIDEO_PER_TASK_LIMIT:-1}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export NUM_GPUS WANDB_MODE PYTHONNOUSERSITE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_ROOT="$(cd "$(dirname "$LORA_ADAPTER_PATH")" && pwd)"
if [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo "Missing LoRA adapter directory: $LORA_ADAPTER_PATH" >&2
    exit 1
fi
if [ ! -f "$LORA_ADAPTER_PATH/adapter_config.json" ]; then
    echo "Missing PEFT adapter config: $LORA_ADAPTER_PATH/adapter_config.json" >&2
    exit 1
fi
if [ ! -f "$LORA_ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$LORA_ADAPTER_PATH/adapter_model.bin" ]; then
    echo "Missing PEFT adapter weights under: $LORA_ADAPTER_PATH" >&2
    exit 1
fi
if [ "$#" -ge 2 ] && [[ "$2" != *=* ]]; then
    EVAL_EXPERIMENT_NAME="$2"
    shift 2
else
    EVAL_EXPERIMENT_NAME="$DEFAULT_EVAL_EXPERIMENT_NAME"
    shift 1
fi

read -r META_MODEL_PATH META_TASK_SUITE META_UNNORM_KEY < <(
    CHECKPOINT_ROOT="$CHECKPOINT_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CHECKPOINT_ROOT"])
metadata_path = root / "checkpoint_metadata.json"
training_config_path = root / "training_config.json"

metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
training = json.loads(training_config_path.read_text()) if training_config_path.exists() else {}

model_path = (
    metadata.get("base_model_path_config")
    or metadata.get("base_model_path")
    or training.get("model", {}).get("path")
    or ""
)
rollout = training.get("rollout", {})
task_suite = rollout.get("task_suite_name") or ""
unnorm_key = rollout.get("unnorm_key") or task_suite
print(model_path, task_suite, unnorm_key)
PY
)

if [ -n "${META_MODEL_PATH}" ] && [ -z "${SFT_MODEL_PATH:-}" ]; then
    export SFT_MODEL_PATH="${META_MODEL_PATH}"
fi
if [ -n "${META_TASK_SUITE}" ] && [ -z "${DATASET_NAME:-}" ]; then
    export DATASET_NAME="${META_TASK_SUITE}"
fi
if [ -n "${CKPT_PATH:-}" ]; then
    export CKPT_PATH
fi
if [ -n "${ALIGN_PATH:-}" ]; then
    export ALIGN_PATH
fi
if [ -n "${LIBERO_ROOT:-}" ]; then
    export LIBERO_ROOT
fi
EXTRA_OVERRIDES=()
if [ -n "${META_UNNORM_KEY}" ]; then
    EXTRA_OVERRIDES+=(actor_rollout_ref.rollout.unnorm_key="$META_UNNORM_KEY")
fi
if [ -n "${NUM_TRIALS_PER_TASK:-}" ]; then
    EXTRA_OVERRIDES+=(data.num_trials_per_task="$NUM_TRIALS_PER_TASK")
fi

cd "$REPO_ROOT"
bash "$SCRIPT_DIR/run_openvla_oft_rl_libero_lora_dense.sh" \
    actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH" \
    actor_rollout_ref.rollout.experiment_name="$EVAL_EXPERIMENT_NAME" \
    trainer.experiment_name="$EVAL_EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.validation.target_rollouts="$TARGET_ROLLOUTS" \
    trainer.validation.start_index="$START_INDEX" \
    trainer.validation.save_video="$SAVE_VIDEO" \
    trainer.validation.video_every_n_calls=1 \
    trainer.validation.video_max_episodes="$VIDEO_MAX_EPISODES" \
    trainer.validation.video_per_task_limit="$VIDEO_PER_TASK_LIMIT" \
    trainer.validation.video_frame_stride="$VIDEO_FRAME_STRIDE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    trainer.wandb_mode="$WANDB_MODE" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@"
