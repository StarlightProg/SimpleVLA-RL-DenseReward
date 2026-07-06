#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/actor/global_step_N/lora_adapter [experiment_name]" >&2
    exit 1
fi

LORA_ADAPTER_PATH="$1"
EVAL_EXPERIMENT_NAME="${2:-$(basename "$(dirname "$LORA_ADAPTER_PATH")")-libero-spatial-videos}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_ROOT="$(cd "$(dirname "$LORA_ADAPTER_PATH")" && pwd)"
if [ "$#" -ge 2 ]; then
    shift 2
else
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
EXTRA_OVERRIDES=()
if [ -n "${META_UNNORM_KEY}" ]; then
    EXTRA_OVERRIDES+=(actor_rollout_ref.rollout.unnorm_key="$META_UNNORM_KEY")
fi

cd "$REPO_ROOT"
bash "$SCRIPT_DIR/run_openvla_oft_rl_libero_lora_dense.sh" \
    actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH" \
    actor_rollout_ref.rollout.experiment_name="$EVAL_EXPERIMENT_NAME" \
    trainer.experiment_name="$EVAL_EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.validation.target_rollouts=40 \
    trainer.validation.save_video=True \
    trainer.validation.video_every_n_calls=1 \
    trainer.validation.video_max_episodes=10 \
    trainer.validation.video_per_task_limit=1 \
    trainer.validation.video_frame_stride=8 \
    trainer.wandb_mode=disabled \
    "${EXTRA_OVERRIDES[@]}" \
    "$@"
