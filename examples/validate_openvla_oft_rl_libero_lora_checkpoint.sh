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
if [ "$#" -ge 2 ]; then
    shift 2
else
    shift 1
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
    "$@"
