#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:?Set OPENVLA_OFT_ROOT to the OpenVLA-OFT checkout}"
RLDS_ROOT="${RLDS_ROOT:-${REPO_ROOT}/data/libero_cube_place/rlds}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/openvla-oft-cube-place}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-150005}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2000}"

if [[ "${ALLOW_LOW_VRAM:-0}" != "1" ]]; then
  python "${REPO_ROOT}/scripts/check_cube_place_resources.py" \
    --stage sft --num-gpus "${NPROC_PER_NODE}" --strict
fi

cd "${OPENVLA_OFT_ROOT}"
torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "${RLDS_ROOT}" \
  --dataset_name libero_cube_place_no_noops \
  --run_root_dir "${SFT_OUTPUT_ROOT}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --save_latest_checkpoint_only False \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --image_aug True \
  --lora_rank 32 \
  --merge_lora_during_training False \
  --run_id_note cube_place--8_acts_chunk--continuous_acts--L1--two_images--proprio \
  "$@"
