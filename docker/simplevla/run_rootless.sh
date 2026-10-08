#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-simplevla-rl:cu121-openvla}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${WORKSPACE_ROOT}/openvla_model}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${CKPT_PATH}/wandb}"
LIBERO_DATASET_HOST="${LIBERO_DATASET_HOST:-${WORKSPACE_ROOT}/LIBERO/libero/datasets}"
NUM_GPUS="${NUM_GPUS:-2}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "all" ]; then
    DOCKER_GPUS="device=${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "${CKPT_PATH}" "${HF_HOME}" "${WANDB_DIR}" "${LIBERO_DATASET_HOST}"

docker run --rm -it \
    --gpus "${DOCKER_GPUS}" \
    --ipc=host \
    --shm-size="${DOCKER_SHM_SIZE:-64g}" \
    --ulimit stack=67108864 \
    -e NUM_GPUS="${NUM_GPUS}" \
    -e WANDB_MODE="${WANDB_MODE:-offline}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-docker-manual}" \
    -e SFT_MODEL_PATH=/workspace/openvla_model \
    -e CKPT_PATH=/workspace/SimpleVLA-RL-BatchSliceFix/checkpoints \
    -e HF_HOME=/workspace/SimpleVLA-RL-BatchSliceFix/cache/huggingface \
    -e WANDB_DIR=/workspace/SimpleVLA-RL-BatchSliceFix/checkpoints/wandb \
    -e LIBERO_DATASET_DIR=/workspace/LIBERO/libero/datasets \
    -e LIBERO_ENV_SMOKE_TEST="${LIBERO_ENV_SMOKE_TEST:-1}" \
    -v "${REPO_ROOT}:/workspace/SimpleVLA-RL-BatchSliceFix" \
    -v "${SFT_MODEL_PATH}:/workspace/openvla_model:ro" \
    -v "${LIBERO_DATASET_HOST}:/workspace/LIBERO/libero/datasets" \
    -w /workspace/SimpleVLA-RL-BatchSliceFix \
    "${IMAGE_NAME}" \
    "$@"
