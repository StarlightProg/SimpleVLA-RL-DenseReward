#!/bin/bash
# Slurm wrapper for clusters that run Docker rootless with NVIDIA passthrough.
# Build once from the repository root:
#   bash docker/simplevla/build.sh
# Submit training:
#   sbatch docker/simplevla/simplevla_docker_2l40.slurm reward.subgoal.clip_dense_reward=0.2

#SBATCH --partition=gpu
#SBATCH --job-name=simplevla_docker
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --qos=long
#SBATCH --time=8-00:00:00
#SBATCH --output=simplevla_docker_%j.log
#SBATCH --error=simplevla_docker_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "${REPO_ROOT}/docker/simplevla/run_rootless.sh" ]; then
    echo "Submit this file from the SimpleVLA-RL-BatchSliceFix repository root." >&2
    exit 2
fi

cd "${REPO_ROOT}"

export IMAGE_NAME="${IMAGE_NAME:-simplevla-rl:cu121-openvla}"
export REPO_ROOT
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
export NUM_GPUS="${NUM_GPUS:-2}"
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-${WORKSPACE_ROOT}/openvla_model}"
export CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-${CKPT_PATH}/wandb}"
export LIBERO_DATASET_HOST="${LIBERO_DATASET_HOST:-${WORKSPACE_ROOT}/LIBERO/libero/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-libero-lora-dense-docker-${SLURM_JOB_ID:-manual}}"
export LIBERO_ENV_SMOKE_TEST="${LIBERO_ENV_SMOKE_TEST:-1}"

echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID:-manual}"
echo "Node:            $(hostname)"
echo "Repository:      ${REPO_ROOT}"
echo "Workspace:       ${WORKSPACE_ROOT}"
echo "Image:           ${IMAGE_NAME}"
echo "SFT model:       ${SFT_MODEL_PATH}"
echo "Checkpoint root: ${CKPT_PATH}"
echo "LIBERO datasets: ${LIBERO_DATASET_HOST}"
echo "W&B mode:        ${WANDB_MODE}"
echo "Experiment:      ${EXPERIMENT_NAME}"
echo "============================================================"

docker version
docker info | sed -n '1,80p'
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

bash docker/simplevla/run_rootless.sh "$@"
