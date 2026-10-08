#!/usr/bin/env bash
set -euo pipefail

ray stop --force >/dev/null 2>&1 || true

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export VERL_FSDP_SUMMON_OFFLOAD_CPU="${VERL_FSDP_SUMMON_OFFLOAD_CPU:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

REPO_ROOT="${REPO_ROOT:-/workspace/SimpleVLA-RL-BatchSliceFix}"
UNITY_WORK_ROOT="${UNITY_WORK_ROOT:-/workspace}"
VERL_ROOT="${VERL_ROOT:-/opt/verl-upstream}"
LIBERO_ROOT="${LIBERO_ROOT:-/opt/LIBERO}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-/opt/openvla-oft}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${UNITY_WORK_ROOT}/openvla_model}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
HF_HOME="${HF_HOME:-${UNITY_WORK_ROOT}/cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${CKPT_PATH}/wandb}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${UNITY_WORK_ROOT}/LIBERO/libero/datasets}"
JOB_TMPDIR="${JOB_TMPDIR:-/tmp/simplevla-${SLURM_JOB_ID:-$$}}"

case "${TMPDIR:-}" in
    ""|/tmp|/tmp/)
        TMPDIR="/tmp/svla-${SLURM_JOB_ID:-$$}"
        ;;
esac
RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}}"
LIBERO_EGL_INIT_LOCK="${LIBERO_EGL_INIT_LOCK:-True}"
LIBERO_EGL_INIT_LOCK_PATH="${LIBERO_EGL_INIT_LOCK_PATH:-${RAY_TMPDIR}/libero_egl_init.lock}"
LIBERO_EGL_INIT_STAGGER_SECONDS="${LIBERO_EGL_INIT_STAGGER_SECONDS:-2.0}"
LIBERO_MP_START_METHOD="${LIBERO_MP_START_METHOD:-spawn}"
MUJOCO_VERSION="${MUJOCO_VERSION:-2.3.7}"
ALLOW_MUJOCO_3="${ALLOW_MUJOCO_3:-0}"

PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
DATASET_NAME="${DATASET_NAME:-libero_spatial}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-libero-lora-dense-docker-${SLURM_JOB_ID:-manual}}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/run_openvla_oft_rl_libero_lora_dense.sh}"

export REPO_ROOT UNITY_WORK_ROOT VERL_ROOT LIBERO_ROOT OPENVLA_OFT_ROOT
export SFT_MODEL_PATH CKPT_PATH HF_HOME WANDB_DIR JOB_TMPDIR TMPDIR RAY_TMPDIR
export LIBERO_CONFIG_PATH LIBERO_DATASET_DIR
export LIBERO_EGL_INIT_LOCK LIBERO_EGL_INIT_LOCK_PATH LIBERO_EGL_INIT_STAGGER_SECONDS
export LIBERO_MP_START_METHOD MUJOCO_VERSION ALLOW_MUJOCO_3
export PROJECT_NAME DATASET_NAME VLA_NAME NUM_GPUS NUM_NODES EXPERIMENT_NAME
export PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p \
    "${CKPT_PATH}" \
    "${HF_HOME}" \
    "${WANDB_DIR}" \
    "${LIBERO_CONFIG_PATH}" \
    "${LIBERO_DATASET_DIR}" \
    "${JOB_TMPDIR}" \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${REPO_ROOT}/rollouts"

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    local temp_root
    for temp_root in "${TMPDIR:-}" "${RAY_TMPDIR:-}"; do
        if [[ "${temp_root}" == /tmp/svla-* || "${temp_root}" == /tmp/simplevla-* ]]; then
            rm -rf "${temp_root}" >/dev/null 2>&1 || true
        fi
    done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

if [ ! -f "${REPO_ROOT}/${RUN_SCRIPT}" ]; then
    echo "Missing run script inside mounted repo: ${REPO_ROOT}/${RUN_SCRIPT}" >&2
    echo "Mount the repository to /workspace/SimpleVLA-RL-BatchSliceFix or set REPO_ROOT/RUN_SCRIPT." >&2
    exit 2
fi

echo "============================================================"
echo "Docker SimpleVLA runtime"
echo "Node:            $(hostname)"
echo "Repository:      ${REPO_ROOT}"
echo "Work root:       ${UNITY_WORK_ROOT}"
echo "veRL root:       ${VERL_ROOT}"
echo "LIBERO root:     ${LIBERO_ROOT}"
echo "OpenVLA-OFT:     ${OPENVLA_OFT_ROOT}"
echo "SFT model:       ${SFT_MODEL_PATH}"
echo "Checkpoint root: ${CKPT_PATH}"
echo "W&B dir:         ${WANDB_DIR}"
echo "LIBERO config:   ${LIBERO_CONFIG_PATH}/config.yaml"
echo "LIBERO datasets: ${LIBERO_DATASET_DIR}"
echo "Ray TMPDIR:      ${TMPDIR}"
echo "Ray temp root:   ${RAY_TMPDIR}"
echo "MuJoCo:          ${MUJOCO_VERSION} (ALLOW_MUJOCO_3=${ALLOW_MUJOCO_3})"
echo "Run script:      ${RUN_SCRIPT}"
echo "============================================================"
nvidia-smi

python /opt/simplevla/preflight.py

cd "${REPO_ROOT}"
ray stop --force >/dev/null 2>&1 || true
echo "Starting ${EXPERIMENT_NAME}"
bash "${RUN_SCRIPT}" "$@"
