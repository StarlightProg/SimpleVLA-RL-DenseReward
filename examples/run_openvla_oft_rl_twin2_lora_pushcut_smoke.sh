#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATASET_NAME="${DATASET_NAME:-move_can_pot}"
export REWARD_MODE="${REWARD_MODE:-dense}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-twin2-${DATASET_NAME}-lora-${REWARD_MODE}-smoke-${SLURM_JOB_ID:-manual}}"
export CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
export PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_NODES="${NUM_NODES:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
export N_SAMPLES="${N_SAMPLES:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export TRAJ_MINI_BATCH_SIZE="${TRAJ_MINI_BATCH_SIZE:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
export VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export VAL_ONLY="${VAL_ONLY:-False}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"

if [ "${LORA_RANK}" -le 0 ]; then
    echo "Smoke run is LoRA-only; set LORA_RANK>0." >&2
    exit 2
fi

if [ "${NUM_GPUS}" != "1" ]; then
    echo "This smoke wrapper defaults to one GPU. Override batch sizes explicitly when NUM_GPUS=${NUM_GPUS}." >&2
fi

echo "Starting RoboTwin2 LoRA pushcut smoke run:"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  reward mode: ${REWARD_MODE}"
echo "  train max steps: ${TRAIN_MAX_STEPS}"
echo "  validation before train: ${VAL_BEFORE_TRAIN}"
echo "  save freq: ${SAVE_FREQ}"
echo "  checkpoint root: ${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

bash "${SCRIPT_DIR}/run_openvla_oft_rl_twin2_lora_pushcut.sh" \
    data.filter_accuracy=False \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    "$@"

RUN_DIR="${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [ ! -d "${RUN_DIR}/actor" ]; then
    echo "Smoke run finished but no actor checkpoint directory was found: ${RUN_DIR}/actor" >&2
    exit 1
fi

mapfile -t ADAPTER_CONFIGS < <(find "${RUN_DIR}/actor" -path "*/lora_adapter/adapter_config.json" -type f | sort)
if [ "${#ADAPTER_CONFIGS[@]}" -eq 0 ]; then
    echo "Smoke run finished but no LoRA adapter checkpoint was found under ${RUN_DIR}/actor." >&2
    exit 1
fi

latest_index="$((${#ADAPTER_CONFIGS[@]} - 1))"
latest_config="${ADAPTER_CONFIGS[${latest_index}]}"
latest_adapter_dir="$(dirname "${latest_config}")"
if [ ! -f "${latest_adapter_dir}/adapter_model.safetensors" ] && [ ! -f "${latest_adapter_dir}/adapter_model.bin" ]; then
    echo "Found ${latest_adapter_dir}, but it has no adapter_model.safetensors or adapter_model.bin." >&2
    exit 1
fi

echo "Smoke run completed and saved LoRA adapter:"
echo "  ${latest_adapter_dir}"
