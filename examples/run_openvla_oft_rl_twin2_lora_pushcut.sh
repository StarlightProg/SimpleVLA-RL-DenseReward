#!/bin/bash
set -euo pipefail
set -x

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export ROBOT_PLATFORM=ALOHA
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
DATASET_NAME="${DATASET_NAME:-move_can_pot}"
REWARD_MODE="${REWARD_MODE:-terminal}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-twin2-${DATASET_NAME}-lora-${REWARD_MODE}}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/../openvla_model}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"
WANDB_MODE="${WANDB_MODE:-offline}"
TRAJ_MINI_BATCH_SIZE="${TRAJ_MINI_BATCH_SIZE:-8}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1000}"
N_SAMPLES="${N_SAMPLES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-${NUM_GPUS}}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-32}"
VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-null}"
VAL_ONLY="${VAL_ONLY:-False}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [ "${DATASET_NAME}" != "move_can_pot" ]; then
    echo "This pushcut launcher supports DATASET_NAME=move_can_pot only for v1." >&2
    exit 1
fi

if [ "${LORA_RANK}" -le 0 ]; then
    echo "LORA_RANK must be positive; pushcut reproduction is LoRA-only." >&2
    exit 1
fi

case "${REWARD_MODE}" in
  terminal)
    SUBGOAL_ARGS=(reward.subgoal.enabled=False)
    ;;
  phase)
    SUBGOAL_ARGS=(
      reward.subgoal.enabled=True
      reward.subgoal.mode=add
      reward.subgoal.log=True
      reward.subgoal.weights.subgoal_progress=0.0
      reward.subgoal.weights.phase_transition=0.3
      reward.subgoal.weights.terminal_success=0.0
    )
    ;;
  dense)
    SUBGOAL_ARGS=(
      reward.subgoal.enabled=True
      reward.subgoal.mode=add
      reward.subgoal.log=True
      reward.subgoal.weights.subgoal_progress=0.2
      reward.subgoal.weights.phase_transition=0.3
      reward.subgoal.weights.terminal_success=0.0
    )
    ;;
  *)
    echo "REWARD_MODE must be one of: terminal, phase, dense" >&2
    exit 1
    ;;
esac

if [ ! -f "${SFT_MODEL_PATH}/config.json" ]; then
    echo "Missing OpenVLA-OFT checkpoint config: ${SFT_MODEL_PATH}/config.json" >&2
    exit 1
fi

if [ ! -f "${SFT_MODEL_PATH}/dataset_statistics.json" ]; then
    echo "Missing action normalization statistics: ${SFT_MODEL_PATH}/dataset_statistics.json" >&2
    exit 1
fi

mkdir -p "${CKPT_PATH}"
bash "${REPO_ROOT}/examples/overwrite_vla_ckpt_utils.sh" "${SFT_MODEL_PATH}"

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=robotwin2_${DATASET_NAME} \
    data.num_trials_per_task=${NUM_TRIALS_PER_TASK} \
    data.n_samples=${N_SAMPLES} \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.model.vla=${VLA_NAME} \
    actor_rollout_ref.model.action_token_len=14 \
    actor_rollout_ref.model.action_chunks_len=25 \
    actor_rollout_ref.model.resume=False \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=llm-projector \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=${TRAJ_MINI_BATCH_SIZE} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.twin2_task_config=demo_randomized \
    actor_rollout_ref.rollout.twin2_instruction_type=seen \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=True \
    actor_rollout_ref.rollout.val_micro_batch_size=${VAL_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.experiment_name=${EXPERIMENT_NAME} \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=robotwin2_${DATASET_NAME}_1k \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=robotwin2_${DATASET_NAME} \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=${SFT_MODEL_PATH} \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.max_steps=${TRAIN_MAX_STEPS} \
    trainer.val_only=${VAL_ONLY} \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    "${SUBGOAL_ARGS[@]}" \
    trainer.runtime_env=${ALIGN_PATH} \
    trainer.wandb_mode=${WANDB_MODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    "$@"
