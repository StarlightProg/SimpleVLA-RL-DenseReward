#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_MODEL_PATH="${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the merged OpenVLA-OFT cube-place checkpoint}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
REWARD_MODE="${REWARD_MODE:-sparse}"
WANDB_MODE="${WANDB_MODE:-offline}"

case "${REWARD_MODE}" in
  sparse)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-cube-place-sparse}"
    SUBGOAL_ARGS=(reward.subgoal.enabled=False)
    ;;
  dense)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-cube-place-dense}"
    SUBGOAL_ARGS=(
      reward.subgoal.enabled=True
      reward.subgoal.mode=add
      reward.subgoal.log=True
    )
    ;;
  *)
    echo "REWARD_MODE must be sparse or dense" >&2
    exit 2
    ;;
esac

if [[ "${ALLOW_LOW_VRAM:-0}" != "1" ]]; then
  python "${REPO_ROOT}/scripts/check_cube_place_resources.py" \
    --stage rl --num-gpus "${NUM_GPUS}" --strict
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true
export TF_CPP_MIN_LOG_LEVEL=3
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL=egl
export VERL_FSDP_SUMMON_OFFLOAD_CPU=0

cd "${REPO_ROOT}"
ray stop --force >/dev/null 2>&1 || true
bash examples/overwrite_vla_ckpt_utils.sh "${SFT_MODEL_PATH}"

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  data.task_suite_name=libero_cube_place \
  data.num_trials_per_task=16 \
  data.n_samples=4 \
  data.filter_accuracy=True \
  data.accuracy_lower_bound=0.0 \
  data.accuracy_upper_bound=1.0 \
  data.oversample_factor=1 \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
  actor_rollout_ref.model.vla=openvla-oft \
  actor_rollout_ref.model.action_token_len=7 \
  actor_rollout_ref.model.action_chunks_len=8 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules=llm-projector \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size="${NUM_GPUS}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.grad_clip=1 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.num_images_in_input=2 \
  actor_rollout_ref.actor.traj_mini_batch_size=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.num_images_in_input=2 \
  actor_rollout_ref.rollout.use_proprio=True \
  actor_rollout_ref.rollout.val_micro_batch_size=1 \
  actor_rollout_ref.rollout.temperature=1.6 \
  actor_rollout_ref.rollout.experiment_name="${EXPERIMENT_NAME}" \
  actor_rollout_ref.rollout.rollout_dir="${REPO_ROOT}/rollouts" \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.unnorm_key=libero_cube_place_no_noops \
  actor_rollout_ref.rollout.model_family=openvla \
  actor_rollout_ref.rollout.task_suite_name=libero_cube_place \
  actor_rollout_ref.rollout.num_steps_wait=10 \
  actor_rollout_ref.rollout.pretrained_checkpoint="${SFT_MODEL_PATH}" \
  actor_rollout_ref.rollout.center_crop=True \
  actor_rollout_ref.rollout.max_episode_steps=320 \
  actor_rollout_ref.rollout.max_prompt_length=256 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${NUM_GPUS}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.libero_env_batch_size=1 \
  actor_rollout_ref.rollout.libero_mp_start_method=fork \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${NUM_GPUS}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.adv_estimator=grpo \
  algorithm.adv_params.verifier_gamma=1.0 \
  algorithm.adv_params.reward_model_gamma=1.0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name=SimpleVLA-RL \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_PATH}/SimpleVLA-RL/${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes="${NUM_NODES}" \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.validation.target_rollouts=20 \
  trainer.validation.save_video=True \
  trainer.validation.video_max_episodes=5 \
  trainer.validation.video_per_task_limit=5 \
  trainer.validation.video_frame_stride=8 \
  trainer.total_epochs=1000 \
  trainer.val_only=False \
  trainer.runtime_env="${ALIGN_PATH}" \
  trainer.wandb_mode="${WANDB_MODE}" \
  trainer.val_before_train=False \
  "${SUBGOAL_ARGS[@]}" \
  "$@"
