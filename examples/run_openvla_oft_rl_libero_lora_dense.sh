# Best-effort: free GPU memory from a previous crashed Ray session (avoids phantom OOM on resume).
ray stop --force 2>/dev/null || true
sleep 1

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL=egl
export VERL_FSDP_SUMMON_OFFLOAD_CPU=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-libero-lora-dense-020-phase-03-2L40-grad-clip-05}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/../openvla_model}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/../checkpoints}"
DATASET_NAME="${DATASET_NAME:-libero_spatial}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"
WANDB_MODE="${WANDB_MODE:-offline}"

RUNTIME_ALIGN_PATH="${TMPDIR:-${REPO_ROOT}/tmp}/align.runtime.json"
mkdir -p "$(dirname "${RUNTIME_ALIGN_PATH}")"
export ALIGN_PATH RUNTIME_ALIGN_PATH
python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["ALIGN_PATH"])
target = Path(os.environ["RUNTIME_ALIGN_PATH"])
config = json.loads(source.read_text(encoding="utf-8"))
env_vars = config.setdefault("env_vars", {})
if os.environ.get("WANDB_API_KEY"):
    env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
else:
    env_vars.pop("WANDB_API_KEY", None)
target.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
ALIGN_PATH="${RUNTIME_ALIGN_PATH}"

set -x

if [ ! -f "${SFT_MODEL_PATH}/config.json" ]; then
    echo "Missing OpenVLA-OFT checkpoint config: ${SFT_MODEL_PATH}/config.json" >&2
    exit 1
fi
if [ ! -f "${SFT_MODEL_PATH}/dataset_statistics.json" ]; then
    echo "Missing action normalization statistics: ${SFT_MODEL_PATH}/dataset_statistics.json" >&2
    exit 1
fi
mkdir -p "${CKPT_PATH}" "${REPO_ROOT}/rollouts"

bash examples/overwrite_vla_ckpt_utils.sh "${SFT_MODEL_PATH}"

# Silence robosuite macro warning in every worker by creating macros_private.py once.
ROBOSUITE_ROOT="$(python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("robosuite")
if spec and spec.submodule_search_locations:
    print(spec.submodule_search_locations[0])
PY
)"
if [ -n "$ROBOSUITE_ROOT" ] && [ ! -f "$ROBOSUITE_ROOT/macros_private.py" ]; then
    python "$ROBOSUITE_ROOT/scripts/setup_macros.py" >/dev/null 2>&1 || true
fi

# 2xL40 (~46GB): lora_rank 16 lowers peak VRAM vs 32 during rollout+update; raise if you have headroom.
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=4 \
    data.n_samples=4 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    data.oversample_factor=1 \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=llm-projector \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=1 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.rollout_dir=$REPO_ROOT/rollouts \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.libero_env_batch_size=1 \
    actor_rollout_ref.rollout.libero_mp_start_method=fork \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.validation.target_rollouts=60 \
    trainer.validation.save_video=False \
    trainer.validation.video_max_episodes=10 \
    trainer.validation.video_per_task_limit=1 \
    trainer.validation.video_frame_stride=8 \
    trainer.validation.video_every_n_calls=999 \
    trainer.total_epochs=1000 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    reward.subgoal.enabled=True \
    reward.subgoal.mode=add \
    reward.subgoal.log=True \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode="${WANDB_MODE}" \
    trainer.val_before_train=False \
    "$@"
