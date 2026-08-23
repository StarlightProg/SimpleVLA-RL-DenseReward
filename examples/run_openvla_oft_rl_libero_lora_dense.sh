#!/bin/bash
set -euo pipefail

# Best-effort: free GPU memory from a previous crashed Ray session (avoids phantom OOM on resume).
ray stop --force 2>/dev/null || true
sleep 1

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export VERL_FSDP_SUMMON_OFFLOAD_CPU="${VERL_FSDP_SUMMON_OFFLOAD_CPU:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplevla-rl-libero-lora-dense-020-phase-03-3L40-r16}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/openvla_model}"
if [ ! -f "${SFT_MODEL_PATH}/config.json" ] && [ -f "${REPO_ROOT}/../openvla_model/config.json" ]; then
    SFT_MODEL_PATH="${REPO_ROOT}/../openvla_model"
fi
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints}"
DATASET_NAME="${DATASET_NAME:-libero_spatial}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_NODES="${NUM_NODES:-1}"
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"
WANDB_MODE="${WANDB_MODE:-offline}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${CKPT_PATH}/wandb}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${LIBERO_ROOT}/libero/datasets}"
case "${TMPDIR:-}" in
    ""|/tmp|/tmp/)
        TMPDIR="/tmp/svla-${SLURM_JOB_ID:-$$}"
        ;;
esac
RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}}"
LIBERO_EGL_INIT_LOCK="${LIBERO_EGL_INIT_LOCK:-True}"
LIBERO_EGL_INIT_LOCK_PATH="${LIBERO_EGL_INIT_LOCK_PATH:-${RAY_TMPDIR}/libero_egl_init.lock}"
LIBERO_EGL_INIT_STAGGER_SECONDS="${LIBERO_EGL_INIT_STAGGER_SECONDS:-2.0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-6}"
N_SAMPLES="${N_SAMPLES:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-6}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-${NUM_GPUS}}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-${NUM_GPUS}}"
VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-1}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
LIBERO_ENV_BATCH_SIZE="${LIBERO_ENV_BATCH_SIZE:-1}"
LIBERO_ENV_INIT_TIMEOUT="${LIBERO_ENV_INIT_TIMEOUT:-180}"
LIBERO_ENV_STEP_TIMEOUT="${LIBERO_ENV_STEP_TIMEOUT:-60}"
LIBERO_ENV_INIT_MAX_RETRIES="${LIBERO_ENV_INIT_MAX_RETRIES:-3}"
LIBERO_ENV_INIT_RETRY_SLEEP="${LIBERO_ENV_INIT_RETRY_SLEEP:-2.0}"
LIBERO_MP_START_METHOD="${LIBERO_MP_START_METHOD:-spawn}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
VIDEO_MAX_EPISODES="${VIDEO_MAX_EPISODES:-10}"
VIDEO_PER_TASK_LIMIT="${VIDEO_PER_TASK_LIMIT:-1}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-8}"
VIDEO_EVERY_N_CALLS="${VIDEO_EVERY_N_CALLS:-999}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
TARGET_ROLLOUTS="${TARGET_ROLLOUTS:-60}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1000}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-null}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
VAL_ONLY="${VAL_ONLY:-False}"
DENSE_REWARD_WEIGHT="${DENSE_REWARD_WEIGHT:-0.2}"
PHASE_TRANSITION_WEIGHT="${PHASE_TRANSITION_WEIGHT:-0.3}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
TASK_SAMPLING_ENABLED="${TASK_SAMPLING_ENABLED:-False}"
TASK_SAMPLING_UNIFORM_FRACTION="${TASK_SAMPLING_UNIFORM_FRACTION:-0.7}"
TASK_SAMPLING_MIN_TASK_PROBABILITY="${TASK_SAMPLING_MIN_TASK_PROBABILITY:-0.05}"
TASK_SAMPLING_EMA_MOMENTUM="${TASK_SAMPLING_EMA_MOMENTUM:-0.8}"
TASK_SAMPLING_DEFAULT_SUCCESS="${TASK_SAMPLING_DEFAULT_SUCCESS:-0.5}"
TASK_SAMPLING_SEED="${TASK_SAMPLING_SEED:-0}"

export LIBERO_ROOT LIBERO_CONFIG_PATH LIBERO_DATASET_DIR
export PROJECT_NAME EXPERIMENT_NAME SFT_MODEL_PATH CKPT_PATH
export DATASET_NAME VLA_NAME NUM_GPUS NUM_NODES ALIGN_PATH WANDB_MODE HF_HOME WANDB_DIR TMPDIR RAY_TMPDIR
export LIBERO_EGL_INIT_LOCK LIBERO_EGL_INIT_LOCK_PATH LIBERO_EGL_INIT_STAGGER_SECONDS
export NUM_TRIALS_PER_TASK N_SAMPLES TRAIN_BATCH_SIZE VAL_BATCH_SIZE
export PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE
export VAL_MICRO_BATCH_SIZE ROLLOUT_MICRO_BATCH_SIZE LIBERO_ENV_BATCH_SIZE
export LIBERO_ENV_INIT_TIMEOUT LIBERO_ENV_STEP_TIMEOUT
export LIBERO_ENV_INIT_MAX_RETRIES LIBERO_ENV_INIT_RETRY_SLEEP LIBERO_MP_START_METHOD
export SAVE_VIDEO VIDEO_MAX_EPISODES VIDEO_PER_TASK_LIMIT VIDEO_FRAME_STRIDE VIDEO_EVERY_N_CALLS
export LORA_RANK LORA_ALPHA TARGET_ROLLOUTS GPU_MEMORY_UTILIZATION
export SAVE_FREQ TEST_FREQ TOTAL_EPOCHS TRAIN_MAX_STEPS VAL_BEFORE_TRAIN VAL_ONLY
export DENSE_REWARD_WEIGHT PHASE_TRANSITION_WEIGHT
export ROLLOUT_TEMPERATURE
export TASK_SAMPLING_ENABLED TASK_SAMPLING_UNIFORM_FRACTION
export TASK_SAMPLING_MIN_TASK_PROBABILITY TASK_SAMPLING_EMA_MOMENTUM
export TASK_SAMPLING_DEFAULT_SUCCESS TASK_SAMPLING_SEED
export PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

write_libero_config() {
    mkdir -p "${LIBERO_CONFIG_PATH}" "${LIBERO_DATASET_DIR}"
    python - <<'PY'
import os
from pathlib import Path

config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
benchmark_root = Path(os.environ["LIBERO_ROOT"]) / "libero" / "libero"
dataset_dir = Path(os.environ["LIBERO_DATASET_DIR"])
config_dir.mkdir(parents=True, exist_ok=True)
dataset_dir.mkdir(parents=True, exist_ok=True)
(config_dir / "config.yaml").write_text(
    "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {dataset_dir}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    ),
    encoding="utf-8",
)
print(f"LIBERO config written to {config_dir / 'config.yaml'}")
PY
}

write_libero_config

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

if [ $((PPO_MINI_BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
if [ $((PPO_MICRO_BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    echo "PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
if [ $((LOG_PROB_MICRO_BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    echo "LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
if [ $((PPO_MICRO_BATCH_SIZE / NUM_GPUS)) -ne 1 ]; then
    echo "Robotics actor requires per-rank PPO micro batch 1; set PPO_MICRO_BATCH_SIZE=NUM_GPUS." >&2
    exit 1
fi

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
env_vars.update(
    {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "NCCL_ASYNC_ERROR_HANDLING": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
        "RAY_memory_monitor_refresh_ms": "0",
        "TMPDIR": os.environ["TMPDIR"],
        "RAY_TMPDIR": os.environ["RAY_TMPDIR"],
        "LIBERO_EGL_INIT_LOCK": os.environ.get("LIBERO_EGL_INIT_LOCK", "True"),
        "LIBERO_EGL_INIT_LOCK_PATH": os.environ["LIBERO_EGL_INIT_LOCK_PATH"],
        "LIBERO_EGL_INIT_STAGGER_SECONDS": os.environ.get("LIBERO_EGL_INIT_STAGGER_SECONDS", "2.0"),
        "LIBERO_MP_START_METHOD": os.environ.get("LIBERO_MP_START_METHOD", "spawn"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
        "TOKENIZERS_PARALLELISM": "true",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3"),
        "ROBOT_PLATFORM": "LIBERO",
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl"),
        "LIBERO_CONFIG_PATH": os.environ["LIBERO_CONFIG_PATH"],
        "LIBERO_DATASET_DIR": os.environ["LIBERO_DATASET_DIR"],
        "VERL_FSDP_SUMMON_OFFLOAD_CPU": "0",
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
    }
)
if os.environ.get("HF_HOME"):
    env_vars["HF_HOME"] = os.environ["HF_HOME"]
if os.environ.get("WANDB_DIR"):
    env_vars["WANDB_DIR"] = os.environ["WANDB_DIR"]
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

python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["SFT_MODEL_PATH"])
index_path = root / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"Missing checkpoint weight index: {index_path}")

index = json.loads(index_path.read_text(encoding="utf-8"))
weights = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in weights if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit("Missing or empty checkpoint weight files: " + ", ".join(missing))

try:
    from safetensors import safe_open
    for name in weights:
        with safe_open(str(root / name), framework="pt"):
            pass
except Exception as exc:
    raise SystemExit(f"Checkpoint safetensors validation failed: {exc}") from exc
PY

python - <<'PY'
import importlib
import sys

checks = [
    ("requests", "requests"),
    ("pydantic", "pydantic"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("hydra-core", "hydra"),
    ("omegaconf", "omegaconf"),
    ("codetiming", "codetiming"),
    ("dill", "dill"),
    ("PyYAML", "yaml"),
    ("pandas", "pandas"),
    ("pylatexenc", "pylatexenc"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("sentencepiece", "sentencepiece"),
    ("timm", "timm"),
    ("tokenizers", "tokenizers"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ray", "ray"),
    ("peft", "peft"),
    ("transformers", "transformers"),
    ("wandb", "wandb"),
    ("libero", "libero.libero"),
    ("SimpleVLA PPO entry", "verl.trainer.main_ppo"),
]
missing = []
for label, module in checks:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{label} ({module}): {exc}")

if missing:
    print("Dependency preflight failed. Missing or broken imports:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)

import tokenizers
import transformers
if transformers.__version__ != "4.40.1" or tokenizers.__version__ != "0.19.1":
    raise SystemExit(
        "OpenVLA-OFT version preflight failed: "
        f"transformers=={transformers.__version__}, tokenizers=={tokenizers.__version__}; "
        "expected transformers==4.40.1 and tokenizers==0.19.1."
    )

print("Dependency preflight OK.")
PY

mkdir -p "${CKPT_PATH}" "${WANDB_DIR}" "${REPO_ROOT}/rollouts"

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

# 3xL40/L40S (~46GB each): global micro/log-prob batches are NUM_GPUS so each
# FSDP rank sees micro batch 1, matching the robotics actor implementation.
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=$NUM_TRIALS_PER_TASK \
    data.n_samples=$N_SAMPLES \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    data.oversample_factor=1 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.task_sampling.enabled=$TASK_SAMPLING_ENABLED \
    data.task_sampling.mode=balanced_hard \
    data.task_sampling.uniform_fraction=$TASK_SAMPLING_UNIFORM_FRACTION \
    data.task_sampling.min_task_probability=$TASK_SAMPLING_MIN_TASK_PROBABILITY \
    data.task_sampling.ema_momentum=$TASK_SAMPLING_EMA_MOMENTUM \
    data.task_sampling.default_success=$TASK_SAMPLING_DEFAULT_SUCCESS \
    data.task_sampling.seed=$TASK_SAMPLING_SEED \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=llm-projector \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
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
    actor_rollout_ref.rollout.val_micro_batch_size=$VAL_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.rollout_dir=$REPO_ROOT/rollouts \
    actor_rollout_ref.rollout.micro_batch_size=$ROLLOUT_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.libero_env_batch_size=$LIBERO_ENV_BATCH_SIZE \
    actor_rollout_ref.rollout.libero_env_init_timeout=$LIBERO_ENV_INIT_TIMEOUT \
    actor_rollout_ref.rollout.libero_env_step_timeout=$LIBERO_ENV_STEP_TIMEOUT \
    actor_rollout_ref.rollout.libero_env_init_max_retries=$LIBERO_ENV_INIT_MAX_RETRIES \
    actor_rollout_ref.rollout.libero_env_init_retry_sleep=$LIBERO_ENV_INIT_RETRY_SLEEP \
    actor_rollout_ref.rollout.libero_egl_init_lock=$LIBERO_EGL_INIT_LOCK \
    actor_rollout_ref.rollout.libero_egl_init_lock_path=$LIBERO_EGL_INIT_LOCK_PATH \
    actor_rollout_ref.rollout.libero_egl_init_stagger_seconds=$LIBERO_EGL_INIT_STAGGER_SECONDS \
    actor_rollout_ref.rollout.libero_mp_start_method=$LIBERO_MP_START_METHOD \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.validation.target_rollouts=$TARGET_ROLLOUTS \
    trainer.validation.save_video=$SAVE_VIDEO \
    trainer.validation.video_max_episodes=$VIDEO_MAX_EPISODES \
    trainer.validation.video_per_task_limit=$VIDEO_PER_TASK_LIMIT \
    trainer.validation.video_frame_stride=$VIDEO_FRAME_STRIDE \
    trainer.validation.video_every_n_calls=$VIDEO_EVERY_N_CALLS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.max_steps=$TRAIN_MAX_STEPS \
    trainer.val_only=$VAL_ONLY \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    reward.subgoal.enabled=True \
    reward.subgoal.mode=add \
    reward.subgoal.log=True \
    reward.subgoal.weights.subgoal_progress=$DENSE_REWARD_WEIGHT \
    reward.subgoal.weights.phase_transition=$PHASE_TRANSITION_WEIGHT \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode="${WANDB_MODE}" \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    "$@"
