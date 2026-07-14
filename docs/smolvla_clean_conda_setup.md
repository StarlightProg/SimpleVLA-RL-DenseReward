# SmolVLA Clean Conda Setup

This is the clean environment recipe tested on this machine on July 13, 2026.

It is intended for:

- local smoke/debug machine: RTX 4060 8GB, NVIDIA driver 590.48.01, CUDA driver 13.1
- remote training machine: 2x NVIDIA L40, expected 48GB VRAM each

The local 8GB machine can validate imports and unit tests. It can start SmolVLA rollout/training, but actor update can OOM because the desktop already uses about 1.3GB VRAM. Use the 2xL40 machine for the real 5-step and long training smoke.

## Paths

Commands assume:

```bash
REPO_ROOT=/home/tamamonomaepc/SimpleVLA-RL-BatchSliceFix
LIBERO_ROOT=/home/tamamonomaepc/liboft/LIBERO
LEROBOT_ROOT=/home/tamamonomaepc/lerobot
ENV_NAME=simplevla-smolvla-clean
```

On the remote 2xL40 machine, change these paths to your remote locations, for example:

```bash
REPO_ROOT=$HOME/SimpleVLA-RL-BatchSliceFix
LIBERO_ROOT=$HOME/liboft/LIBERO
LEROBOT_ROOT=$HOME/lerobot
ENV_NAME=simplevla-smolvla-l40
```

## 1. Download Code

```bash
cd "$HOME"
git clone <your-simplevla-repo-url> SimpleVLA-RL-BatchSliceFix
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git liboft
git clone https://github.com/huggingface/lerobot.git lerobot
```

If this repo or LeRobot has local patches, copy those exact directories to the remote instead of cloning fresh.

## 2. Create Env

```bash
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel
```

## 3. Install PyTorch

For this local RTX 4060 with driver CUDA 13.1, the tested versions are:

```bash
pip install torch==2.11.0 torchvision==0.26.0
```

This installs CUDA 13 PyTorch wheels from normal PyPI:

```text
torch 2.11.0+cu130
torchvision 0.26.0+cu130
```

For the remote 2xL40, use the same command if `nvidia-smi` shows a recent driver with CUDA 13.x support. If the remote driver only supports CUDA 12.x, use the official CUDA 12.8 PyTorch index instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Expected on 2xL40:

```text
cuda available: True
gpu count: 2
0 NVIDIA L40
1 NVIDIA L40
```

## 4. Install Training Dependencies

```bash
pip install \
  numpy==2.2.6 \
  ray[default]==2.56.0 \
  transformers==5.5.4 accelerate==1.13.0 peft==0.19.1 safetensors==0.7.0 \
  hydra-core==1.3.4 omegaconf==2.3.1 codetiming==1.4.0 einops==0.8.2 tensordict==0.13.0 \
  wandb==0.24.2 tqdm pandas scipy \
  datasets diffusers sentencepiece pyarrow dill multiprocess xxhash numba \
  bddl easydict json-numpy jsonlines av==15.1.0 nltk gym-notices num2words==0.5.14 \
  gymnasium==1.3.0 gym==0.25.2 mujoco==3.8.1 robosuite==1.4.0 \
  imageio==2.37.3 imageio-ffmpeg==0.6.0 \
  opencv-python-headless==4.13.0.92 h5py matplotlib pytest timm \
  tensorflow==2.20.0
```

TensorFlow is used by the LIBERO/OpenVLA image preprocessing utilities. It may print:

```text
Could not find cuda drivers on your machine, GPU will not be used
```

That is acceptable. TensorFlow should stay on CPU; PyTorch is what must see CUDA.

`robosuite==1.4.0` may also install `opencv-python`; in the tested clean env `cv2.__version__` resolves to `5.0.0`. That is OK as long as `python -m pip check` is clean and `import cv2` works.

After TensorFlow, restore pins that TensorFlow may upgrade:

```bash
pip install --no-deps numpy==2.2.6 packaging==25.0 setuptools==80.10.2 protobuf==6.33.6
```

## 5. Install LeRobot and LIBERO

Use editable installs so the repo gets the same SmolVLA code path as development:

```bash
pip install -e "$LEROBOT_ROOT"
pip install -e "$LIBERO_ROOT"
```

## 6. Verify Environment

```bash
export PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT"
export MUJOCO_GL=egl
export ROBOT_PLATFORM=LIBERO

python - <<'PY'
import importlib
import torch
import av
import tensorflow as tf

mods = [
    "torch", "torchvision", "ray", "tensordict", "tensorflow",
    "transformers", "accelerate", "peft", "safetensors",
    "hydra", "omegaconf", "codetiming", "einops", "wandb",
    "datasets", "diffusers", "sentencepiece", "pyarrow", "dill",
    "multiprocess", "xxhash", "numba", "bddl", "easydict",
    "json_numpy", "jsonlines", "av", "nltk", "gymnasium", "gym",
    "mujoco", "robosuite", "cv2", "h5py", "matplotlib",
    "lerobot", "libero", "verl",
]

for name in mods:
    mod = importlib.import_module(name)
    print("OK", name, getattr(mod, "__version__", "no __version__"))

from lerobot.policies.smolvla import SmolVLAPolicy
from libero.libero.envs import OffScreenRenderEnv

print("SmolVLAPolicy:", SmolVLAPolicy)
print("OffScreenRenderEnv:", OffScreenRenderEnv)
print("av.option:", hasattr(av, "option"))
print("torch cuda:", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("tensorflow:", tf.__version__)
PY
```

Then:

```bash
pip check
```

Expected:

```text
No broken requirements found.
```

## 7. Run Repo Tests

```bash
cd "$REPO_ROOT"
PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT" python -m pytest tests -q
```

Verified locally in `simplevla-smolvla-clean`:

```text
25 passed
```

## 8. Local 8GB Smoke

This local machine can start SmolVLA training, load the model, create LIBERO envs, collect rollout data, and enter actor update. On the RTX 4060 8GB desktop, actor update can still OOM because only about 6.3GB is available to the training process after desktop/GUI memory.

Use this only as a dependency/startup check:

```bash
cd "$REPO_ROOT"
conda activate "$ENV_NAME"
ray stop --force || true

CKPT_PATH=/tmp/smolvla_clean_env_smoke \
WANDB_MODE=disabled \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=100 \
VAL_BEFORE_TRAIN=False \
TRAIN_BATCH_SIZE=1 \
N_SAMPLES=1 \
NUM_IMAGES_IN_INPUT=1 \
LORA_RANK=1 \
LORA_ALPHA=1 \
MAX_EPISODE_STEPS=2 \
bash examples/run_smolvla_rl_libero_lora_8gb.sh \
  trainer.final_save_after_train=False \
  trainer.final_val_after_train=False
```

On this local device, reaching model load and actor update confirms the setup is complete, even if the final actor update OOMs.

## 9. 2xL40 Five-Step Training Smoke

Run this on the remote 2xL40 device. This is the real "training starts and does not crash" test.

```bash
cd "$REPO_ROOT"
conda activate "$ENV_NAME"
ray stop --force || true

export PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT"
export MUJOCO_GL=egl
export ROBOT_PLATFORM=LIBERO
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=online
export WANDB_API_KEY='your_new_wandb_key'

NUM_GPUS=2 \
CKPT_PATH="$REPO_ROOT/checkpoints_smolvla_l40" \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  data.train_batch_size=2 \
  data.n_samples=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  trainer.total_training_steps=5 \
  trainer.save_freq=5 \
  trainer.test_freq=5
```

Success signs:

```text
Started a local Ray instance
Loading SmolVLA policy
collected ... rollouts
update_actor
actor/flow_loss
actor/grad_norm
Saved LoRA adapter
```

Check checkpoint:

```bash
find "$REPO_ROOT/checkpoints_smolvla_l40" -path '*lora_adapter/adapter_model.safetensors' -print
```

## 10. Long 2xL40 Training

After the five-step smoke passes:

```bash
cd "$REPO_ROOT"
conda activate "$ENV_NAME"
ray stop --force || true

NUM_GPUS=2 \
CKPT_PATH="$REPO_ROOT/checkpoints_smolvla_l40" \
WANDB_MODE=online \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  data.train_batch_size=2 \
  data.n_samples=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10
```

## 11. Terminal Reward Only

```bash
NUM_GPUS=2 \
CKPT_PATH="$REPO_ROOT/checkpoints_smolvla_l40_terminal" \
WANDB_MODE=online \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  reward.subgoal.enabled=False \
  reward.subgoal.mode=disabled
```

## 12. Full-Model Test

LoRA is recommended first. For full-model smoke:

```bash
NUM_GPUS=2 \
CKPT_PATH="$REPO_ROOT/checkpoints_smolvla_l40_full" \
WANDB_MODE=online \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  trainer.total_training_steps=5 \
  trainer.save_freq=5 \
  trainer.test_freq=5
```

Expected checkpoint sizes:

- LoRA: usually tens of MB.
- Full SmolVLA: approximately 2-3GB per checkpoint.

## 13. Troubleshooting

If `ModuleNotFoundError: tensordict`:

```bash
pip install tensordict==0.13.0
```

If `av has no attribute option`:

```bash
pip install --force-reinstall av==15.1.0
```

If `No module named robosuite` or `OffScreenRenderEnv is not defined`:

```bash
pip install robosuite==1.4.0 gym==0.25.2
```

If `num2words is required to run SmolVLM processor`:

```bash
pip install num2words==0.5.14
```

If TensorFlow says CUDA drivers are missing:

- Ignore it if PyTorch CUDA works.
- Verify PyTorch instead:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

If the local 8GB machine OOMs during actor update:

- This is expected on the RTX 4060 desktop.
- Run the five-step smoke on the 2xL40 machine.
- Do not use the local OOM as evidence that the L40 setup is broken.
