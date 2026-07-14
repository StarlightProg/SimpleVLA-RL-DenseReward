# SmolVLA Remote Setup on 2x L40

This guide creates a fresh Conda environment for running SmolVLA RL training on a remote machine with two NVIDIA L40 GPUs.

The commands assume:

- repo path: `~/SimpleVLA-RL-BatchSliceFix`
- LIBERO source path: `~/liboft/LIBERO`
- Conda environment name: `simplevla-smolvla-l40`

Adjust paths if your remote machine uses a different layout.

## 1. Get the Code

Clone or copy this repo to the remote machine:

```bash
cd ~
git clone <your-simplevla-repo-url> SimpleVLA-RL-BatchSliceFix
```

Install LIBERO from source:

```bash
cd ~
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git liboft
```

## 2. Create the Conda Environment

```bash
conda create -n simplevla-smolvla-l40 python=3.12 -y
conda activate simplevla-smolvla-l40
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch

First check the remote NVIDIA driver:

```bash
nvidia-smi
```

If the driver supports CUDA 13, install the CUDA 13 nightly build:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

If CUDA 13 wheels are unavailable or the driver is older, use the CUDA 12.8 build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify that PyTorch sees both GPUs:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Expected result: `gpu_count: 2` and both devices should be L40 GPUs.

## 4. Install Python Dependencies

```bash
pip install \
  "ray[default]" \
  transformers accelerate peft safetensors \
  hydra-core omegaconf codetiming einops tensordict \
  wandb tqdm pandas numpy scipy \
  datasets diffusers sentencepiece pyarrow dill multiprocess xxhash numba \
  bddl easydict json-numpy jsonlines av==15.1.0 nltk gym-notices \
  gymnasium mujoco imageio imageio-ffmpeg \
  opencv-python-headless h5py matplotlib pytest
```

Install LeRobot with SmolVLA support:

```bash
pip install "lerobot[smolvla]"
```

If that extra is unavailable in your installed LeRobot release, use:

```bash
pip install lerobot
pip install diffusers sentencepiece tokenizers datasets
```

Install LIBERO:

```bash
cd ~/liboft/LIBERO
pip install -e .
```

If you already created the env and hit missing-package errors, repair it with:

```bash
pip install \
  datasets diffusers sentencepiece pyarrow dill multiprocess xxhash numba \
  bddl easydict json-numpy jsonlines av==15.1.0 nltk gym-notices
```

## 5. Configure Runtime Environment

```bash
cd ~/SimpleVLA-RL-BatchSliceFix

export PYTHONPATH=~/liboft/LIBERO:~/SimpleVLA-RL-BatchSliceFix
export MUJOCO_GL=egl
export ROBOT_PLATFORM=LIBERO
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
```

For WandB, do not store the API key in `align.json`. Set it in the shell:

```bash
export WANDB_API_KEY='your_new_wandb_key'
export WANDB_MODE=online
```

If you do not want WandB logging:

```bash
export WANDB_MODE=disabled
```

## 6. Smoke Test Imports

```bash
cd ~/SimpleVLA-RL-BatchSliceFix

python - <<'PY'
import torch
import ray
import transformers
import peft
import lerobot
import datasets
import diffusers
import sentencepiece
import pyarrow
import bddl
import json_numpy
from libero.libero import benchmark

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
print("ray:", ray.__version__)
print("LIBERO ok")
PY
```

## 7. Run Local Unit Tests

These tests check SmolVLA integration and OpenVLA compatibility logic without launching a full training job:

```bash
cd ~/SimpleVLA-RL-BatchSliceFix
python -m pytest tests -q
```

Expected result:

```text
25 passed
```

The exact warning count may differ.

## 8. Five-Step SmolVLA LoRA Smoke Training

Run this before a long training job. It checks rollout, validation, actor update, WandB logging, and checkpoint saving.

```bash
cd ~/SimpleVLA-RL-BatchSliceFix

ray stop --force || true

NUM_GPUS=2 \
CKPT_PATH=$PWD/checkpoints_smolvla_l40 \
WANDB_MODE=online \
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

Checkpoint output should appear under:

```text
checkpoints_smolvla_l40/SimpleVLA-RL/simplevla-rl-smolvla-libero-lora-dense/actor/
```

LoRA checkpoints are expected to be small, usually tens of MB depending on rank and target modules.

## 9. Long SmolVLA LoRA Training

After the five-step smoke test passes:

```bash
cd ~/SimpleVLA-RL-BatchSliceFix

ray stop --force || true

NUM_GPUS=2 \
CKPT_PATH=$PWD/checkpoints_smolvla_l40 \
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

## 10. Terminal Reward Only

The default SmolVLA launcher enables dense subgoal reward:

```text
reward.subgoal.enabled=True
reward.subgoal.mode=add
```

To train with normal terminal reward only:

```bash
NUM_GPUS=2 \
CKPT_PATH=$PWD/checkpoints_smolvla_l40_terminal \
WANDB_MODE=online \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  reward.subgoal.enabled=False \
  reward.subgoal.mode=disabled
```

## 11. Full-Model SmolVLA Training

LoRA is recommended first. For full-model training, disable LoRA:

```bash
NUM_GPUS=2 \
CKPT_PATH=$PWD/checkpoints_smolvla_l40_full \
WANDB_MODE=online \
bash examples/run_smolvla_rl_libero_lora_dense.sh \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  trainer.total_training_steps=5 \
  trainer.save_freq=5 \
  trainer.test_freq=5
```

Expected checkpoint size:

- LoRA checkpoint: usually tens of MB.
- Full SmolVLA checkpoint: approximately 2-3 GB per checkpoint, based on local full-save behavior.

## 12. Useful Checks During Training

Watch GPU memory:

```bash
watch -n 1 nvidia-smi
```

Find saved checkpoints:

```bash
find checkpoints_smolvla_l40 -maxdepth 5 -type d -name 'global_step_*' | sort
```

Check a LoRA adapter exists:

```bash
find checkpoints_smolvla_l40 -path '*lora_adapter/adapter_model.safetensors' -print
```

Inspect latest logs:

```bash
find wandb -maxdepth 2 -type f -name 'debug.log' -o -name 'output.log'
```

## Notes

- Keep OpenVLA defaults untouched; SmolVLA is selected explicitly by the SmolVLA launcher via `actor_rollout_ref.model.vla=smolvla`.
- SmolVLA uses an advantage-weighted flow-matching objective in this repo, not OpenVLA token PPO log-probs.
- If the five-step smoke test OOMs on 2xL40, reduce `data.train_batch_size` to `1`, keep `data.n_samples=4`, and retry.
- If validation is too slow, reduce `trainer.validation.target_rollouts` or increase `trainer.test_freq`.
