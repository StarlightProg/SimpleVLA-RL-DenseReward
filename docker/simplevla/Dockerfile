FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_VERSION=Miniconda3-py310_24.11.1-0-Linux-x86_64
ARG PYTHON_VERSION=3.10.14
ARG VERL_REF=v0.2.x
ARG MUJOCO_VERSION=2.3.7
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG PYTORCH_TORCH_SPEC=torch==2.4.0+cu121
ARG PYTORCH_TORCHVISION_SPEC=torchvision==0.19.0+cu121
ARG PYTORCH_TORCHAUDIO_SPEC=torchaudio==2.4.0+cu121

ENV CONDA_ROOT=/opt/conda \
    CONDA_ENV_NAME=simplevla-rl \
    CUDA_HOME=/usr/local/cuda \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=true \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_DEBUG=WARN \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    ROBOT_PLATFORM=LIBERO \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    VERL_FSDP_SUMMON_OFFLOAD_CPU=0 \
    HF_HUB_DISABLE_XET=1 \
    MUJOCO_VERSION=${MUJOCO_VERSION} \
    ALLOW_MUJOCO_3=0 \
    PYTORCH_CUDA_VERSION=12.1 \
    PYTORCH_CUDA_TAG=cu121 \
    PYTORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL} \
    PYTORCH_TORCH_SPEC=${PYTORCH_TORCH_SPEC} \
    PYTORCH_TORCHVISION_SPEC=${PYTORCH_TORCHVISION_SPEC} \
    PYTORCH_TORCHAUDIO_SPEC=${PYTORCH_TORCHAUDIO_SPEC} \
    VERL_ROOT=/opt/verl-upstream \
    LIBERO_ROOT=/opt/LIBERO \
    OPENVLA_OFT_ROOT=/opt/openvla-oft

ENV PATH=${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin:${CONDA_ROOT}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    gcc-12 \
    g++-12 \
    gcc-11 \
    g++-11 \
    cmake \
    ninja-build \
    pkg-config \
    patchelf \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libglib2.0-0 \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fL "https://repo.anaconda.com/miniconda/${CONDA_VERSION}.sh" -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}" \
    && rm -f /tmp/miniconda.sh \
    && conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" \
    && conda clean -afy

RUN git clone --branch "${VERL_REF}" --depth 1 https://github.com/volcengine/verl.git "${VERL_ROOT}" \
    && git clone --depth 1 https://github.com/moojink/openvla-oft.git "${OPENVLA_OFT_ROOT}" \
    && git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_ROOT}"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --upgrade \
        "${PYTORCH_TORCH_SPEC}" \
        "${PYTORCH_TORCHVISION_SPEC}" \
        "${PYTORCH_TORCHAUDIO_SPEC}" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}" \
    && python -m pip install -e "${VERL_ROOT}" \
    && python -m pip install -e "${LIBERO_ROOT}" \
    && python -m pip install -e "${OPENVLA_OFT_ROOT}" --no-deps \
    && python -m pip install -r "${OPENVLA_OFT_ROOT}/experiments/robot/libero/libero_requirements.txt"

RUN python -m pip install \
        accelerate==0.25.0 \
        codetiming \
        diffusers==0.30.3 \
        "dill>=0.3.0,<0.4.2" \
        draccus==0.8.0 \
        einops \
        fastapi \
        h5py \
        "huggingface-hub>=0.23,<1.0" \
        hydra-core \
        imageio \
        imageio-ffmpeg \
        filelock \
        "fsspec[http]>=2023.1.0,<=2026.4.0" \
        jinja2 \
        "mpmath==1.3.0" \
        "mujoco==${MUJOCO_VERSION}" \
        json-numpy \
        jsonlines \
        matplotlib \
        networkx \
        ninja \
        omegaconf \
        "numpy<2" \
        "opencv-python<4.12" \
        packaging \
        pandas \
        peft==0.11.1 \
        protobuf==3.20.3 \
        pylatexenc \
        "pydantic>=2.9,<3" \
        PyYAML \
        "requests>=2.31.0" \
        rich \
        "robosuite==1.4.1" \
        safetensors \
        sentencepiece==0.1.99 \
        "sympy>=1.13.1,<1.14" \
        timm==0.9.10 \
        "typing-extensions>=4.8.0" \
        tokenizers==0.19.1 \
        uvicorn \
        wandb \
        tensorflow==2.15.0 \
        tensorflow-datasets==4.9.3 \
        tensorflow-graphics==2021.12.3 \
        tensorflow-metadata==1.15.0 \
    && python -m pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla.git" \
    && python -m pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" \
    && python -m pip install --upgrade \
        "${PYTORCH_TORCH_SPEC}" \
        "${PYTORCH_TORCHVISION_SPEC}" \
        "${PYTORCH_TORCHAUDIO_SPEC}" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}" \
    && python -m pip install --upgrade --force-reinstall --no-deps "mpmath==1.3.0"

RUN python - <<'PY'
import torch
import tokenizers
import transformers
import mujoco
import safetensors

assert torch.__version__ == "2.4.0+cu121", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert tokenizers.__version__ == "0.19.1", tokenizers.__version__
assert transformers.__version__ == "4.40.1", transformers.__version__
assert mujoco.__version__ == "2.3.7", mujoco.__version__
print("Build preflight OK:", torch.__version__, torch.version.cuda, transformers.__version__, safetensors.__version__)
PY

COPY docker/simplevla/entrypoint.sh /opt/simplevla/entrypoint.sh
COPY docker/simplevla/preflight.py /opt/simplevla/preflight.py
RUN chmod +x /opt/simplevla/entrypoint.sh

WORKDIR /workspace/SimpleVLA-RL-BatchSliceFix
ENTRYPOINT ["/opt/simplevla/entrypoint.sh"]
