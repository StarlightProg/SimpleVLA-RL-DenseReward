import contextlib
import importlib
import io
import json
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def write_libero_config() -> None:
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


def check_imports() -> None:
    checks = [
        ("torch", "torch"),
        ("sympy", "sympy"),
        ("mpmath", "mpmath"),
        ("dill", "dill"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
        ("hydra-core", "hydra"),
        ("omegaconf", "omegaconf"),
        ("codetiming", "codetiming"),
        ("PyYAML", "yaml"),
        ("pandas", "pandas"),
        ("pylatexenc", "pylatexenc"),
        ("numpy", "numpy"),
        ("mujoco", "mujoco"),
        ("Pillow", "PIL"),
        ("sentencepiece", "sentencepiece"),
        ("timm", "timm"),
        ("tensordict", "tensordict"),
        ("tokenizers", "tokenizers"),
        ("ray", "ray"),
        ("peft", "peft"),
        ("robosuite", "robosuite"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("wandb", "wandb"),
        ("libero", "libero.libero"),
        ("SimpleVLA PPO entry", "verl.trainer.main_ppo"),
        ("SimpleVLA rollout", "verl.workers.rollout.rob_rollout"),
    ]
    missing = []
    for label, module in checks:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{label} ({module}): {exc}")
    if missing:
        print("Dependency preflight failed. Missing or broken imports:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)

    import mujoco
    import robosuite
    import tokenizers
    import torch
    import transformers

    if torch.__version__ != "2.4.0+cu121" or torch.version.cuda != "12.1":
        fail(
            "Torch CUDA build mismatch: "
            f"torch=={torch.__version__}, torch.version.cuda={torch.version.cuda}; "
            "expected torch==2.4.0+cu121 and CUDA 12.1."
        )
    if transformers.__version__ != "4.40.1" or tokenizers.__version__ != "0.19.1":
        fail(
            "OpenVLA-OFT version preflight failed: "
            f"transformers=={transformers.__version__}, tokenizers=={tokenizers.__version__}; "
            "expected transformers==4.40.1 and tokenizers==0.19.1."
        )
    requested_mujoco = os.environ.get("MUJOCO_VERSION", "2.3.7")
    allow_mujoco_3 = os.environ.get("ALLOW_MUJOCO_3", "0") == "1"
    if mujoco.__version__ != requested_mujoco:
        fail(f"MuJoCo version mismatch: got mujoco=={mujoco.__version__}; expected mujoco=={requested_mujoco}.")
    if not mujoco.__version__.startswith("2.") and not (allow_mujoco_3 and mujoco.__version__.startswith("3.")):
        fail(
            f"LIBERO simulation preflight failed: mujoco=={mujoco.__version__}. "
            "Use ALLOW_MUJOCO_3=1 when intentionally testing MuJoCo 3.x."
        )
    print(
        "Dependency preflight OK:",
        f"torch=={torch.__version__}",
        f"torch.cuda=={torch.version.cuda}",
        f"transformers=={transformers.__version__}",
        f"tokenizers=={tokenizers.__version__}",
        f"mujoco=={mujoco.__version__}",
        f"robosuite=={getattr(robosuite, '__version__', 'unknown')}",
    )


def check_gpus() -> None:
    import torch

    expected = int(os.environ["NUM_GPUS"])
    count = torch.cuda.device_count()
    print("Python:", sys.version)
    print("Allocated CUDA devices:", count)
    if count != expected:
        fail(f"Expected {expected} GPUs, but PyTorch sees {count}. Check docker --gpus/NVIDIA passthrough.")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 1024**3
        print(f"GPU {index}: {props.name}, {gib:.1f} GiB")
        if gib < 40:
            fail(
                f"GPU {index} has only {gib:.1f} GiB. "
                f"This configuration expects {expected} L40/A100/A800-class GPUs with >=40 GiB."
            )


def checkpoint_is_complete() -> None:
    root = Path(os.environ["SFT_MODEL_PATH"])
    required = ["config.json", "dataset_statistics.json", "model.safetensors.index.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        fail("Missing checkpoint files under SFT_MODEL_PATH: " + ", ".join(missing))

    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        fail(f"Cannot read model.safetensors.index.json: {exc}")

    weights = sorted(set(index.get("weight_map", {}).values()))
    if not weights:
        fail("No weight files listed in model.safetensors.index.json")

    missing_weights = [
        name for name in weights
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    if missing_weights:
        fail("Missing or empty checkpoint weight files: " + ", ".join(missing_weights))

    try:
        from safetensors import safe_open

        for name in weights:
            with safe_open(str(root / name), framework="pt"):
                pass
    except Exception as exc:  # noqa: BLE001
        fail(f"Checkpoint safetensors validation failed: {exc}")
    print("Checkpoint preflight OK:", root)


def write_runtime_align() -> None:
    repo_root = Path(os.environ["REPO_ROOT"])
    source = Path(os.environ.get("ALIGN_PATH", repo_root / "align.json"))
    target = Path(os.environ["JOB_TMPDIR"]) / "align.docker.json"
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
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "TOKENIZERS_PARALLELISM": "true",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3"),
            "ROBOT_PLATFORM": "LIBERO",
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl"),
            "MUJOCO_VERSION": os.environ.get("MUJOCO_VERSION", "2.3.7"),
            "ALLOW_MUJOCO_3": os.environ.get("ALLOW_MUJOCO_3", "0"),
            "LIBERO_CONFIG_PATH": os.environ["LIBERO_CONFIG_PATH"],
            "LIBERO_DATASET_DIR": os.environ["LIBERO_DATASET_DIR"],
            "VERL_FSDP_SUMMON_OFFLOAD_CPU": "0",
            "HF_HOME": os.environ["HF_HOME"],
            "WANDB_DIR": os.environ["WANDB_DIR"],
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
        }
    )
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    target.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["ALIGN_PATH"] = str(target)
    print("Runtime Ray env written to:", target)


def libero_smoke_test() -> None:
    if os.environ.get("LIBERO_ENV_SMOKE_TEST", "1") == "0":
        print("LIBERO_ENV_SMOKE_TEST=0: skipping LIBERO environment smoke test.")
        return

    from libero.libero import benchmark
    from verl.utils.libero_utils import get_libero_dummy_action, get_libero_env

    print(
        "LIBERO smoke env:",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '')}",
        f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '')}",
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', '')}",
    )

    env = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark.get_benchmark_dict()[os.environ["DATASET_NAME"]]()
        task = task_suite.get_task(0)
        initial_states = task_suite.get_task_init_states(0)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(2):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        print(f"LIBERO env smoke test OK: {task_description}")
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()


def main() -> None:
    write_libero_config()
    check_imports()
    check_gpus()
    checkpoint_is_complete()
    write_runtime_align()
    libero_smoke_test()


if __name__ == "__main__":
    main()
