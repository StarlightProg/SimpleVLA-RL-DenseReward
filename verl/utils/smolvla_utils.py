"""Small compatibility layer for optional LeRobot SmolVLA support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch


SMOLVLA_INSTALL_HINT = (
    "SmolVLA support requires LeRobot with the SmolVLA extra. "
    'Install a compatible LeRobot checkout with: pip install -e ".[smolvla]"'
)

ACTION_KEY = "action"
OBS_STATE_KEY = "observation.state"
OBS_LANGUAGE_TOKENS_KEY = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK_KEY = "observation.language.attention_mask"
PREPROCESSOR_NORM_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
POSTPROCESSOR_NORM_FILE = "policy_postprocessor_step_1_unnormalizer_processor.safetensors"


@dataclass
class MinimalTokenizer:
    """Tiny tokenizer shim for trainer metadata on non-token policies."""

    pad_token_id: int = 0
    eos_token_id: int = 0
    bos_token_id: int = 0
    pad_token: str = "<pad>"
    eos_token: str = "<eos>"

    def save_pretrained(self, path: str) -> None:
        return None


def require_smolvla_policy():
    try:
        from lerobot.policies.smolvla import SmolVLAPolicy
    except ImportError:
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError(SMOLVLA_INSTALL_HINT) from exc
    return SmolVLAPolicy


def load_smolvla_policy(model_path: str, **kwargs: Any):
    policy_cls = require_smolvla_policy()
    policy = policy_cls.from_pretrained(model_path, **kwargs)
    policy._verl_smolvla_norm_stats = load_smolvla_processor_stats(model_path)
    _wrap_frozen_image_embed_no_grad(policy)
    return policy


def _resolve_processor_state_file(model_path: str, filename: str) -> str | None:
    local_path = Path(model_path) / filename
    if local_path.exists():
        return str(local_path)

    if Path(model_path).exists():
        return None

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_path, filename=filename)
    except Exception:
        return None


def load_smolvla_processor_stats(model_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """Load LeRobot SmolVLA normalizer/unnormalizer stats when present."""
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        warnings.warn(f"Could not import safetensors to load SmolVLA processor stats: {exc}")
        return {}

    stats = {}
    for name, filename in (
        ("preprocessor", PREPROCESSOR_NORM_FILE),
        ("postprocessor", POSTPROCESSOR_NORM_FILE),
    ):
        state_path = _resolve_processor_state_file(str(model_path), filename)
        if state_path is None:
            warnings.warn(
                f"SmolVLA checkpoint {model_path!r} is missing {filename}; "
                "state normalization/action unnormalization will be skipped."
            )
            continue
        stats[name] = load_file(state_path, device="cpu")
    return stats


def _wrap_frozen_image_embed_no_grad(policy: Any) -> None:
    """Avoid storing gradients through SmolVLA's frozen vision encoder."""
    model = getattr(policy, "model", None)
    vlm = getattr(model, "vlm_with_expert", None)
    if vlm is None or not bool(getattr(vlm, "freeze_vision_encoder", False)):
        return
    if getattr(vlm, "_verl_no_grad_image_embed", False):
        return

    embed_image = vlm.embed_image

    def embed_image_no_grad(image: torch.Tensor):
        with torch.no_grad():
            return embed_image(image)

    vlm.embed_image = embed_image_no_grad
    vlm._verl_no_grad_image_embed = True


def unwrap_smolvla_policy(policy: Any):
    seen = set()
    current = policy
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_fsdp_wrapped_module"):
            current = current._fsdp_wrapped_module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        base_model = getattr(current, "base_model", None)
        if base_model is not None:
            current = base_model
            continue
        wrapped_model = getattr(current, "model", None)
        if wrapped_model is not None and hasattr(wrapped_model, "config") and not hasattr(current, "config"):
            current = wrapped_model
            continue
        break
    return current


def get_smolvla_tokenizer(policy: Any):
    policy = unwrap_smolvla_policy(policy)
    model = getattr(policy, "model", None)
    vlm = getattr(model, "vlm_with_expert", None)
    processor = getattr(vlm, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return MinimalTokenizer()
    return tokenizer


def smolvla_policy_config(policy: Any):
    policy = unwrap_smolvla_policy(policy)
    config = getattr(policy, "config", None)
    if config is None:
        raise ValueError("Loaded SmolVLA policy does not expose a `config` attribute.")
    return config


def smolvla_image_feature_keys(policy: Any) -> list[str]:
    config = smolvla_policy_config(policy)
    image_features = getattr(config, "image_features", None)
    if image_features is None:
        raise ValueError("SmolVLA policy config does not expose `image_features`.")
    if isinstance(image_features, dict):
        return list(image_features.keys())
    return list(image_features)


def smolvla_action_chunk_len(policy: Any) -> int:
    config = smolvla_policy_config(policy)
    return int(getattr(config, "chunk_size", getattr(config, "n_action_steps", 1)))


def smolvla_image_resize_size(policy: Any) -> int | tuple[int, int]:
    """Return the visual input size expected by a loaded SmolVLA checkpoint."""
    config = smolvla_policy_config(policy)
    input_features = getattr(config, "input_features", None)
    if not input_features:
        return 224

    features = input_features.values() if isinstance(input_features, dict) else input_features
    for feature in features:
        feature_type = getattr(feature, "type", None)
        if feature_type is None and isinstance(feature, dict):
            feature_type = feature.get("type")
        if str(feature_type).split(".")[-1].upper() != "VISUAL":
            continue

        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is None or len(shape) < 3:
            continue
        height, width = int(shape[-2]), int(shape[-1])
        return height if height == width else (height, width)

    return 224


def _norm_stats(policy: Any, processor: str, key: str):
    policy = unwrap_smolvla_policy(policy)
    stats = getattr(policy, "_verl_smolvla_norm_stats", {}) or {}
    processor_stats = stats.get(processor, {})
    mean = processor_stats.get(f"{key}.mean")
    std = processor_stats.get(f"{key}.std")
    if mean is None or std is None:
        return None, None
    return mean, std


def normalize_smolvla_state(policy: Any, state: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_stats(policy, "preprocessor", OBS_STATE_KEY)
    if mean is None or std is None:
        return state
    mean = mean.to(device=state.device, dtype=state.dtype)
    std = std.to(device=state.device, dtype=state.dtype).clamp_min(1e-8)
    return (state - mean) / std


def unnormalize_smolvla_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_stats(policy, "postprocessor", ACTION_KEY)
    if mean is None or std is None:
        return actions
    mean = mean.to(device=actions.device, dtype=actions.dtype)
    std = std.to(device=actions.device, dtype=actions.dtype).clamp_min(1e-8)
    return actions * std + mean


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(image)).to(device=device, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {tuple(tensor.shape)}")
    if tensor.shape[-1] != 3:
        raise ValueError(f"Expected RGB image in last dimension, got {tuple(tensor.shape)}")
    return tensor.permute(2, 0, 1).contiguous() / 255.0


def tokenize_smolvla_tasks(policy: Any, task_descriptions: list[str], device: torch.device, max_length: int):
    tokenizer = get_smolvla_tokenizer(policy)
    prompts = [str(task).strip() for task in task_descriptions]
    if callable(tokenizer):
        tokens = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            OBS_LANGUAGE_TOKENS_KEY: tokens["input_ids"].to(device=device),
            OBS_LANGUAGE_ATTENTION_MASK_KEY: tokens["attention_mask"].to(device=device, dtype=torch.bool),
        }

    # Last-resort fallback for mocked policies in unit smoke tests.
    batch_size = len(prompts)
    return {
        OBS_LANGUAGE_TOKENS_KEY: torch.zeros(batch_size, 1, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK_KEY: torch.ones(batch_size, 1, dtype=torch.bool, device=device),
    }


def build_smolvla_batch(
    policy: Any,
    inputs: list[dict[str, Any]],
    task_descriptions: list[str],
    *,
    device: torch.device,
    max_prompt_length: int,
) -> dict[str, torch.Tensor]:
    image_keys = smolvla_image_feature_keys(policy)
    batch = tokenize_smolvla_tasks(policy, task_descriptions, device, max_prompt_length)
    state = torch.stack(
        [torch.as_tensor(item["state"], dtype=torch.float32, device=device) for item in inputs],
        dim=0,
    )
    batch[OBS_STATE_KEY] = normalize_smolvla_state(policy, state)

    for feature_idx, feature_key in enumerate(image_keys):
        source_key = "full_image" if feature_idx == 0 else "wrist_image"
        if any(source_key not in item for item in inputs):
            continue
        batch[feature_key] = torch.stack(
            [_image_to_tensor(item[source_key], device=device) for item in inputs],
            dim=0,
        )
    return batch
