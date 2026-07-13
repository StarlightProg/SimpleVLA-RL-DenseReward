"""Small compatibility layer for optional LeRobot SmolVLA support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    _wrap_frozen_image_embed_no_grad(policy)
    return policy


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
            padding=True,
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
    batch[OBS_STATE_KEY] = torch.stack(
        [torch.as_tensor(item["state"], dtype=torch.float32, device=device) for item in inputs],
        dim=0,
    )

    for feature_idx, feature_key in enumerate(image_keys):
        source_key = "full_image" if feature_idx == 0 else "wrist_image"
        if any(source_key not in item for item in inputs):
            continue
        batch[feature_key] = torch.stack(
            [_image_to_tensor(item[source_key], device=device) for item in inputs],
            dim=0,
        )
    return batch
