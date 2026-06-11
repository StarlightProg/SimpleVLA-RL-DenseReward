"""Validation video selection helpers."""

import numpy as np


def select_validation_video_indices(non_tensor_batch, meta_info, batch_size, n_samples, video_max_episodes):
    local_mask_raw = None
    if isinstance(non_tensor_batch, dict):
        local_mask_raw = non_tensor_batch.get("validation_video_selected", None)

    mask_raw = local_mask_raw
    if mask_raw is None:
        mask_raw = meta_info.get("validation_video_mask", None)

    if mask_raw is None:
        return set(range(min(video_max_episodes, batch_size)))

    video_mask = np.asarray(mask_raw, dtype=np.bool_).reshape(-1)
    if video_mask.size * n_samples == batch_size:
        video_mask = np.repeat(video_mask, n_samples)
    elif video_mask.size != batch_size:
        if local_mask_raw is None:
            print(
                "Validation video mask was not shard-local; falling back to local video selection",
                flush=True,
            )
            return set(range(min(video_max_episodes, batch_size)))
        video_mask = np.zeros(batch_size, dtype=np.bool_)

    return {i for i, flag in enumerate(video_mask.tolist()) if flag}
