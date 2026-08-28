import numpy as np


HARD_TASK_SAMPLING_MODES = {"balanced_hard", "clip_hard"}


def normalize_task_sampling_mode(mode) -> str:
    return str(mode or "balanced_hard").strip().lower()


def uses_hard_task_sampler(mode) -> bool:
    return normalize_task_sampling_mode(mode) in HARD_TASK_SAMPLING_MODES


def is_clip_hard_mode(mode) -> bool:
    return normalize_task_sampling_mode(mode) == "clip_hard"


def make_prompt_group_ids(task_suites, task_ids, trial_ids, uids):
    group_ids = []
    for suite, task_id, trial_id, uid in zip(task_suites, task_ids, trial_ids, uids):
        group_ids.append(f"{suite}:task_{int(task_id)}:trial_{int(trial_id)}:{uid}")
    return np.array(group_ids, dtype=object)


def clip_hard_retry_prompt_mask(acc_values, n_samples: int, target_accuracy: float) -> np.ndarray:
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    acc_array = np.asarray(acc_values, dtype=np.float64).reshape(-1)
    if acc_array.size % n_samples != 0:
        raise ValueError(
            f"acc_values length {acc_array.size} must be divisible by n_samples={n_samples}"
        )
    group_acc = acc_array.reshape(-1, n_samples).mean(axis=1)
    return group_acc < float(target_accuracy)
