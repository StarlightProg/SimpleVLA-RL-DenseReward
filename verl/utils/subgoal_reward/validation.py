from __future__ import annotations

from collections.abc import Mapping, Sequence

from .task_specs import LIBERO_PICK_PLACE_PHASE_NAMES


def phase_success_batch_key(phase_name: str) -> str:
    return f"subgoal_phase_success_{phase_name}"


def tally_phase_success(
    supported_values: Sequence[object],
    phase_values: Mapping[str, Sequence[object]],
    take_n: int,
) -> tuple[int, dict[str, int]]:
    """Count per-phase episode success for the validation rows being retained."""
    take_n = int(take_n)
    if take_n < 0 or len(supported_values) < take_n:
        raise ValueError("supported_values does not cover the requested validation rows")

    counts = {phase_name: 0 for phase_name in LIBERO_PICK_PLACE_PHASE_NAMES}
    for phase_name in LIBERO_PICK_PLACE_PHASE_NAMES:
        if phase_name not in phase_values or len(phase_values[phase_name]) < take_n:
            raise ValueError(f"Missing phase-success values for {phase_name}")

    supported_rollouts = 0
    for index in range(take_n):
        if not bool(supported_values[index]):
            continue
        supported_rollouts += 1
        for phase_name in LIBERO_PICK_PLACE_PHASE_NAMES:
            if bool(phase_values[phase_name][index]):
                counts[phase_name] += 1
    return supported_rollouts, counts
