from __future__ import annotations

from dataclasses import dataclass

from .libero_state import LiberoState
from .phases import (
    GraspObjectPhase,
    LiftObjectPhase,
    MoveToTargetPhase,
    Phase,
    PlaceOrSuccessPhase,
    ReachObjectPhase,
    Thresholds,
)


# The current LIBERO dense-reward task spec is deliberately explicit.  Keep
# this ordered tuple close to the spec so validation can publish stable metric
# names without inferring them from an individual trajectory.
LIBERO_PICK_PLACE_PHASE_NAMES = (
    "reach_object",
    "grasp_object",
    "lift_object",
    "move_to_target",
    "place_or_success",
)


@dataclass
class TaskSpec:
    name: str
    phases: list[Phase]
    supported: bool = True


def infer_task_spec(state: LiberoState, thresholds: Thresholds) -> TaskSpec | None:
    text = " ".join(part for part in [state.task_name, state.instruction] if part).lower()
    looks_like_pick_place = any(word in text for word in ("put", "place", "pick")) and any(
        word in text
        for word in ("on", "in", "into", "onto", "to", "above", "right", "left", "below", "next")
    )
    if not looks_like_pick_place:
        return None
    if state.object_position is None or state.target_position is None:
        return None

    initial_z = float(state.object_position[2])
    phases = [
        ReachObjectPhase(thresholds, initial_z),
        GraspObjectPhase(thresholds, initial_z),
        LiftObjectPhase(thresholds, initial_z),
        MoveToTargetPhase(thresholds, initial_z),
        PlaceOrSuccessPhase(thresholds, initial_z),
    ]
    return TaskSpec(name="pick_place", phases=phases)
