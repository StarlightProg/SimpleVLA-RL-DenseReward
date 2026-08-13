from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from .dense_reward import DenseRewardManager, RewardWeights
from .libero_state import LiberoState
from .phases import (
    GraspObjectPhase,
    LiftObjectPhase,
    MoveToTargetPhase,
    PlaceOrSuccessPhase,
    ReachObjectPhase,
    Thresholds,
)
from .task_specs import TaskSpec
from .tracker import OnlineSubgoalTracker


def _to_container(config: Any) -> dict:
    if config is None:
        return {}
    if OmegaConf is not None and OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def _get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _vec3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _pose_position(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    pose_p = getattr(value, "p", None)
    if pose_p is not None:
        return _vec3(pose_p)
    return _vec3(value)


def _actor_position(actor: Any) -> np.ndarray | None:
    if actor is None:
        return None
    get_pose = getattr(actor, "get_pose", None)
    if callable(get_pose):
        try:
            return _pose_position(get_pose())
        except Exception:
            return None
    return None


def _arm_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return None


class Robotwin2MoveCanPotStateExtractor:
    def extract(self, env: Any, done: bool = False) -> LiberoState:
        arm = _arm_name(getattr(env, "arm_tag", None))
        gripper_position = self._gripper_position(env, arm)
        gripper_open = self._gripper_open(env, arm)
        object_position = _actor_position(getattr(env, "can", None))
        target_position = _pose_position(getattr(env, "target_pose", None))
        success = bool(done or getattr(env, "eval_success", False))
        object_in_gripper = self._object_in_gripper(
            object_position=object_position,
            gripper_position=gripper_position,
            gripper_open=gripper_open,
        )
        return LiberoState(
            task_name=getattr(env, "task_name", "move_can_pot"),
            instruction=getattr(env, "instruction", None),
            gripper_position=gripper_position,
            gripper_open=gripper_open,
            object_position=object_position,
            target_position=target_position,
            object_in_gripper=object_in_gripper,
            success=success,
        )

    def _gripper_position(self, env: Any, arm: str | None) -> np.ndarray | None:
        robot = getattr(env, "robot", None)
        if robot is None:
            return None
        getter_names = (
            ("get_left_tcp_pose", "get_left_ee_pose")
            if arm == "left"
            else ("get_right_tcp_pose", "get_right_ee_pose")
        )
        for getter_name in getter_names:
            getter = getattr(robot, getter_name, None)
            if not callable(getter):
                continue
            try:
                position = _vec3(getter())
            except Exception:
                position = None
            if position is not None:
                return position
        return None

    def _gripper_open(self, env: Any, arm: str | None) -> float | None:
        robot = getattr(env, "robot", None)
        if robot is None:
            return None
        getter_name = "get_left_gripper_val" if arm == "left" else "get_right_gripper_val"
        getter = getattr(robot, getter_name, None)
        if not callable(getter):
            return None
        try:
            return float(np.clip(getter(), 0.0, 1.0))
        except Exception:
            return None

    def _object_in_gripper(
        self,
        object_position: np.ndarray | None,
        gripper_position: np.ndarray | None,
        gripper_open: float | None,
    ) -> bool | None:
        if object_position is None or gripper_position is None or gripper_open is None:
            return None
        return bool(gripper_open < 0.5 and np.linalg.norm(object_position - gripper_position) <= 0.12)


def _move_can_pot_spec(state: LiberoState, thresholds: Thresholds) -> TaskSpec | None:
    if state.object_position is None or state.target_position is None:
        return None
    initial_z = float(state.object_position[2])
    return TaskSpec(
        name="robotwin2_move_can_pot_grasp_place",
        phases=[
            ReachObjectPhase(thresholds, initial_z),
            GraspObjectPhase(thresholds, initial_z),
            LiftObjectPhase(thresholds, initial_z),
            MoveToTargetPhase(thresholds, initial_z),
            PlaceOrSuccessPhase(thresholds, initial_z),
        ],
    )


class Robotwin2SubgoalRewardEngine:
    def __init__(self, config: Any = None):
        cfg = _to_container(config)
        self.enabled = bool(_get(cfg, "enabled", False))
        self.unsupported_task_behavior = str(_get(cfg, "unsupported_task_behavior", "terminal_only"))
        self.use_best_progress = bool(_get(cfg, "use_best_progress", True))
        self.clip_dense_reward = _get(cfg, "clip_dense_reward", 0.05)

        thresholds_cfg = _to_container(cfg.get("thresholds"))
        self.thresholds = Thresholds(
            reach_distance=float(_get(thresholds_cfg, "reach_distance", 0.1)),
            target_distance=float(_get(thresholds_cfg, "target_distance", 0.06)),
            lift_height=float(_get(thresholds_cfg, "lift_height", 0.08)),
        )

        weights_cfg = _to_container(cfg.get("weights"))
        self.reward_manager = DenseRewardManager(
            RewardWeights(
                subgoal_progress=float(_get(weights_cfg, "subgoal_progress", 0.2)),
                phase_transition=float(_get(weights_cfg, "phase_transition", 0.05)),
                terminal_success=float(_get(weights_cfg, "terminal_success", 1.0)),
                smoothness=float(_get(weights_cfg, "smoothness", 0.0)),
            ),
            clip_dense_reward=self.clip_dense_reward,
        )
        self.extractor = Robotwin2MoveCanPotStateExtractor()
        self.trackers: dict[int, OnlineSubgoalTracker] = {}

    def reset(self, env_index: int | None = None):
        if env_index is None:
            self.trackers.clear()
        else:
            self.trackers.pop(int(env_index), None)

    def step(
        self,
        env_index: int,
        env: Any,
        action: Any,
        env_reward: float,
        done: bool,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        if not self.enabled:
            return self._terminal_only(done=done, env_reward=env_reward, phase_name="disabled")

        task_metadata = task_metadata or {}
        task_name = str(task_metadata.get("task_name") or getattr(env, "task_name", ""))
        if task_name not in {"move_can_pot", "robotwin2_move_can_pot"}:
            if self.unsupported_task_behavior == "error":
                raise ValueError(f"Unsupported RoboTwin2 subgoal task: {task_name or 'unknown'}")
            return self._terminal_only(done=done, env_reward=env_reward, phase_name="terminal_only")

        state = self.extractor.extract(env=env, done=done)
        task_spec = _move_can_pot_spec(state, self.thresholds)
        if task_spec is None:
            if self.unsupported_task_behavior == "error":
                raise ValueError("RoboTwin2 move_can_pot state is missing object or target pose")
            return self._terminal_only(done=state.success, env_reward=env_reward, phase_name="terminal_only")

        tracker = self._tracker_for(env_index, task_spec)
        step_info = tracker.update(state, action=action)
        reward_parts = self.reward_manager.compute(
            positive_delta=step_info.positive_delta,
            phase_completed=step_info.phase_completed,
            terminal_success=state.success,
            action_delta_l2=step_info.action_delta_l2,
            env_reward=env_reward,
        )

        subgoal_info = step_info.as_numeric_dict()
        subgoal_info["subgoal_has_object"] = float(state.object_position is not None)
        subgoal_info["subgoal_has_target"] = float(state.target_position is not None)
        subgoal_info["subgoal_has_gripper"] = float(state.gripper_position is not None)
        subgoal_info["phase_name"] = step_info.phase_name
        if done:
            self.reset(env_index)
        return subgoal_info, reward_parts.as_dict()

    def _tracker_for(self, env_index: int, task_spec: TaskSpec) -> OnlineSubgoalTracker:
        env_index = int(env_index)
        tracker = self.trackers.get(env_index)
        if tracker is None:
            tracker = OnlineSubgoalTracker(
                task_spec=task_spec,
                use_best_progress=self.use_best_progress,
                auto_complete_on_success=False,
            )
            self.trackers[env_index] = tracker
        return tracker

    def _terminal_only(
        self,
        done: bool,
        env_reward: float,
        phase_name: str,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        return (
            {
                "subgoal_supported": 0.0,
                "subgoal_phase_id": -1.0,
                "subgoal_has_object": 0.0,
                "subgoal_has_target": 0.0,
                "subgoal_has_gripper": 0.0,
                "subgoal_progress": 0.0,
                "subgoal_best_progress": 0.0,
                "subgoal_positive_delta": 0.0,
                "subgoal_phase_completed": 0.0,
                "success": float(done),
                "action_delta_l2": 0.0,
                "phase_name": phase_name,
            },
            {
                "reward_env": float(env_reward),
                "reward_subgoal": 0.0,
                "reward_phase": 0.0,
                "reward_terminal": 0.0,
                "reward_smoothness": 0.0,
                "reward_total": float(env_reward),
            },
        )
