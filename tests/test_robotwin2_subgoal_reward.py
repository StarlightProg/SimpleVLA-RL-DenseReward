import unittest
from pathlib import Path

import numpy as np

from verl.utils.subgoal_reward.robotwin2 import Robotwin2SubgoalRewardEngine


class FakePose:
    def __init__(self, p):
        self.p = np.array(p, dtype=np.float32)


class FakeActor:
    def __init__(self, p):
        self.pose = FakePose(p)

    def get_pose(self):
        return self.pose

    def set_pos(self, p):
        self.pose = FakePose(p)


class FakeRobot:
    def __init__(self):
        self.left_ee = np.array([-1.0, 0.0, 0.74], dtype=np.float32)
        self.right_ee = np.array([1.0, 0.0, 0.74], dtype=np.float32)
        self.left_gripper = 1.0
        self.right_gripper = 1.0

    def get_left_ee_pose(self):
        return self.left_ee

    def get_right_ee_pose(self):
        return self.right_ee

    def get_left_gripper_val(self):
        return self.left_gripper

    def get_right_gripper_val(self):
        return self.right_gripper


class FakeMoveCanPotEnv:
    def __init__(self):
        self.task_name = "move_can_pot"
        self.arm_tag = "right"
        self.robot = FakeRobot()
        self.can = FakeActor([0.0, 0.0, 0.74])
        self.target_pose = FakePose([0.20, 0.0, 0.74])
        self.eval_success = False
        self.instruction = "move the can near the pot"
        self.step_lim = 200

    def set_state(self, can, gripper, gripper_open, success=False):
        self.can.set_pos(can)
        self.robot.right_ee = np.array(gripper, dtype=np.float32)
        self.robot.right_gripper = float(gripper_open)
        self.eval_success = bool(success)

    def take_action(self, action):
        self.eval_success = bool(self.eval_success)

    def get_obs(self):
        return {"observation": {"head_camera": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}}}


def engine_config(subgoal_progress=0.0, phase_transition=1.0, terminal_success=0.0):
    return {
        "enabled": True,
        "clip_dense_reward": 10.0,
        "weights": {
            "subgoal_progress": subgoal_progress,
            "phase_transition": phase_transition,
            "terminal_success": terminal_success,
            "smoothness": 0.0,
        },
    }


def deterministic_grasp_move_policy():
    return [
        ([0.00, 0.00, 0.74], [0.20, 0.00, 0.74], 1.0, False),
        ([0.00, 0.00, 0.74], [0.00, 0.00, 0.74], 1.0, False),
        ([0.00, 0.00, 0.74], [0.00, 0.00, 0.74], 0.0, False),
        ([0.00, 0.00, 0.83], [0.00, 0.00, 0.83], 0.0, False),
        ([0.20, 0.00, 0.83], [0.20, 0.00, 0.83], 0.0, False),
        ([0.20, 0.00, 0.74], [0.20, 0.00, 0.74], 1.0, True),
    ]


def deterministic_push_policy():
    return [
        ([0.00, 0.00, 0.74], [0.20, 0.00, 0.74], 1.0, False),
        ([0.00, 0.00, 0.74], [0.00, 0.00, 0.74], 1.0, False),
        ([0.08, 0.00, 0.74], [0.00, 0.00, 0.74], 1.0, False),
        ([0.16, 0.00, 0.74], [0.00, 0.00, 0.74], 1.0, False),
        ([0.20, 0.00, 0.74], [0.00, 0.00, 0.74], 1.0, True),
    ]


def rollout_policy(policy_states, config, terminal_env_reward=True):
    env = FakeMoveCanPotEnv()
    engine = Robotwin2SubgoalRewardEngine(config)
    action = np.zeros((25, 14), dtype=np.float32)
    totals = {
        "terminal": 0.0,
        "phase": 0.0,
        "dense_total": 0.0,
        "subgoal_progress": 0.0,
        "phase_completed": 0.0,
    }
    phase_names = []
    for can, gripper, gripper_open, success in policy_states:
        env.set_state(can, gripper, gripper_open, success)
        env_reward = float(success) if terminal_env_reward else 0.0
        info, rewards = engine.step(
            env_index=0,
            env=env,
            action=action,
            env_reward=env_reward,
            done=success,
            task_metadata={"task_name": "move_can_pot"},
        )
        totals["terminal"] += env_reward
        totals["phase"] += rewards["reward_phase"]
        totals["dense_total"] += rewards["reward_total"]
        totals["subgoal_progress"] += rewards["reward_subgoal"]
        totals["phase_completed"] += info["subgoal_phase_completed"]
        phase_names.append(info["phase_name"])
    totals["phase_names"] = phase_names
    return totals


class Robotwin2SubgoalRewardTest(unittest.TestCase):
    def test_grasp_move_place_completes_ordered_phases(self):
        env = FakeMoveCanPotEnv()
        engine = Robotwin2SubgoalRewardEngine(engine_config())
        action = np.zeros((25, 14), dtype=np.float32)
        states = deterministic_grasp_move_policy()[1:]

        completed = 0.0
        phase_reward = 0.0
        for can, gripper, gripper_open, success in states:
            env.set_state(can, gripper, gripper_open, success)
            info, rewards = engine.step(
                env_index=0,
                env=env,
                action=action,
                env_reward=float(success),
                done=success,
                task_metadata={"task_name": "move_can_pot"},
            )
            completed += info["subgoal_phase_completed"]
            phase_reward += rewards["reward_phase"]

        self.assertEqual(completed, 5.0)
        self.assertEqual(phase_reward, 5.0)

    def test_deterministic_policies_count_terminal_phase_and_dense_rewards(self):
        terminal_config = engine_config(subgoal_progress=0.0, phase_transition=0.0, terminal_success=0.0)
        phase_config = engine_config(subgoal_progress=0.0, phase_transition=0.3, terminal_success=0.0)
        dense_config = engine_config(subgoal_progress=0.2, phase_transition=0.3, terminal_success=0.0)

        push_terminal = rollout_policy(deterministic_push_policy(), terminal_config)
        grasp_terminal = rollout_policy(deterministic_grasp_move_policy(), terminal_config)
        self.assertEqual(push_terminal["terminal"], 1.0)
        self.assertEqual(grasp_terminal["terminal"], 1.0)

        push_phase = rollout_policy(deterministic_push_policy(), phase_config)
        grasp_phase = rollout_policy(deterministic_grasp_move_policy(), phase_config)
        self.assertEqual(push_phase["phase_completed"], 1.0)
        self.assertEqual(push_phase["phase"], 0.3)
        self.assertEqual(grasp_phase["phase_completed"], 5.0)
        self.assertEqual(grasp_phase["phase"], 1.5)

        push_dense = rollout_policy(deterministic_push_policy(), dense_config)
        grasp_dense = rollout_policy(deterministic_grasp_move_policy(), dense_config)
        self.assertEqual(push_dense["phase_completed"], 1.0)
        self.assertEqual(push_dense["phase"], 0.3)
        self.assertGreater(push_dense["subgoal_progress"], 0.0)
        self.assertLess(push_dense["dense_total"], grasp_dense["dense_total"])
        self.assertEqual(grasp_dense["phase_completed"], 5.0)
        self.assertEqual(grasp_dense["phase"], 1.5)
        self.assertGreater(grasp_dense["subgoal_progress"], push_dense["subgoal_progress"])

    def test_push_only_success_does_not_get_grasp_or_lift_phase_credit(self):
        env = FakeMoveCanPotEnv()
        engine = Robotwin2SubgoalRewardEngine(engine_config(terminal_success=1.0))
        env.set_state(can=[0.20, 0.0, 0.74], gripper=[1.0, 0.0, 0.74], gripper_open=1.0, success=True)

        info, rewards = engine.step(
            env_index=0,
            env=env,
            action=np.zeros((25, 14), dtype=np.float32),
            env_reward=1.0,
            done=True,
            task_metadata={"task_name": "move_can_pot"},
        )

        self.assertEqual(info["subgoal_phase_completed"], 0.0)
        self.assertEqual(rewards["reward_phase"], 0.0)
        self.assertEqual(rewards["reward_terminal"], 1.0)
        self.assertLessEqual(info["subgoal_phase_id"], 0.0)

    def test_unsupported_robotwin2_task_falls_back_to_terminal_only(self):
        env = FakeMoveCanPotEnv()
        engine = Robotwin2SubgoalRewardEngine(engine_config())
        info, rewards = engine.step(
            env_index=0,
            env=env,
            action=np.zeros((25, 14), dtype=np.float32),
            env_reward=1.0,
            done=True,
            task_metadata={"task_name": "place_a2b_left"},
        )

        self.assertEqual(info["subgoal_supported"], 0.0)
        self.assertEqual(info["phase_name"], "terminal_only")
        self.assertEqual(rewards["reward_total"], 1.0)

    def test_lora_pushcut_launcher_is_lora_only_and_has_reward_modes(self):
        script = Path("examples/run_openvla_oft_rl_twin2_lora_pushcut.sh").read_text(encoding="utf-8")

        self.assertIn('LORA_RANK="${LORA_RANK:-16}"', script)
        self.assertIn("actor_rollout_ref.model.lora_rank=${LORA_RANK}", script)
        self.assertIn("actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}", script)
        self.assertIn("actor_rollout_ref.model.target_modules=llm-projector", script)
        self.assertIn("  terminal)", script)
        self.assertIn("  phase)", script)
        self.assertIn("  dense)", script)
        self.assertIn("reward.subgoal.weights.phase_transition=0.3", script)
        self.assertIn("reward.subgoal.weights.subgoal_progress=0.2", script)

    def test_lora_pushcut_smoke_launcher_has_tiny_steps_and_adapter_check(self):
        script = Path("examples/run_openvla_oft_rl_twin2_lora_pushcut_smoke.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-1}"', script)
        self.assertIn('SAVE_FREQ="${SAVE_FREQ:-1}"', script)
        self.assertIn('TEST_FREQ="${TEST_FREQ:-1}"', script)
        self.assertIn('VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"', script)
        self.assertIn('REWARD_MODE="${REWARD_MODE:-dense}"', script)
        self.assertIn("lora_adapter/adapter_config.json", script)
        self.assertIn("adapter_model.safetensors", script)
        self.assertIn("run_openvla_oft_rl_twin2_lora_pushcut.sh", script)

    def test_robotwin_wrapper_step_returns_subgoal_metrics_when_enabled(self):
        try:
            from verl.workers.rollout.rob_rollout import RobotwinEnvWrapper
        except Exception as exc:
            self.skipTest(f"rollout dependencies unavailable: {exc}")

        env = FakeMoveCanPotEnv()
        env.set_state(can=[0.0, 0.0, 0.74], gripper=[0.0, 0.0, 0.74], gripper_open=1.0)
        wrapper = RobotwinEnvWrapper(
            task_name="move_can_pot",
            trial_id=100000,
            trial_seed=100000,
            config={},
            version="2.0",
        )
        wrapper.env = env
        wrapper.subgoal_engine = Robotwin2SubgoalRewardEngine(engine_config())
        wrapper.task_metadata = {"task_name": "move_can_pot"}

        _, _, metrics = wrapper.step(np.zeros((25, 14), dtype=np.float32))

        self.assertIsNotNone(metrics)
        self.assertIn("reward_total", metrics)
        self.assertIn("reward_phase", metrics)
        self.assertEqual(metrics["subgoal_supported"], 1.0)


if __name__ == "__main__":
    unittest.main()
