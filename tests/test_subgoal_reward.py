import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.main_ppo import RobRewardManager
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.subgoal_reward.engine import LiberoSubgoalRewardEngine
from verl.utils.subgoal_reward.libero_state import LiberoState
from verl.utils.subgoal_reward.phases import GraspObjectPhase, ReachObjectPhase, Thresholds
from verl.utils.subgoal_reward.task_specs import TaskSpec
from verl.utils.subgoal_reward.tracker import OnlineSubgoalTracker


def state_at(distance, success=False):
    return LiberoState(
        task_name="put_object_on_target",
        instruction="put the cube on the plate",
        gripper_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        object_position=np.array([distance, 0.0, 0.0], dtype=np.float32),
        target_position=np.array([0.0, 0.2, 0.0], dtype=np.float32),
        gripper_open=1.0,
        success=success,
    )


class SubgoalRewardTest(unittest.TestCase):
    def _cfg(self, **kwargs):
        class Cfg(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)

        return Cfg(**kwargs)

    def test_tracker_reset(self):
        spec = TaskSpec("test", [ReachObjectPhase(Thresholds(reach_distance=0.05))])
        tracker = OnlineSubgoalTracker(spec)
        tracker.update(state_at(0.10))
        self.assertGreater(tracker.best_progress_so_far, 0.0)
        tracker.reset()
        self.assertEqual(tracker.phase_id, 0)
        self.assertEqual(tracker.best_progress_so_far, 0.0)

    def test_best_progress_prevents_reward_farming(self):
        spec = TaskSpec("test", [ReachObjectPhase(Thresholds(reach_distance=0.05))])
        tracker = OnlineSubgoalTracker(spec, use_best_progress=True)
        deltas = [
            tracker.update(state_at(0.15)).positive_delta,
            tracker.update(state_at(0.10)).positive_delta,
            tracker.update(state_at(0.15)).positive_delta,
            tracker.update(state_at(0.08)).positive_delta,
        ]
        self.assertGreater(deltas[1], 0.0)
        self.assertEqual(deltas[2], 0.0)
        self.assertGreater(deltas[3], 0.0)

    def test_phase_transition_resets_best_progress(self):
        thresholds = Thresholds(reach_distance=0.05)
        spec = TaskSpec("test", [ReachObjectPhase(thresholds), GraspObjectPhase(thresholds)])
        tracker = OnlineSubgoalTracker(spec)
        info = tracker.update(state_at(0.04))
        self.assertTrue(info.phase_completed)
        self.assertEqual(tracker.phase_id, 1)
        self.assertEqual(tracker.best_progress_so_far, 0.0)

    def test_phase_id_does_not_decrease(self):
        thresholds = Thresholds(reach_distance=0.05)
        spec = TaskSpec("test", [ReachObjectPhase(thresholds), GraspObjectPhase(thresholds)])
        tracker = OnlineSubgoalTracker(spec)
        tracker.update(state_at(0.04))
        tracker.update(state_at(0.20))
        self.assertEqual(tracker.phase_id, 1)

    def test_one_tracker_per_rollout_instance(self):
        engine = LiberoSubgoalRewardEngine({"enabled": True})
        obs_near = {
            "robot0_eef_pos": np.array([0.0, 0.0, 0.0]),
            "robot0_gripper_qpos": np.array([0.04, 0.04]),
            "cube_pos": np.array([0.04, 0.0, 0.0]),
            "target_pos": np.array([0.0, 0.2, 0.0]),
        }
        obs_far = dict(obs_near)
        obs_far["cube_pos"] = np.array([0.15, 0.0, 0.0])
        metadata = {"task_name": "libero_test", "instruction": "put the cube on the plate"}
        engine.step(0, None, None, obs_near, np.zeros(7), 0.0, False, {}, metadata)
        engine.step(1, None, None, obs_far, np.zeros(7), 0.0, False, {}, metadata)
        self.assertEqual(engine.trackers[0].phase_id, 1)
        self.assertEqual(engine.trackers[1].phase_id, 0)

    def test_resetting_one_tracker_does_not_affect_others(self):
        engine = LiberoSubgoalRewardEngine({"enabled": True})
        engine.trackers[0] = OnlineSubgoalTracker()
        engine.trackers[1] = OnlineSubgoalTracker()
        engine.reset(0)
        self.assertNotIn(0, engine.trackers)
        self.assertIn(1, engine.trackers)

    def test_groupwise_normalization_uses_group_id_not_global(self):
        rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
        mask = torch.ones_like(rewards)
        group_id = np.array(["a", "a", "b", "b"], dtype=object)
        advantages, _ = compute_grpo_outcome_advantage(rewards, mask, group_id)
        self.assertAlmostEqual(float(advantages[0, 0]), float(advantages[2, 0]), places=5)
        self.assertAlmostEqual(float(advantages[1, 0]), float(advantages[3, 0]), places=5)

    def test_disabled_engine_preserves_env_reward(self):
        engine = LiberoSubgoalRewardEngine({"enabled": False})
        _, reward_parts = engine.step(0, None, None, {}, np.zeros(7), 0.75, False, {}, {})
        self.assertEqual(reward_parts["reward_total"], 0.75)
        self.assertEqual(reward_parts["reward_subgoal"], 0.0)

    def test_reward_manager_subgoal_modes(self):
        actor_model = self._cfg(action_token_len=7)
        actor_rollout_ref = self._cfg(model=actor_model)
        verifier = self._cfg(reward_coef=5)
        reward = self._cfg(subgoal={"enabled": True, "mode": "log_only", "log": True})
        config = self._cfg(actor_rollout_ref=actor_rollout_ref, verifier=verifier, reward=reward)
        manager = RobRewardManager(num_examine=0, config=config)

        batch = TensorDict(
            {
                "responses": torch.zeros((1, 2, 7), dtype=torch.long),
                "finish_step": torch.tensor([2], dtype=torch.long),
                "complete": torch.tensor([True], dtype=torch.bool),
                "reward_total": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                "acc": torch.tensor([1.0], dtype=torch.float32),
                "format_correctness": torch.tensor([1.0], dtype=torch.float32),
            },
            batch_size=[1],
        )

        results = {}
        for mode in ("log_only", "add", "replace"):
            config.reward.subgoal["mode"] = mode
            reward_tensor_dict, metrics = manager(DataProto(batch=batch.clone()))
            results[mode] = (
                reward_tensor_dict["all"].sum().item(),
                reward_tensor_dict["subgoal_scores"].sum().item(),
                metrics["subgoal_dense"],
            )

        self.assertAlmostEqual(results["log_only"][0], 5.0, places=5)
        self.assertAlmostEqual(results["add"][0], 5.3, places=5)
        self.assertAlmostEqual(results["replace"][0], 0.3, places=5)
        for _, subgoal_sum, metric_value in results.values():
            self.assertAlmostEqual(subgoal_sum, 0.3, places=5)
            self.assertAlmostEqual(metric_value, 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
