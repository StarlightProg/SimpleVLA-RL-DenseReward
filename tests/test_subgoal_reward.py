import unittest
import importlib
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from verl.utils.subgoal_reward.engine import LiberoSubgoalRewardEngine
from verl.utils.subgoal_reward.libero_state import LiberoStateExtractor
from verl.utils.subgoal_reward.libero_state import LiberoState
from verl.utils.subgoal_reward.phases import GraspObjectPhase, ReachObjectPhase, Thresholds
from verl.utils.subgoal_reward.task_specs import TaskSpec
from verl.utils.subgoal_reward.tracker import OnlineSubgoalTracker
from verl.utils.validation_video import select_validation_video_indices

try:
    from tensordict import TensorDict
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.main_ppo import RobRewardManager
except ImportError:
    TensorDict = None
    DataProto = None
    compute_grpo_outcome_advantage = None
    RobRewardManager = None


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


def libero_obs(eef, bowl, plate, gripper=0.04):
    return {
        "robot0_eef_pos": np.array(eef, dtype=np.float32),
        "robot0_gripper_qpos": np.array([gripper, gripper], dtype=np.float32),
        "akita_black_bowl_1_pos": np.array(bowl, dtype=np.float32),
        "akita_black_bowl_2_pos": np.array([-0.20, 0.32, 0.90], dtype=np.float32),
        "glazed_rim_porcelain_ramekin_1_pos": np.array([-0.20, 0.19, 0.90], dtype=np.float32),
        "plate_1_pos": np.array(plate, dtype=np.float32),
    }


def load_libero_video_utils():
    if "tensorflow" not in sys.modules:
        try:
            importlib.import_module("tensorflow")
        except ImportError:
            fake_tf = types.ModuleType("tensorflow")

            class FakeConfig:
                @staticmethod
                def set_visible_devices(*args, **kwargs):
                    return None

            fake_tf.config = FakeConfig()
            sys.modules["tensorflow"] = fake_tf

    if "imageio" not in sys.modules:
        try:
            importlib.import_module("imageio")
        except ImportError:
            fake_imageio = types.ModuleType("imageio")

            class FakeWriter:
                def __init__(self, path, fps=30):
                    self._file = open(path, "wb")

                def append_data(self, img):
                    self._file.write(np.asarray(img, dtype=np.uint8).tobytes())

                def close(self):
                    self._file.close()

            fake_imageio.get_writer = lambda path, fps=30: FakeWriter(path, fps)
            sys.modules["imageio"] = fake_imageio

    libero_utils = importlib.import_module("verl.utils.libero_utils")
    return libero_utils._default_rollout_root, libero_utils.save_rollout_video


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

    def test_validation_video_selection_uses_shard_local_mask(self):
        meta_info = {"validation_video_mask": np.array([True, False, True, False], dtype=np.bool_)}
        first_shard = {"validation_video_selected": np.array([True, False], dtype=object)}
        second_shard = {"validation_video_selected": np.array([True, False], dtype=object)}

        self.assertEqual(
            select_validation_video_indices(first_shard, meta_info, batch_size=2, n_samples=1, video_max_episodes=10),
            {0},
        )
        self.assertEqual(
            select_validation_video_indices(second_shard, meta_info, batch_size=2, n_samples=1, video_max_episodes=10),
            {0},
        )

    def test_save_rollout_video_uses_repo_rollouts_dir(self):
        default_rollout_root, save_rollout_video = load_libero_video_utils()
        exp_name = "__test_video_rollout__"
        frames = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.full((16, 16, 3), 255, dtype=np.uint8),
        ]

        mp4_path = Path(
            save_rollout_video(
                frames,
                exp_name,
                "dummy_task",
                123,
                True,
                rollout_root=default_rollout_root(),
            )
        )
        try:
            self.assertTrue(mp4_path.exists())
            self.assertGreater(mp4_path.stat().st_size, 0)
            self.assertEqual(mp4_path.suffix, ".mp4")
            self.assertEqual(mp4_path.parent, default_rollout_root() / exp_name)
        finally:
            if mp4_path.exists():
                mp4_path.unlink()
            test_dir = default_rollout_root() / exp_name
            if test_dir.exists() and not any(test_dir.iterdir()):
                test_dir.rmdir()

    def test_bddl_obj_of_interest_selects_exact_libero_object_and_target(self):
        with TemporaryDirectory() as tmpdir:
            bddl_path = Path(tmpdir) / "task.bddl"
            bddl_path.write_text(
                """
                (:obj_of_interest
                  akita_black_bowl_1
                  plate_1
                )
                (:goal
                  (And (On akita_black_bowl_1 plate_1))
                )
                """,
                encoding="utf-8",
            )
            obs = libero_obs(
                eef=[-0.21, -0.01, 1.17],
                bowl=[-0.06, 0.20, 0.90],
                plate=[0.05, 0.20, 0.90],
                gripper=0.04,
            )

            state = LiberoStateExtractor().extract(
                obs=obs,
                task_metadata={
                    "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
                    "bddl_file_path": str(bddl_path),
                },
            )

        np.testing.assert_allclose(state.object_position, obs["akita_black_bowl_1_pos"])
        np.testing.assert_allclose(state.target_position, obs["plate_1_pos"])

    def test_perfect_libero_pick_place_completes_all_phases(self):
        with TemporaryDirectory() as tmpdir:
            bddl_path = Path(tmpdir) / "task.bddl"
            bddl_path.write_text(
                """
                (:obj_of_interest
                  akita_black_bowl_1
                  plate_1
                )
                """,
                encoding="utf-8",
            )
            metadata = {
                "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
                "bddl_file_path": str(bddl_path),
            }
            engine = LiberoSubgoalRewardEngine(
                {
                    "enabled": True,
                    "clip_dense_reward": 1.0,
                    "weights": {
                        "subgoal_progress": 0.0,
                        "phase_transition": 1.0,
                        "terminal_success": 0.0,
                        "smoothness": 0.0,
                    },
                }
            )
            plate = np.array([0.05, 0.20, 0.90], dtype=np.float32)
            states = [
                (libero_obs([0.00, 0.00, 1.00], [-0.06, 0.20, 0.90], plate, 0.04), False),
                (libero_obs([-0.06, 0.20, 0.91], [-0.06, 0.20, 0.90], plate, 0.04), False),
                (libero_obs([-0.06, 0.20, 0.91], [-0.06, 0.20, 0.90], plate, 0.00), False),
                (libero_obs([-0.06, 0.20, 1.00], [-0.06, 0.20, 0.99], plate, 0.00), False),
                (libero_obs([0.05, 0.20, 1.00], [0.05, 0.20, 0.99], plate, 0.00), False),
                (libero_obs([0.05, 0.20, 0.93], [0.05, 0.20, 0.90], plate, 0.04), True),
            ]

            completed = 0.0
            phase_reward = 0.0
            previous_obs = states[0][0]
            for obs, done in states[1:]:
                subgoal_info, reward_parts = engine.step(
                    env_index=0,
                    env=None,
                    obs=previous_obs,
                    next_obs=obs,
                    action=np.zeros(7, dtype=np.float32),
                    env_reward=0.0,
                    done=done,
                    info={"success": done},
                    task_metadata=metadata,
                )
                completed += subgoal_info["subgoal_phase_completed"]
                phase_reward += reward_parts["reward_phase"]
                previous_obs = obs

        self.assertEqual(completed, 5.0)
        self.assertEqual(phase_reward, 5.0)

    def test_resetting_one_tracker_does_not_affect_others(self):
        engine = LiberoSubgoalRewardEngine({"enabled": True})
        engine.trackers[0] = OnlineSubgoalTracker()
        engine.trackers[1] = OnlineSubgoalTracker()
        engine.reset(0)
        self.assertNotIn(0, engine.trackers)
        self.assertIn(1, engine.trackers)

    def test_groupwise_normalization_uses_group_id_not_global(self):
        if compute_grpo_outcome_advantage is None:
            self.skipTest("GRPO advantage test requires full RL dependencies")

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
        if TensorDict is None or DataProto is None or RobRewardManager is None:
            self.skipTest("reward manager test requires full RL dependencies")

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
