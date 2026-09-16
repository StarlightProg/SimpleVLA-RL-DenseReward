import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayTrainer


def make_trainer(mode: str):
    trainer = RayTrainer.__new__(RayTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "accuracy_lower_bound": 0.0,
                "accuracy_upper_bound": 1.0,
                "filter_accuracy": True,
                "filter_truncated": False,
                "task_sampling": {
                    "enabled": True,
                    "mode": mode,
                    "clip_target_accuracy": 0.1,
                },
            }
        }
    )
    return trainer


class TaskSamplingTrainerTest(unittest.TestCase):
    def test_clip_hard_uses_clip_target_as_lower_bound(self):
        self.assertEqual(make_trainer("clip_hard")._effective_accuracy_lower_bound(), 0.1)
        self.assertEqual(make_trainer("balanced_hard")._effective_accuracy_lower_bound(), 0.0)

    def test_clip_hard_retry_prompts_selects_only_failed_groups(self):
        trainer = make_trainer("clip_hard")
        prompt_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[5], [6], [7]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0], [0], [0]], dtype=torch.int64),
                },
                batch_size=[3],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"] * 3, dtype=object),
            },
        )
        roll_batch = DataProto(
            batch=TensorDict(
                {
                    # n_samples=4 gives group accuracies: 0.00, 0.25, 1.00.
                    "acc": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32),
                },
                batch_size=[12],
            )
        )

        retry_prompts = trainer._clip_hard_retry_prompts(prompt_batch, roll_batch, n_samples=4)

        self.assertEqual(len(retry_prompts), 1)
        self.assertEqual(int(retry_prompts.batch["task_id"][0].item()), 5)

    def test_clip_hard_retry_drops_stale_prompt_runtime_ids_before_concat(self):
        trainer = make_trainer("clip_hard")
        prompt_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[1], [2]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0], [0]], dtype=torch.int64),
                },
                batch_size=[2],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"] * 2, dtype=object),
                "uid": np.array(["old-failed", "old-good"], dtype=object),
                "group_id": np.array(["old-group-failed", "old-group-good"], dtype=object),
            },
        )
        roll_batch = DataProto(
            batch=TensorDict(
                {
                    "acc": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32),
                },
                batch_size=[8],
            )
        )
        fresh_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.arange(7, dtype=torch.int64).reshape(7, 1),
                    "trial_id": torch.zeros((7, 1), dtype=torch.int64),
                },
                batch_size=[7],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"] * 7, dtype=object),
            },
        )

        retry_prompts = trainer._clip_hard_retry_prompts(prompt_batch, roll_batch, n_samples=4)
        newbatch = trainer._concat_data_proto([retry_prompts, fresh_batch])

        self.assertEqual(len(retry_prompts), 1)
        self.assertNotIn("uid", retry_prompts.non_tensor_batch)
        self.assertNotIn("group_id", retry_prompts.non_tensor_batch)
        self.assertEqual(len(newbatch), 8)

    def test_clip_hard_retry_generation_does_not_mix_fresh_prompts(self):
        trainer = make_trainer("clip_hard")
        retry_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[1]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0]], dtype=torch.int64),
                },
                batch_size=[1],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"], dtype=object),
            },
        )

        self.assertFalse(trainer._should_fetch_fresh_prompts(True, retry_batch))
        self.assertTrue(trainer._should_fetch_fresh_prompts(False, retry_batch))
        self.assertTrue(trainer._should_fetch_fresh_prompts(True, []))

    def test_clip_hard_retry_padding_repeats_failed_prompts_for_workers(self):
        trainer = make_trainer("clip_hard")
        retry_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[4]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0]], dtype=torch.int64),
                },
                batch_size=[1],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"], dtype=object),
            },
        )

        padded = trainer._pad_clip_retry_prompts_for_workers(retry_batch, worker_count=2)

        self.assertEqual(len(padded), 2)
        self.assertEqual(padded.batch["task_id"].reshape(-1).tolist(), [4, 4])
        self.assertEqual(padded.batch["trial_id"].reshape(-1).tolist(), [0, 0])

    def test_clip_hard_retry_padding_makes_prompt_count_divisible_by_workers(self):
        trainer = make_trainer("clip_hard")
        retry_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[0], [4], [6]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0], [1], [2]], dtype=torch.int64),
                },
                batch_size=[3],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"] * 3, dtype=object),
            },
        )

        padded = trainer._pad_clip_retry_prompts_for_workers(retry_batch, worker_count=2)

        self.assertEqual(len(padded), 4)
        self.assertEqual(padded.batch["task_id"].reshape(-1).tolist(), [0, 4, 6, 0])
        self.assertEqual(padded.batch["trial_id"].reshape(-1).tolist(), [0, 1, 2, 0])

    def test_clip_hard_padding_is_dropped_before_scoring(self):
        trainer = make_trainer("clip_hard")
        retry_batch = DataProto(
            batch=TensorDict(
                {
                    "task_id": torch.tensor([[4]], dtype=torch.int64),
                    "trial_id": torch.tensor([[0]], dtype=torch.int64),
                },
                batch_size=[1],
            ),
            non_tensor_batch={
                "task_suite_name": np.array(["libero_spatial"], dtype=object),
            },
        )

        padded = trainer._pad_clip_retry_prompts_for_workers(retry_batch, worker_count=2)
        cleaned = trainer._drop_clip_hard_padding(padded)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.batch["task_id"].reshape(-1).tolist(), [4])
        self.assertNotIn("clip_hard_padding", cleaned.non_tensor_batch)

    def test_prompt_batch_size_must_be_divisible_by_workers(self):
        RayTrainer._validate_prompt_batching_for_workers(batch_size=8, worker_count=2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            RayTrainer._validate_prompt_batching_for_workers(batch_size=3, worker_count=2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            RayTrainer._validate_prompt_batching_for_workers(batch_size=1, worker_count=2)


if __name__ == "__main__":
    unittest.main()
