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


if __name__ == "__main__":
    unittest.main()
