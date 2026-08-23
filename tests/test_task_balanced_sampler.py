import unittest
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from verl.utils.dataset.rob_dataset import TaskBalancedHardBatchSampler, collate_fn


class FakeTaskDataset(Dataset):
    def __init__(self, num_tasks=10, trials_per_task=20):
        self.data = []
        for task_id in range(num_tasks):
            for trial_id in range(trials_per_task):
                self.data.append(
                    {
                        "task_suite_name": "libero_spatial",
                        "task_id": torch.tensor([task_id], dtype=torch.int64),
                        "trial_id": torch.tensor([trial_id], dtype=torch.int64),
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TaskBalancedHardBatchSamplerTest(unittest.TestCase):
    def test_hard_task_probability_increases_when_success_is_low(self):
        dataset = FakeTaskDataset()
        sampler = TaskBalancedHardBatchSampler(
            dataset,
            batch_size=10,
            uniform_fraction=0.7,
            min_task_probability=0.05,
            ema_momentum=0.0,
            default_success=0.5,
            seed=123,
        )
        sampler.set_task_success({task_id: task_id / 10.0 for task_id in range(10)})
        probs = sampler.task_probabilities()

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertGreaterEqual(min(probs.values()), 0.05)
        self.assertGreater(probs[0], probs[9])

    def test_round_robin_uniform_slots_prevent_task_starvation(self):
        dataset = FakeTaskDataset()
        sampler = TaskBalancedHardBatchSampler(
            dataset,
            batch_size=10,
            uniform_fraction=0.7,
            min_task_probability=0.05,
            ema_momentum=0.0,
            default_success=0.5,
            seed=7,
        )

        batches = []
        for batch_idx, indices in zip(range(4), iter(sampler)):
            del batch_idx
            tasks = [int(dataset[index]["task_id"].item()) for index in indices]
            batches.append(tasks)

        all_counts = Counter(task for batch in batches for task in batch)
        self.assertEqual(set(all_counts), set(range(10)))
        self.assertGreaterEqual(min(all_counts.values()), 2)

    def test_dataloader_collates_balanced_batches(self):
        dataset = FakeTaskDataset(num_tasks=10, trials_per_task=4)
        sampler = TaskBalancedHardBatchSampler(dataset, batch_size=10, seed=11)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

        batch = next(iter(loader))
        self.assertEqual(batch["task_id"].shape[0], 10)
        self.assertEqual(batch["trial_id"].shape[0], 10)
        tasks = batch["task_id"].reshape(-1).tolist()
        self.assertGreaterEqual(len(set(tasks)), 7)


if __name__ == "__main__":
    unittest.main()
