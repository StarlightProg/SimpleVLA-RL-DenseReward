import unittest

import numpy as np

from verl.utils.task_sampling import (
    clip_hard_retry_prompt_mask,
    is_clip_hard_mode,
    make_prompt_group_ids,
    normalize_task_sampling_mode,
    uses_hard_task_sampler,
)


class TaskSamplingUtilsTest(unittest.TestCase):
    def test_balanced_and_clip_modes_use_hard_sampler(self):
        self.assertTrue(uses_hard_task_sampler("balanced_hard"))
        self.assertTrue(uses_hard_task_sampler("clip_hard"))
        self.assertFalse(uses_hard_task_sampler("random"))
        self.assertEqual(normalize_task_sampling_mode(None), "balanced_hard")

    def test_clip_hard_mode_detection(self):
        self.assertTrue(is_clip_hard_mode("clip_hard"))
        self.assertTrue(is_clip_hard_mode(" CLIP_HARD "))
        self.assertFalse(is_clip_hard_mode("balanced_hard"))

    def test_prompt_group_ids_are_unique_for_same_task(self):
        group_ids = make_prompt_group_ids(
            task_suites=np.array(["libero_spatial", "libero_spatial"], dtype=object),
            task_ids=[5, 5],
            trial_ids=[0, 1],
            uids=np.array(["uid-a", "uid-b"], dtype=object),
        )

        self.assertEqual(len(set(group_ids.tolist())), 2)
        self.assertIn("task_5", group_ids[0])
        self.assertIn("trial_0", group_ids[0])

    def test_clip_hard_retry_mask_keeps_only_groups_below_target(self):
        # n_samples=4 gives group accuracies: 0.00, 0.25, 1.00.
        retry_mask = clip_hard_retry_prompt_mask(
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
            n_samples=4,
            target_accuracy=0.1,
        )

        self.assertEqual(retry_mask.tolist(), [True, False, False])

    def test_clip_hard_retry_mask_rejects_bad_grouping(self):
        with self.assertRaises(ValueError):
            clip_hard_retry_prompt_mask([0, 1, 0], n_samples=4, target_accuracy=0.1)


if __name__ == "__main__":
    unittest.main()
