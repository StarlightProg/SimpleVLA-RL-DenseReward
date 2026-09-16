import unittest

from omegaconf import OmegaConf

from verl.utils.dataset.rob_dataset import normalize_libero_task_ids


class LiberoTaskFilteringTest(unittest.TestCase):
    def test_null_like_values_select_all_tasks(self):
        self.assertIsNone(normalize_libero_task_ids(None, num_tasks_in_suite=10))
        self.assertIsNone(normalize_libero_task_ids("null", num_tasks_in_suite=10))
        self.assertIsNone(normalize_libero_task_ids("all", num_tasks_in_suite=10))
        self.assertIsNone(normalize_libero_task_ids("", num_tasks_in_suite=10))

    def test_single_integer_and_string_are_supported(self):
        self.assertEqual(normalize_libero_task_ids(1, num_tasks_in_suite=10), [1])
        self.assertEqual(normalize_libero_task_ids("5", num_tasks_in_suite=10), [5])

    def test_comma_separated_and_hydra_lists_are_supported(self):
        self.assertEqual(normalize_libero_task_ids("1,5", num_tasks_in_suite=10), [1, 5])
        self.assertEqual(normalize_libero_task_ids("[1, 5]", num_tasks_in_suite=10), [1, 5])
        self.assertEqual(
            normalize_libero_task_ids(OmegaConf.create([1, 5]), num_tasks_in_suite=10),
            [1, 5],
        )

    def test_duplicate_ids_are_deduplicated_without_reordering(self):
        self.assertEqual(normalize_libero_task_ids("5,1,5", num_tasks_in_suite=10), [5, 1])

    def test_invalid_task_ids_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "outside task suite range"):
            normalize_libero_task_ids("10", num_tasks_in_suite=10)
        with self.assertRaisesRegex(ValueError, "Invalid LIBERO task id"):
            normalize_libero_task_ids("not-a-task", num_tasks_in_suite=10)
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_libero_task_ids([], num_tasks_in_suite=10)


if __name__ == "__main__":
    unittest.main()
