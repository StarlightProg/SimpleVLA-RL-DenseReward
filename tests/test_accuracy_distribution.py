import unittest
from collections import Counter

from verl.utils.accuracy_distribution import accuracy_distribution_counts, accuracy_distribution_metric_values


class AccuracyDistributionTest(unittest.TestCase):
    def test_buckets_four_sample_accuracy_values(self):
        counts = accuracy_distribution_counts([0.25, 0.5, 0.75, 1.0, 0.999999], n_samples=4)

        self.assertEqual(counts[0.25], 1)
        self.assertEqual(counts[0.5], 1)
        self.assertEqual(counts[0.75], 1)
        self.assertEqual(counts[1.0], 2)

    def test_metric_values_include_step_and_cumulative_counts(self):
        metrics = accuracy_distribution_metric_values(
            step_counts=Counter({0.25: 1, 1.0: 2}),
            cumulative_counts=Counter({0.25: 3, 0.5: 4, 0.75: 5, 1.0: 6}),
            n_samples=4,
        )

        self.assertEqual(metrics["train_accuracy_distribution/step_0.25"], 1)
        self.assertEqual(metrics["train_accuracy_distribution/step_0.50"], 0)
        self.assertEqual(metrics["train_accuracy_distribution/step_0.75"], 0)
        self.assertEqual(metrics["train_accuracy_distribution/step_1.00"], 2)
        self.assertEqual(metrics["train_accuracy_distribution/cumulative_0.25"], 3)
        self.assertEqual(metrics["train_accuracy_distribution/cumulative_0.50"], 4)
        self.assertEqual(metrics["train_accuracy_distribution/cumulative_0.75"], 5)
        self.assertEqual(metrics["train_accuracy_distribution/cumulative_1.00"], 6)


if __name__ == "__main__":
    unittest.main()
