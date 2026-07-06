from collections import Counter


def accuracy_distribution_counts(values, n_samples: int):
    """Bucket per-prompt mean accuracies onto exact n_samples fractions."""

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    counts = Counter()
    for value in values:
        bucket = round(float(value) * n_samples) / n_samples
        bucket = min(1.0, max(0.0, bucket))
        counts[bucket] += 1
    return counts


def accuracy_distribution_metric_values(step_counts, cumulative_counts, n_samples: int):
    buckets = [i / n_samples for i in range(1, n_samples + 1)]
    metrics = {}
    for bucket in buckets:
        suffix = f"{bucket:.2f}"
        metrics[f"train_accuracy_distribution/step_{suffix}"] = int(step_counts.get(bucket, 0))
        metrics[f"train_accuracy_distribution/cumulative_{suffix}"] = int(cumulative_counts.get(bucket, 0))
    return metrics
