"""State-cluster bootstrap intervals for paired action-agreement comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_bootstrap_options(
    replicates: int, seed: int, confidence_level: float
) -> None:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 0
    ):
        raise ValueError("bootstrap_replicates must be a non-negative integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be strictly between 0 and 1.")


def state_cluster_intervals(
    states: np.ndarray,
    correct: Sequence[np.ndarray],
    pairs: Sequence[tuple[int, int]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
    estimand: str,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Resample state IDs, preserving every row belonging to each draw."""

    unique_states, inverse = np.unique(states, return_inverse=True)
    cluster_count = len(unique_states)
    cluster_sizes = np.bincount(inverse)
    cluster_correct = np.stack(
        [np.bincount(inverse, weights=flags.astype(np.int64)) for flags in correct]
    )
    deltas = np.empty((len(pairs), replicates), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        denominator = int(cluster_sizes[sampled].sum())
        accuracies = cluster_correct[:, sampled].sum(axis=1) / denominator
        for index, (left, right) in enumerate(pairs):
            deltas[index, replicate] = accuracies[left] - accuracies[right]
    tail = (1 - confidence_level) / 2
    intervals = [
        {
            "estimate": float(
                (correct[left].sum() - correct[right].sum()) / len(states)
            ),
            "lower": float(np.quantile(deltas[index], tail)),
            "upper": float(np.quantile(deltas[index], 1 - tail)),
        }
        for index, (left, right) in enumerate(pairs)
    ]
    metadata = {
        "method": "percentile state-cluster bootstrap",
        "resampling_unit": "state",
        "unique_states": cluster_count,
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "estimand": estimand,
    }
    return intervals, metadata
