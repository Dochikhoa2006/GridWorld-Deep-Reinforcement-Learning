"""Paired evaluation of multiple saved policies on identical rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import load_checkpoint
from .data import (
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
)
from .evaluation import (
    classification_metrics,
    overlap_sliced_agreement,
    predict_actions,
)
from .reproducibility import resolve_device, sha256_file


def compare_evaluations(
    checkpoints: list[str | Path],
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    train_csv: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare correctness and action agreement on the same evaluation rows."""

    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 0
    ):
        raise ValueError("bootstrap_replicates must be a non-negative integer.")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    checkpoint_paths = [Path(path) for path in checkpoints]
    if len({path.resolve() for path in checkpoint_paths}) != len(checkpoint_paths):
        raise ValueError("Checkpoint paths must be distinct.")
    data_paths = {
        "challenge": Path(challenge_csv),
        "solution": Path(solution_csv),
    }
    if train_csv is not None:
        data_paths["train"] = Path(train_csv)
    checkpoint_hashes = [sha256_file(path) for path in checkpoint_paths]
    data_hashes = {name: sha256_file(path) for name, path in data_paths.items()}
    loaded = [load_checkpoint(path) for path in checkpoint_paths]
    num_states = loaded[0][0].num_states
    num_actions = loaded[0][0].num_actions
    if any(
        model.num_states != num_states or model.num_actions != num_actions
        for model, _ in loaded[1:]
    ):
        raise ValueError("Checkpoints must have identical state and action dimensions.")
    challenge, solution = load_evaluation_data(
        data_paths["challenge"],
        data_paths["solution"],
        num_states=num_states,
        num_actions=num_actions,
    )
    states = challenge["state"].to_numpy(dtype=np.int64)
    targets = solution["action"].to_numpy(dtype=np.int64)
    selected_device = resolve_device(device)
    predictions = [
        predict_actions(
            model.to(selected_device),
            states,
            device=selected_device,
            batch_size=batch_size,
        )
        for model, _ in loaded
    ]
    correct = [prediction == targets for prediction in predictions]
    result: dict[str, Any] = {
        "schema_version": 1,
        "num_states": num_states,
        "num_actions": num_actions,
        "evaluation_rows": len(targets),
        "tie_breaking": "lowest action index among exact maximum Q-values",
        "inputs": {
            name: {"path": str(path), "sha256": data_hashes[name]}
            for name, path in data_paths.items()
        },
        "checkpoints": [
            {
                "path": str(path),
                "sha256": fingerprint,
                "algorithm": metadata["algorithm"],
                "metrics": classification_metrics(
                    targets, prediction, num_actions=num_actions
                ),
            }
            for path, fingerprint, (_, metadata), prediction in zip(
                checkpoint_paths, checkpoint_hashes, loaded, predictions, strict=True
            )
        ],
        "pairs": [
            {
                "left": left,
                "right": right,
                "both_correct": int(np.count_nonzero(correct[left] & correct[right])),
                "left_only_correct": int(
                    np.count_nonzero(correct[left] & ~correct[right])
                ),
                "right_only_correct": int(
                    np.count_nonzero(~correct[left] & correct[right])
                ),
                "both_wrong": int(np.count_nonzero(~correct[left] & ~correct[right])),
                "action_agreement": int(
                    np.count_nonzero(predictions[left] == predictions[right])
                ),
                "action_agreement_rate": float(
                    np.mean(predictions[left] == predictions[right])
                ),
            }
            for left in range(len(loaded))
            for right in range(left + 1, len(loaded))
        ],
    }
    if "train" in data_paths:
        train = load_transition_csv(
            data_paths["train"], num_states=num_states, num_actions=num_actions
        )
        result["evaluation_split_diagnostics"] = evaluation_split_diagnostics(
            train, solution, num_states=num_states, num_actions=num_actions
        )
        for row, prediction in zip(result["checkpoints"], predictions, strict=True):
            row["overlap_sliced_agreement"] = overlap_sliced_agreement(
                train, solution, targets, prediction
            )

    if bootstrap_replicates:
        _add_cluster_bootstrap(
            result,
            states,
            correct,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=float(confidence_level),
        )

    for path, fingerprint in zip(checkpoint_paths, checkpoint_hashes, strict=True):
        if sha256_file(path) != fingerprint:
            raise ValueError(
                f"Checkpoint changed during paired evaluation: {path}; "
                "retry with stable files."
            )
    for name, path in data_paths.items():
        if sha256_file(path) != data_hashes[name]:
            raise ValueError(
                f"{name.capitalize()} changed during paired evaluation: {path}; "
                "retry with stable files."
            )
    return result


def _add_cluster_bootstrap(
    result: dict[str, Any],
    states: np.ndarray,
    correct: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> None:
    """Resample state IDs, retaining all rows belonging to each sampled state."""

    unique_states, inverse = np.unique(states, return_inverse=True)
    cluster_count = len(unique_states)
    cluster_sizes = np.bincount(inverse)
    cluster_correct = np.stack(
        [np.bincount(inverse, weights=flags.astype(np.int64)) for flags in correct]
    )
    pairs = result["pairs"]
    deltas = np.empty((len(pairs), replicates), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        denominator = int(cluster_sizes[sampled].sum())
        accuracies = cluster_correct[:, sampled].sum(axis=1) / denominator
        for index, pair in enumerate(pairs):
            deltas[index, replicate] = (
                accuracies[pair["left"]] - accuracies[pair["right"]]
            )
    tail = (1 - confidence_level) / 2
    for index, pair in enumerate(pairs):
        pair["accuracy_difference_interval"] = {
            "estimate": (pair["left_only_correct"] - pair["right_only_correct"])
            / result["evaluation_rows"],
            "lower": float(np.quantile(deltas[index], tail)),
            "upper": float(np.quantile(deltas[index], 1 - tail)),
        }
    result["bootstrap"] = {
        "method": "percentile state-cluster bootstrap",
        "resampling_unit": "state",
        "unique_states": cluster_count,
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "estimand": "left row accuracy minus right row accuracy",
    }
