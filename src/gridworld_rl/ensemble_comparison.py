"""Paired coverage and accuracy comparison of saved ensemble predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
)
from .ensemble_evaluation import evaluate_ensemble_csv
from .evaluation import overlap_slice_masks
from .reproducibility import sha256_file
from .uncertainty import validate_bootstrap_options


def _saved_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    accepted = []
    suggested = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, strict=True):
            accepted.append(row["action"] != "")
            suggested.append(int(row.get("suggested_action") or row["action"]))
    return np.asarray(accepted, dtype=bool), np.asarray(suggested, dtype=np.int64)


def _bootstrap_differences(
    states,
    left_accepted,
    right_accepted,
    left_correct,
    right_correct,
    *,
    replicates,
    seed,
    confidence_level,
) -> dict[str, Any]:
    unique_states, inverse = np.unique(states, return_inverse=True)
    clusters = len(unique_states)
    sizes = np.bincount(inverse)
    left_counts = np.bincount(inverse, weights=left_accepted.astype(np.int64))
    right_counts = np.bincount(inverse, weights=right_accepted.astype(np.int64))
    shared = left_accepted & right_accepted
    shared_counts = np.bincount(inverse, weights=shared.astype(np.int64))
    left_correct_counts = np.bincount(
        inverse, weights=(shared & left_correct).astype(np.int64)
    )
    right_correct_counts = np.bincount(
        inverse, weights=(shared & right_correct).astype(np.int64)
    )
    coverage_draws = np.empty(replicates)
    accuracy_draws = []
    generator = np.random.default_rng(seed)
    for index in range(replicates):
        sampled = generator.integers(0, clusters, size=clusters)
        coverage_draws[index] = (
            left_counts[sampled].sum() - right_counts[sampled].sum()
        ) / sizes[sampled].sum()
        shared_total = shared_counts[sampled].sum()
        if shared_total:
            accuracy_draws.append(
                (
                    left_correct_counts[sampled].sum()
                    - right_correct_counts[sampled].sum()
                )
                / shared_total
            )
    tail = (1 - confidence_level) / 2

    def interval(estimate, draws):
        return (
            None
            if not len(draws)
            else {
                "estimate": estimate,
                "lower": float(np.quantile(draws, tail)),
                "upper": float(np.quantile(draws, 1 - tail)),
            }
        )

    shared_total = int(shared.sum())
    accuracy_estimate = (
        float(
            (
                left_correct[shared].astype(int) - right_correct[shared].astype(int)
            ).mean()
        )
        if shared_total
        else None
    )
    return {
        "method": "percentile state-cluster bootstrap",
        "resampling_unit": "state",
        "unique_states": clusters,
        "replicates": replicates,
        "shared_accuracy_replicates": len(accuracy_draws),
        "seed": seed,
        "confidence_level": confidence_level,
        "coverage_difference_interval": interval(
            float(left_accepted.mean() - right_accepted.mean()), coverage_draws
        ),
        "shared_accuracy_difference_interval": interval(
            accuracy_estimate, accuracy_draws
        ),
    }


def _slice_comparison(
    mask,
    left_accepted,
    right_accepted,
    left_correct,
    right_correct,
) -> dict[str, Any]:
    rows = int(mask.sum())
    left = mask & left_accepted
    right = mask & right_accepted
    shared = left & right
    shared_rows = int(shared.sum())
    left_rows = int(left.sum())
    right_rows = int(right.sum())
    return {
        "rows": rows,
        "left_accepted": left_rows,
        "right_accepted": right_rows,
        "both_accepted": shared_rows,
        "left_only_accepted": int((left & ~right).sum()),
        "right_only_accepted": int((~left & right).sum()),
        "neither_accepted": int((mask & ~left & ~right).sum()),
        "left_coverage": left_rows / rows if rows else None,
        "right_coverage": right_rows / rows if rows else None,
        "coverage_difference": (left_rows - right_rows) / rows if rows else None,
        "shared_left_accuracy": float(left_correct[shared].mean())
        if shared_rows
        else None,
        "shared_right_accuracy": float(right_correct[shared].mean())
        if shared_rows
        else None,
        "shared_accuracy_difference": (
            float(
                (
                    left_correct[shared].astype(int) - right_correct[shared].astype(int)
                ).mean()
            )
            if shared_rows
            else None
        ),
    }


def compare_ensemble_predictions(
    left_csv: str | Path,
    right_csv: str | Path,
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    train_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Compare accepted rows and paired correctness on the same labeled split."""

    validate_bootstrap_options(bootstrap_replicates, bootstrap_seed, confidence_level)
    paths = {
        "left": Path(left_csv),
        "right": Path(right_csv),
        "challenge": Path(challenge_csv),
        "solution": Path(solution_csv),
    }
    if train_csv is not None:
        paths["train"] = Path(train_csv)
    if paths["left"].resolve() == paths["right"].resolve():
        raise ValueError("Left and right ensemble prediction CSVs must be distinct.")
    fingerprints = {name: sha256_file(path) for name, path in paths.items()}
    left_result = evaluate_ensemble_csv(
        paths["left"], paths["challenge"], paths["solution"]
    )
    right_result = evaluate_ensemble_csv(
        paths["right"], paths["challenge"], paths["solution"]
    )
    if left_result["num_actions"] != right_result["num_actions"]:
        raise ValueError(
            "Ensemble prediction CSVs must have the same action dimensions."
        )
    challenge, solution = load_evaluation_data(
        paths["challenge"], paths["solution"], num_actions=left_result["num_actions"]
    )
    states = challenge["state"].to_numpy(dtype=np.int64)
    targets = solution["action"].to_numpy(dtype=np.int64)
    left_accepted, left_suggested = _saved_actions(paths["left"])
    right_accepted, right_suggested = _saved_actions(paths["right"])
    if len(left_accepted) != len(states) or len(right_accepted) != len(states):
        raise ValueError("Ensemble prediction row counts changed during comparison.")
    shared = left_accepted & right_accepted
    shared_count = int(shared.sum())
    left_correct = left_suggested == targets
    right_correct = right_suggested == targets
    result = {
        "schema_version": 1,
        "num_rows": len(states),
        "inputs": {
            name: {"path": str(path), "sha256": fingerprints[name]}
            for name, path in paths.items()
        },
        "left": {
            "coverage": left_result["coverage"],
            "selective_accuracy": left_result["selective_accuracy"],
        },
        "right": {
            "coverage": right_result["coverage"],
            "selective_accuracy": right_result["selective_accuracy"],
        },
        "coverage_difference": float(left_accepted.mean() - right_accepted.mean()),
        "both_accepted": shared_count,
        "left_only_accepted": int((left_accepted & ~right_accepted).sum()),
        "right_only_accepted": int((~left_accepted & right_accepted).sum()),
        "neither_accepted": int((~left_accepted & ~right_accepted).sum()),
        "shared_left_accuracy": float(left_correct[shared].mean())
        if shared_count
        else None,
        "shared_right_accuracy": float(right_correct[shared].mean())
        if shared_count
        else None,
        "shared_accuracy_difference": float(
            (
                left_correct[shared].astype(int) - right_correct[shared].astype(int)
            ).mean()
        )
        if shared_count
        else None,
        "shared_action_disagreements": int(
            (left_suggested[shared] != right_suggested[shared]).sum()
        ),
        "suggested_action_disagreements": int(
            (left_suggested != right_suggested).sum()
        ),
    }
    if "train" in paths:
        train = load_transition_csv(
            paths["train"], num_actions=left_result["num_actions"]
        )
        result["evaluation_split_diagnostics"] = evaluation_split_diagnostics(
            train, solution, num_actions=left_result["num_actions"]
        )
        result["overlap_sliced_comparison"] = {
            name: _slice_comparison(
                mask, left_accepted, right_accepted, left_correct, right_correct
            )
            for name, mask in overlap_slice_masks(train, solution).items()
        }
    if bootstrap_replicates:
        result["bootstrap"] = _bootstrap_differences(
            states,
            left_accepted,
            right_accepted,
            left_correct,
            right_correct,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=float(confidence_level),
        )
    for name, path in paths.items():
        if sha256_file(path) != fingerprints[name]:
            raise ValueError(
                f"{name.capitalize()} changed during ensemble comparison: {path}."
            )
    return result
