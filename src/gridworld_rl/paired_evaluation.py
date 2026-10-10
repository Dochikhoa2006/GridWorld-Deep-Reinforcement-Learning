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
from .uncertainty import state_cluster_intervals, validate_bootstrap_options


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
    validate_bootstrap_options(bootstrap_replicates, bootstrap_seed, confidence_level)
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
        intervals, metadata = state_cluster_intervals(
            states,
            correct,
            [(pair["left"], pair["right"]) for pair in result["pairs"]],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=float(confidence_level),
            estimand="left row accuracy minus right row accuracy",
        )
        for pair, interval in zip(result["pairs"], intervals, strict=True):
            pair["accuracy_difference_interval"] = interval
        result["bootstrap"] = metadata

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
