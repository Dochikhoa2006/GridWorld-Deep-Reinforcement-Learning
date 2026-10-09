"""Evaluate a saved checkpoint on an aligned challenge and solution pair."""

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


def evaluate_checkpoint(
    checkpoint: str | Path,
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    train_csv: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Return reproducible agreement metrics without changing run artifacts."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    paths = {
        "checkpoint": Path(checkpoint),
        "challenge": Path(challenge_csv),
        "solution": Path(solution_csv),
    }
    if train_csv is not None:
        paths["train"] = Path(train_csv)
    fingerprints = {name: sha256_file(path) for name, path in paths.items()}

    model, metadata = load_checkpoint(paths["checkpoint"])
    challenge, solution = load_evaluation_data(
        paths["challenge"],
        paths["solution"],
        num_states=model.num_states,
        num_actions=model.num_actions,
    )
    selected_device = resolve_device(device)
    model.to(selected_device)
    predictions = predict_actions(
        model,
        challenge["state"].to_numpy(dtype=np.int64),
        device=selected_device,
        batch_size=batch_size,
    )
    targets = solution["action"].to_numpy(dtype=np.int64)
    result: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": metadata["algorithm"],
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "tie_breaking": "lowest action index among exact maximum Q-values",
        "inputs": {
            name: {"path": str(path), "sha256": fingerprints[name]}
            for name, path in paths.items()
        },
        "metrics": classification_metrics(
            targets, predictions, num_actions=model.num_actions
        ),
    }
    if "train" in paths:
        train = load_transition_csv(
            paths["train"],
            num_states=model.num_states,
            num_actions=model.num_actions,
        )
        result["evaluation_split_diagnostics"] = evaluation_split_diagnostics(
            train,
            solution,
            num_states=model.num_states,
            num_actions=model.num_actions,
        )
        result["overlap_sliced_agreement"] = overlap_sliced_agreement(
            train, solution, targets, predictions
        )

    for name, path in paths.items():
        if sha256_file(path) != fingerprints[name]:
            raise ValueError(
                f"{name.capitalize()} changed during checkpoint evaluation: "
                f"{path}; retry with stable files."
            )
    return result
