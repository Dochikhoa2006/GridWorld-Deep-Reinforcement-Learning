"""Evaluate the effect of restricting a policy to logged actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .data import (
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
)
from .evaluation import classification_metrics, overlap_sliced_agreement
from .reproducibility import resolve_device, sha256_file
from .supported_policy import select_supported_action


def evaluate_supported_policy(
    checkpoint: str | Path,
    train_csv: str | Path,
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Compare constrained and original policy agreement on aligned rows."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    paths = {
        "checkpoint": Path(checkpoint),
        "train": Path(train_csv),
        "challenge": Path(challenge_csv),
        "solution": Path(solution_csv),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    model, metadata = load_checkpoint(paths["checkpoint"])
    train = load_transition_csv(
        paths["train"], num_states=model.num_states, num_actions=model.num_actions
    )
    challenge, solution = load_evaluation_data(
        paths["challenge"],
        paths["solution"],
        num_states=model.num_states,
        num_actions=model.num_actions,
    )
    support = {
        int(state): sorted(set(actions))
        for state, actions in train.groupby("state")["action"]
    }
    selected_device = resolve_device(device)
    model.to(selected_device)
    states = challenge["state"].to_numpy(dtype=np.int64)
    targets = solution["action"].to_numpy(dtype=np.int64)
    constrained = np.empty(len(states), dtype=np.int64)
    original = np.empty(len(states), dtype=np.int64)
    observed_rows = 0
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch_states = states[start : start + batch_size]
            values = model(
                torch.tensor(batch_states, dtype=torch.long, device=selected_device)
            )
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"Checkpoint produces non-finite Q-values at evaluation rows "
                    f"{start}..{start + len(batch_states) - 1}."
                )
            for index, (state, q_values) in enumerate(
                zip(batch_states, values.cpu().tolist(), strict=True)
            ):
                logged_actions = support.get(int(state), [])
                observed_rows += bool(logged_actions)
                action, unrestricted = select_supported_action(q_values, logged_actions)
                constrained[start + index] = action
                original[start + index] = unrestricted
    original_metrics = classification_metrics(
        targets, original, num_actions=model.num_actions
    )
    constrained_metrics = classification_metrics(
        targets, constrained, num_actions=model.num_actions
    )
    original_correct = original == targets
    constrained_correct = constrained == targets
    result = {
        "schema_version": 1,
        "algorithm": metadata["algorithm"],
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "selection_rule": "highest Q-value among logged actions; lowest action index on ties",
        "inputs": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "unconstrained": {
            "metrics": original_metrics,
            "overlap_slices": overlap_sliced_agreement(
                train, solution, targets, original
            ),
        },
        "supported": {
            "metrics": constrained_metrics,
            "overlap_slices": overlap_sliced_agreement(
                train, solution, targets, constrained
            ),
        },
        "paired": {
            "evaluation_rows": len(states),
            "observed_state_rows": observed_rows,
            "unobserved_state_rows": len(states) - observed_rows,
            "changed_rows": int(np.count_nonzero(constrained != original)),
            "both_correct": int(
                np.count_nonzero(original_correct & constrained_correct)
            ),
            "unconstrained_only_correct": int(
                np.count_nonzero(original_correct & ~constrained_correct)
            ),
            "supported_only_correct": int(
                np.count_nonzero(~original_correct & constrained_correct)
            ),
            "both_wrong": int(
                np.count_nonzero(~original_correct & ~constrained_correct)
            ),
            "supported_minus_unconstrained_accuracy": (
                constrained_metrics["accuracy"] - original_metrics["accuracy"]
            ),
        },
        "evaluation_split_diagnostics": evaluation_split_diagnostics(
            train,
            solution,
            num_states=model.num_states,
            num_actions=model.num_actions,
        ),
    }
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(
                f"{name.capitalize()} changed during supported-policy evaluation: "
                f"{path}; retry with stable files."
            )
    return result
