"""Audit whether a saved policy chooses actions present in logged training data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import load_checkpoint
from .data import load_transition_csv
from .evaluation import predict_actions
from .reproducibility import resolve_device, sha256_file


def audit_policy_support(
    checkpoint: str | Path,
    train_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Measure exact logged state-action support for observed states.

    A chosen action is supported when it occurs at least once with that state
    in the training CSV. This is a coverage diagnostic, not a value estimate.
    """

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    checkpoint_path = Path(checkpoint)
    train_path = Path(train_csv)
    checkpoint_hash = sha256_file(checkpoint_path)
    train_hash = sha256_file(train_path)
    model, metadata = load_checkpoint(checkpoint_path)
    frame = load_transition_csv(
        train_path, num_states=model.num_states, num_actions=model.num_actions
    )
    states = frame["state"].to_numpy(dtype=np.int64)
    actions = frame["action"].to_numpy(dtype=np.int64)
    observed, inverse, state_rows = np.unique(
        states, return_inverse=True, return_counts=True
    )
    counts = np.zeros((len(observed), model.num_actions), dtype=np.int64)
    np.add.at(counts, (inverse, actions), 1)

    selected_device = resolve_device(device)
    model.to(selected_device)
    chosen = predict_actions(
        model, observed, device=selected_device, batch_size=batch_size
    )
    supported = counts[np.arange(len(observed)), chosen] > 0
    unsupported = observed[~supported].tolist()
    supported_rows = int(state_rows[supported].sum())
    chosen_counts = np.bincount(chosen, minlength=model.num_actions).tolist()
    unsupported_counts = np.bincount(
        chosen[~supported], minlength=model.num_actions
    ).tolist()

    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError(
            "Checkpoint changed during support audit; retry with a stable file."
        )
    if sha256_file(train_path) != train_hash:
        raise ValueError(
            "Training CSV changed during support audit; retry with a stable file."
        )

    return {
        "schema_version": 1,
        "algorithm": metadata["algorithm"],
        "checkpoint_sha256": checkpoint_hash,
        "train_sha256": train_hash,
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "training_rows": len(frame),
        "observed_states": len(observed),
        "unobserved_states": model.num_states - len(observed),
        "supported_observed_states": int(supported.sum()),
        "unsupported_observed_states": unsupported,
        "observed_state_support_rate": float(supported.mean()),
        "logged_row_support_rate": supported_rows / len(frame),
        "chosen_action_counts": chosen_counts,
        "unsupported_chosen_action_counts": unsupported_counts,
    }
