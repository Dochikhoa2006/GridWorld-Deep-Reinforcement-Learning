"""Provided evaluation-split policy diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def predict_actions(
    model: nn.Module,
    states: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict greedy actions without retaining an autograd graph."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    # ``torch.tensor`` deliberately copies here: pandas can expose a read-only
    # NumPy view, which otherwise triggers a warning even though inference does
    # not mutate the input.
    state_tensor = (
        states.detach().to(dtype=torch.long)
        if isinstance(states, torch.Tensor)
        else torch.tensor(np.asarray(states), dtype=torch.long)
    )
    predictions: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(state_tensor), batch_size):
                batch = state_tensor[start : start + batch_size].to(device)
                q_values = model(batch)
                if not torch.isfinite(q_values).all():
                    raise ValueError(
                        f"Non-finite Q-values during prediction at rows "
                        f"{start}..{start + len(batch) - 1}."
                    )
                predictions.append(q_values.argmax(dim=1).cpu())
    finally:
        model.train(was_training)
    if not predictions:
        return np.empty(0, dtype=np.int64)
    return torch.cat(predictions).numpy()


def classification_metrics(
    targets: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
    *,
    num_actions: int = 4,
) -> dict[str, Any]:
    """Compute evaluation-split agreement, confusion, support, and action recall."""

    if (
        isinstance(num_actions, bool)
        or not isinstance(num_actions, (int, np.integer))
        or num_actions <= 0
    ):
        raise ValueError("num_actions must be a positive integer.")

    actual = np.asarray(targets)
    predicted = np.asarray(predictions)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")
    if len(actual) != len(predicted):
        raise ValueError("targets and predictions must have equal lengths.")
    if len(actual) == 0:
        raise ValueError("Cannot evaluate empty targets.")
    for name, labels in (("targets", actual), ("predictions", predicted)):
        if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
            raise ValueError(f"{name} must contain finite integer action labels.")
        if labels.dtype.kind == "f" and (labels != np.floor(labels)).any():
            raise ValueError(f"{name} must contain integer action labels.")
    if ((actual < 0) | (actual >= num_actions)).any():
        raise ValueError(f"targets must be in the range 0..{num_actions - 1}.")
    if ((predicted < 0) | (predicted >= num_actions)).any():
        raise ValueError(f"predictions must be in the range 0..{num_actions - 1}.")
    actual = actual.astype(np.int64)
    predicted = predicted.astype(np.int64)

    confusion = np.zeros((num_actions, num_actions), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(num_actions, dtype=np.float64),
        where=support != 0,
    )
    return {
        "accuracy": float((actual == predicted).mean()),
        "num_examples": len(actual),
        "confusion_matrix": confusion.tolist(),
        "per_action_recall": {
            str(action): float(recall[action]) for action in range(num_actions)
        },
        "per_action_support": {
            str(action): int(support[action]) for action in range(num_actions)
        },
    }
