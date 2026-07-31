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

    # ``torch.tensor`` deliberately copies here: pandas can expose a read-only
    # NumPy view, which otherwise triggers a warning even though inference does
    # not mutate the input.
    state_tensor = torch.tensor(np.asarray(states), dtype=torch.long)
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(state_tensor), batch_size):
            batch = state_tensor[start : start + batch_size].to(device)
            predictions.append(model(batch).argmax(dim=1).cpu())
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

    actual = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")
    if len(actual) != len(predicted):
        raise ValueError("targets and predictions must have equal lengths.")
    if len(actual) == 0:
        raise ValueError("Cannot evaluate empty targets.")
    if ((actual < 0) | (actual >= num_actions)).any():
        raise ValueError(f"targets must be in the range 0..{num_actions - 1}.")
    if ((predicted < 0) | (predicted >= num_actions)).any():
        raise ValueError(f"predictions must be in the range 0..{num_actions - 1}.")

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
