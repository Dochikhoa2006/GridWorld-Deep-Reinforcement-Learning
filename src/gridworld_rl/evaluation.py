"""Provided evaluation-split policy diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import REQUIRED_COLUMNS


def predict_actions(
    model: nn.Module,
    states: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict greedy actions in batches into a single output array."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    state_values = states if isinstance(states, torch.Tensor) else np.asarray(states)
    predictions = np.empty(len(state_values), dtype=np.int64)
    num_actions: int | None = None
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(state_values), batch_size):
                selection = state_values[start : start + batch_size]
                # torch.tensor copies each NumPy slice, including read-only
                # pandas views, without copying the entire evaluation array.
                batch = (
                    selection.detach().to(device=device, dtype=torch.long)
                    if isinstance(selection, torch.Tensor)
                    else torch.tensor(selection, dtype=torch.long, device=device)
                )
                q_values = model(batch)
                if (
                    not isinstance(q_values, torch.Tensor)
                    or q_values.ndim != 2
                    or q_values.shape[0] != len(batch)
                    or q_values.shape[1] < 2
                    or (num_actions is not None and q_values.shape[1] != num_actions)
                ):
                    raise ValueError(
                        f"Model must return consistent [batch, actions] Q-values "
                        f"during prediction at rows {start}..{start + len(batch) - 1}."
                    )
                num_actions = q_values.shape[1]
                if not torch.isfinite(q_values).all():
                    raise ValueError(
                        f"Non-finite Q-values during prediction at rows "
                        f"{start}..{start + len(batch) - 1}."
                    )
                predictions[start : start + len(batch)] = (
                    q_values.argmax(dim=1).cpu().numpy()
                )
    finally:
        model.train(was_training)
    return predictions


def classification_metrics(
    targets: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
    *,
    num_actions: int = 4,
) -> dict[str, Any]:
    """Compute agreement and class metrics from the evaluation confusion matrix."""

    if (
        isinstance(num_actions, bool)
        or not isinstance(num_actions, (int, np.integer))
        or num_actions <= 0
    ):
        raise ValueError("num_actions must be a positive integer.")

    actual = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )
    predicted = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else np.asarray(predictions)
    )
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
    prediction_count = confusion.sum(axis=0)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(num_actions, dtype=np.float64),
        where=support != 0,
    )
    precision = np.divide(
        np.diag(confusion),
        prediction_count,
        out=np.zeros(num_actions, dtype=np.float64),
        where=prediction_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_actions, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float((actual == predicted).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1[support > 0].mean()),
        "num_examples": len(actual),
        "confusion_matrix": confusion.tolist(),
        "per_action_recall": {
            str(action): float(recall[action]) for action in range(num_actions)
        },
        "per_action_precision": {
            str(action): float(precision[action]) for action in range(num_actions)
        },
        "per_action_f1": {
            str(action): float(f1[action]) for action in range(num_actions)
        },
        "per_action_support": {
            str(action): int(support[action]) for action in range(num_actions)
        },
        "per_action_prediction_count": {
            str(action): int(prediction_count[action]) for action in range(num_actions)
        },
    }


def policy_agreement_matrix(
    predictions: Mapping[str, Sequence[int] | np.ndarray], *, num_actions: int
) -> dict[str, Any]:
    """Compare algorithms on aligned rows without consulting target actions."""

    if not predictions:
        raise ValueError("Policy agreement requires at least one algorithm.")
    if (
        isinstance(num_actions, bool)
        or not isinstance(num_actions, int)
        or num_actions <= 1
    ):
        raise ValueError("num_actions must be an integer greater than 1.")
    algorithms = list(predictions)
    if any(not isinstance(name, str) or not name for name in algorithms):
        raise ValueError("Policy agreement algorithm names must be non-empty strings.")
    arrays = []
    for name, values in predictions.items():
        actions = np.asarray(values)
        if (
            actions.ndim != 1
            or not len(actions)
            or actions.dtype.kind not in "iu"
            or ((actions < 0) | (actions >= num_actions)).any()
        ):
            raise ValueError(f"Invalid policy predictions for {name}.")
        arrays.append(actions)
    num_examples = len(arrays[0])
    if any(len(actions) != num_examples for actions in arrays[1:]):
        raise ValueError("Policy predictions must have aligned row counts.")

    size = len(algorithms)
    agreement = [[1.0] * size for _ in range(size)]
    disagreements = [[0] * size for _ in range(size)]
    for first in range(size):
        for second in range(first + 1, size):
            count = int(np.count_nonzero(arrays[first] != arrays[second]))
            disagreements[first][second] = disagreements[second][first] = count
            rate = (num_examples - count) / num_examples
            agreement[first][second] = agreement[second][first] = rate
    return {
        "algorithms": algorithms,
        "num_examples": num_examples,
        "agreement": agreement,
        "disagreements": disagreements,
    }


EVALUATION_SLICES = (
    "exact_training_transition",
    "seen_state_new_transition",
    "unseen_state",
)


def overlap_sliced_agreement(
    train: pd.DataFrame,
    solution: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, int | float | None]]:
    """Partition evaluation rows by exposure to the fixed training dataset.

    Exact matches use all five transition columns, including reward and done.
    Every evaluation row belongs to exactly one slice. An empty slice has null
    accuracy, not an invented zero or perfect score.
    """

    if len(solution) != len(targets) or len(targets) != len(predictions):
        raise ValueError("Evaluation rows, targets, and predictions must align.")
    columns = list(REQUIRED_COLUMNS)
    train_rows = set(train.loc[:, columns].itertuples(index=False, name=None))
    train_states = set(train["state"])
    exact = np.fromiter(
        (
            row in train_rows
            for row in solution.loc[:, columns].itertuples(index=False, name=None)
        ),
        dtype=bool,
        count=len(solution),
    )
    seen = solution["state"].isin(train_states).to_numpy(dtype=bool)
    masks = (exact, seen & ~exact, ~seen)
    return {
        name: {
            "rows": int(mask.sum()),
            "accuracy": (
                float(np.mean(targets[mask] == predictions[mask]))
                if mask.any()
                else None
            ),
        }
        for name, mask in zip(EVALUATION_SLICES, masks, strict=True)
    }
