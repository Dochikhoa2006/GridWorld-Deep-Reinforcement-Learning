"""Streaming majority-vote predictions from several saved checkpoints."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from numbers import Real
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .inference import (
    eligible_prediction_actions,
    iter_state_batches,
    load_prediction_support,
)
from .reproducibility import resolve_device, sha256_file
from .supported_policy import select_supported_action


def _write_ensemble_batch(
    writer,
    models,
    paths,
    states,
    start,
    device,
    num_actions,
    action_counts=None,
    min_action_count=1,
    weights=None,
):
    batch = torch.tensor(states, dtype=torch.long, device=device)
    member_values = []
    with torch.inference_mode():
        for model, path in zip(models, paths, strict=True):
            values = model(batch)
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"Checkpoint {path} produces non-finite Q-values at input rows "
                    f"{start}..{start + len(states) - 1}."
                )
            member_values.append(values.cpu().tolist())
    for index, state in enumerate(states):
        q_rows = [member[index] for member in member_values]
        raw_actions = [max(range(num_actions), key=row.__getitem__) for row in q_rows]
        counts = action_counts.get(state, {}) if action_counts is not None else {}
        eligible, fallback = eligible_prediction_actions(counts, min_action_count)
        actions = [select_supported_action(row, eligible)[0] for row in q_rows]
        votes = [actions.count(action) for action in range(num_actions)]
        weighted_votes = (
            [
                math.fsum(
                    weight
                    for action, weight in zip(actions, weights, strict=True)
                    if action == candidate
                )
                for candidate in range(num_actions)
            ]
            if weights is not None
            else None
        )
        winner = max(
            range(num_actions),
            key=(weighted_votes if weighted_votes is not None else votes).__getitem__,
        )
        winning_votes = votes[winner]
        row = [
            state,
            winner,
            winning_votes,
            winning_votes / len(models),
            winning_votes == len(models),
            ";".join(map(str, actions)),
        ]
        if weighted_votes is not None:
            row.extend([weighted_votes[winner], weighted_votes[winner] / sum(weights)])
        if action_counts is not None:
            raw_votes = (
                [
                    math.fsum(
                        weight
                        for action, weight in zip(raw_actions, weights, strict=True)
                        if action == candidate
                    )
                    for candidate in range(num_actions)
                ]
                if weights is not None
                else [raw_actions.count(action) for action in range(num_actions)]
            )
            raw_winner = max(range(num_actions), key=raw_votes.__getitem__)
            row.extend(
                [
                    raw_winner,
                    ";".join(map(str, raw_actions)),
                    winner != raw_winner,
                    ";".join(map(str, sorted(counts))),
                    ";".join(map(str, eligible)),
                    ";".join(f"{action}:{counts[action]}" for action in sorted(counts)),
                    fallback,
                ]
            )
        writer.writerow(
            [*row, *votes, *([] if weighted_votes is None else weighted_votes)]
        )


def predict_ensemble_csv(
    checkpoints: list[str | Path],
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
    train_csv: str | Path | None = None,
    min_action_count: int = 1,
    weights: list[float] | None = None,
) -> Path:
    """Write a majority-vote CSV in input order, preserving duplicate states."""

    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required.")
    if weights is not None and (
        len(weights) != len(checkpoints)
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(weight)
            or weight <= 0
            for weight in weights
        )
        or not math.isfinite(sum(weights))
    ):
        raise ValueError(
            "weights must contain one positive finite value per checkpoint with a finite total."
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if (
        isinstance(min_action_count, bool)
        or not isinstance(min_action_count, int)
        or min_action_count <= 0
    ):
        raise ValueError("min_action_count must be a positive integer.")
    if train_csv is None and min_action_count != 1:
        raise ValueError("min_action_count requires a training CSV.")
    paths = [Path(path) for path in checkpoints]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Checkpoint paths must be distinct.")
    destination = Path(output_csv)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Prediction output already exists: {destination}")
    source = Path(input_csv)
    if not source.is_file():
        raise FileNotFoundError(f"State CSV not found: {source}")
    selected_device = resolve_device(device)
    fingerprints = [sha256_file(path) for path in paths]
    models = [load_checkpoint(path)[0].to(selected_device) for path in paths]
    num_states, num_actions = models[0].num_states, models[0].num_actions
    if any(
        model.num_states != num_states or model.num_actions != num_actions
        for model in models[1:]
    ):
        raise ValueError("Checkpoints must have identical state and action dimensions.")
    source_fingerprint = sha256_file(source)
    train_path = Path(train_csv) if train_csv is not None else None
    train_fingerprint = sha256_file(train_path) if train_path is not None else None
    action_counts = (
        load_prediction_support(train_path, num_states, num_actions)
        if train_path is not None
        else None
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary)
            header = [
                "state",
                "action",
                "votes",
                "agreement_fraction",
                "unanimous",
                "member_actions",
            ]
            if weights is not None:
                header.extend(["vote_weight", "agreement_weight_fraction"])
            if action_counts is not None:
                header.extend(
                    [
                        "unconstrained_action",
                        "unconstrained_member_actions",
                        "was_constrained",
                        "logged_actions",
                        "eligible_actions",
                        "logged_action_counts",
                        "support_fallback",
                    ]
                )
            writer.writerow(
                header
                + [f"votes_{action}" for action in range(num_actions)]
                + (
                    []
                    if weights is None
                    else [f"weight_{action}" for action in range(num_actions)]
                )
            )
            for start, states in iter_state_batches(source, num_states, batch_size):
                _write_ensemble_batch(
                    writer,
                    models,
                    paths,
                    states,
                    start,
                    selected_device,
                    num_actions,
                    action_counts,
                    min_action_count,
                    weights,
                )
        for path, fingerprint in zip(paths, fingerprints, strict=True):
            if sha256_file(path) != fingerprint:
                raise ValueError(f"Checkpoint changed during prediction: {path}.")
        if sha256_file(source) != source_fingerprint:
            raise ValueError(
                "State CSV changed during prediction; retry with a stable file."
            )
        if train_path is not None and sha256_file(train_path) != train_fingerprint:
            raise ValueError(
                "Training CSV changed during prediction; retry with a stable file."
            )
        os.link(temporary_path, destination)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Invalid state CSV {source}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
