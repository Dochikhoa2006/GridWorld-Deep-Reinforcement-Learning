"""Streaming majority-vote predictions from several saved checkpoints."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .inference import iter_state_batches
from .reproducibility import resolve_device, sha256_file


def _write_ensemble_batch(writer, models, paths, states, start, device, num_actions):
    batch = torch.tensor(states, dtype=torch.long, device=device)
    member_actions = []
    with torch.inference_mode():
        for model, path in zip(models, paths, strict=True):
            values = model(batch)
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"Checkpoint {path} produces non-finite Q-values at input rows "
                    f"{start}..{start + len(states) - 1}."
                )
            member_actions.append(values.argmax(dim=1).cpu().tolist())
    for index, state in enumerate(states):
        actions = [member[index] for member in member_actions]
        votes = [actions.count(action) for action in range(num_actions)]
        winner = max(range(num_actions), key=votes.__getitem__)
        winning_votes = votes[winner]
        writer.writerow(
            [
                state,
                winner,
                winning_votes,
                winning_votes / len(models),
                winning_votes == len(models),
                ";".join(map(str, actions)),
                *votes,
            ]
        )


def predict_ensemble_csv(
    checkpoints: list[str | Path],
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Path:
    """Write a majority-vote CSV in input order, preserving duplicate states."""

    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
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
            writer.writerow(
                [
                    "state",
                    "action",
                    "votes",
                    "agreement_fraction",
                    "unanimous",
                    "member_actions",
                ]
                + [f"votes_{action}" for action in range(num_actions)]
            )
            for start, states in iter_state_batches(source, num_states, batch_size):
                _write_ensemble_batch(
                    writer, models, paths, states, start, selected_device, num_actions
                )
        for path, fingerprint in zip(paths, fingerprints, strict=True):
            if sha256_file(path) != fingerprint:
                raise ValueError(f"Checkpoint changed during prediction: {path}.")
        if sha256_file(source) != source_fingerprint:
            raise ValueError(
                "State CSV changed during prediction; retry with a stable file."
            )
        os.link(temporary_path, destination)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Invalid state CSV {source}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
