"""Batch inference for a one-column CSV of discrete states."""

from __future__ import annotations

import csv
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .reproducibility import resolve_device, sha256_file


def _write_batch(
    writer,
    model: torch.nn.Module,
    states: list[int],
    *,
    start: int,
    device: torch.device,
) -> None:
    batch = torch.tensor(states, dtype=torch.long, device=device)
    with torch.inference_mode():
        q_values = model(batch)
        if not torch.isfinite(q_values).all():
            raise ValueError(
                f"Checkpoint produces non-finite Q-values at input rows "
                f"{start}..{start + len(batch) - 1}."
            )
        values = q_values.cpu().tolist()
        best = q_values.topk(2, dim=1).values.cpu().double()
        gaps = (best[:, 0] - best[:, 1]).tolist()
        actions = q_values.argmax(dim=1).cpu().tolist()
    for state, action, gap, values_row in zip(
        states, actions, gaps, values, strict=True
    ):
        writer.writerow([state, action, gap, *values_row])


def predict_csv(
    checkpoint: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Path:
    """Predict actions in input order and publish a new CSV without clobbering."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    destination = Path(output_csv)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Prediction output already exists: {destination}")
    source = Path(input_csv)
    if not source.is_file():
        raise FileNotFoundError(f"State CSV not found: {source}")
    selected_device = resolve_device(device)
    fingerprint = sha256_file(checkpoint)
    model, _metadata = load_checkpoint(checkpoint)
    model.to(selected_device)
    source_fingerprint = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with (
            source.open(encoding="utf-8", newline="") as input_file,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary)
            writer.writerow(
                ["state", "action", "action_gap"]
                + [f"q_{action}" for action in range(model.num_actions)]
            )
            reader = csv.reader(input_file, strict=True)
            if next(reader, None) != ["state"]:
                raise ValueError(
                    "State CSV must contain exactly one column named 'state'."
                )
            pending: list[int] = []
            row_count = 0
            for row in reader:
                if len(row) != 1:
                    raise ValueError(
                        f"State CSV row {row_count} must contain one state."
                    )
                try:
                    value = Decimal(row[0])
                except InvalidOperation as exc:
                    raise ValueError(
                        f"State CSV has a non-integer or non-finite state at row {row_count}."
                    ) from exc
                if not value.is_finite() or value != value.to_integral_value():
                    raise ValueError(
                        f"State CSV has a non-integer or non-finite state at row {row_count}."
                    )
                if not 0 <= value < model.num_states:
                    raise ValueError(
                        f"State CSV states must be in 0..{model.num_states - 1}; "
                        f"out-of-range row: {row_count}."
                    )
                pending.append(int(value))
                row_count += 1
                if len(pending) == batch_size:
                    _write_batch(
                        writer,
                        model,
                        pending,
                        start=row_count - len(pending),
                        device=selected_device,
                    )
                    pending.clear()
            if row_count == 0:
                raise ValueError("State CSV must contain at least one row.")
            if pending:
                _write_batch(
                    writer,
                    model,
                    pending,
                    start=row_count - len(pending),
                    device=selected_device,
                )
        if sha256_file(checkpoint) != fingerprint:
            raise ValueError(
                "Checkpoint changed during prediction; retry with a stable file."
            )
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
