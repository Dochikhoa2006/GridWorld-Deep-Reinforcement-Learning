"""Batch inference for a one-column CSV of discrete states."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import load_checkpoint
from .reproducibility import resolve_device, sha256_file


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
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f"Invalid state CSV {source}: {exc}") from exc
    if frame.columns.tolist() != ["state"]:
        raise ValueError("State CSV must contain exactly one column named 'state'.")
    if frame.empty:
        raise ValueError("State CSV must contain at least one row.")
    numeric = pd.to_numeric(frame["state"], errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(numeric) | (numeric != np.floor(numeric))
    if invalid.any():
        rows = np.flatnonzero(invalid)[:5].tolist()
        raise ValueError(
            f"State CSV has non-integer or non-finite states at rows {rows}."
        )

    selected_device = resolve_device(device)
    fingerprint = sha256_file(checkpoint)
    model, _metadata = load_checkpoint(checkpoint)
    out_of_range = (numeric < 0) | (numeric >= model.num_states)
    if out_of_range.any():
        rows = np.flatnonzero(out_of_range)[:5].tolist()
        raise ValueError(
            f"State CSV states must be in 0..{model.num_states - 1}; "
            f"out-of-range rows: {rows}."
        )
    states = numeric.astype(np.int64)
    model.to(selected_device)
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
                ["state", "action", "action_gap"]
                + [f"q_{action}" for action in range(model.num_actions)]
            )
            with torch.inference_mode():
                for start in range(0, len(states), batch_size):
                    batch = torch.tensor(
                        states[start : start + batch_size],
                        dtype=torch.long,
                        device=selected_device,
                    )
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
                        states[start : start + batch_size],
                        actions,
                        gaps,
                        values,
                        strict=True,
                    ):
                        writer.writerow([int(state), action, gap, *values_row])
        if sha256_file(checkpoint) != fingerprint:
            raise ValueError(
                "Checkpoint changed during prediction; retry with a stable file."
            )
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
