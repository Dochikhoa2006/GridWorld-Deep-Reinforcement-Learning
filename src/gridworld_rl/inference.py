"""Batch inference for a one-column CSV of discrete states."""

from __future__ import annotations

import csv
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .data import load_transition_csv
from .reproducibility import resolve_device, sha256_file
from .supported_policy import select_supported_action


def _write_batch(
    writer,
    model: torch.nn.Module,
    states: list[int],
    *,
    start: int,
    device: torch.device,
    compact: bool = False,
    support: dict[int, list[int]] | None = None,
    action_counts: dict[int, dict[int, int]] | None = None,
    min_action_count: int = 1,
) -> None:
    batch = torch.tensor(states, dtype=torch.long, device=device)
    with torch.inference_mode():
        q_values = model(batch)
        if not torch.isfinite(q_values).all():
            raise ValueError(
                f"Checkpoint produces non-finite Q-values at input rows "
                f"{start}..{start + len(batch) - 1}."
            )
        values = q_values.cpu().tolist() if support is not None or not compact else None
        best = q_values.topk(2, dim=1).values.cpu().double()
        gaps = (best[:, 0] - best[:, 1]).tolist()
        actions = q_values.argmax(dim=1).cpu().tolist()
    for index, (state, action, gap) in enumerate(
        zip(states, actions, gaps, strict=True)
    ):
        if support is None:
            writer.writerow(
                [state, action, gap] + ([] if values is None else values[index])
            )
            continue
        q_row = values[index]
        logged = support.get(state, [])
        counts = action_counts[state] if action_counts is not None and logged else {}
        eligible = [
            candidate for candidate in logged if counts[candidate] >= min_action_count
        ]
        fallback = bool(logged and not eligible)
        if fallback:
            maximum = max(counts.values())
            eligible = [
                candidate for candidate in logged if counts[candidate] == maximum
            ]
        selected, unrestricted = select_supported_action(q_row, eligible)
        if len(eligible) == 1:
            supported_gap = None
        elif eligible:
            ranked = sorted((q_row[candidate] for candidate in eligible), reverse=True)
            supported_gap = ranked[0] - ranked[1]
        else:
            supported_gap = gap
        writer.writerow(
            [
                state,
                selected,
                supported_gap,
                unrestricted,
                gap,
                selected != unrestricted,
                ";".join(str(candidate) for candidate in logged),
                ";".join(str(candidate) for candidate in eligible),
                ";".join(f"{candidate}:{counts[candidate]}" for candidate in logged),
                fallback,
            ]
            + ([] if compact else q_row)
        )


def iter_state_batches(source: Path, num_states: int, batch_size: int):
    """Yield validated states in CSV order without loading the whole file."""

    with source.open(encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file, strict=True)
        if next(reader, None) != ["state"]:
            raise ValueError("State CSV must contain exactly one column named 'state'.")
        pending: list[int] = []
        row_count = 0
        for row in reader:
            if len(row) != 1:
                raise ValueError(f"State CSV row {row_count} must contain one state.")
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
            if not 0 <= value < num_states:
                raise ValueError(
                    f"State CSV states must be in 0..{num_states - 1}; "
                    f"out-of-range row: {row_count}."
                )
            pending.append(int(value))
            row_count += 1
            if len(pending) == batch_size:
                yield row_count - len(pending), pending
                pending = []
        if row_count == 0:
            raise ValueError("State CSV must contain at least one row.")
        if pending:
            yield row_count - len(pending), pending


def predict_csv(
    checkpoint: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
    compact: bool = False,
    train_csv: str | Path | None = None,
    min_action_count: int = 1,
) -> Path:
    """Predict actions in input order and publish a new CSV without clobbering."""

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
    train_path = Path(train_csv) if train_csv is not None else None
    train_fingerprint = sha256_file(train_path) if train_path is not None else None
    support = None
    action_counts = None
    if train_path is not None:
        frame = load_transition_csv(
            train_path, num_states=model.num_states, num_actions=model.num_actions
        )
        action_counts = {}
        for (state, action), count in frame.groupby(["state", "action"]).size().items():
            action_counts.setdefault(int(state), {})[int(action)] = int(count)
        support = {state: sorted(counts) for state, counts in action_counts.items()}
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
            header = (
                ["state", "action", "action_gap"]
                if support is None
                else [
                    "state",
                    "action",
                    "supported_action_gap",
                    "unconstrained_action",
                    "unconstrained_action_gap",
                    "was_constrained",
                    "logged_actions",
                    "eligible_actions",
                    "logged_action_counts",
                    "support_fallback",
                ]
            )
            writer.writerow(
                header
                + (
                    []
                    if compact
                    else [f"q_{action}" for action in range(model.num_actions)]
                )
            )
            for start, pending in iter_state_batches(
                source, model.num_states, batch_size
            ):
                _write_batch(
                    writer,
                    model,
                    pending,
                    start=start,
                    device=selected_device,
                    compact=compact,
                    support=support,
                    action_counts=action_counts,
                    min_action_count=min_action_count,
                )
        if sha256_file(checkpoint) != fingerprint:
            raise ValueError(
                "Checkpoint changed during prediction; retry with a stable file."
            )
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
