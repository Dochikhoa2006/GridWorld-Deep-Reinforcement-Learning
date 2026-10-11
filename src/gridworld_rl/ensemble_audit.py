"""Replay saved ensemble predictions against their declared inputs."""

from __future__ import annotations

import csv
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .ensemble_inference import predict_ensemble_csv
from .reproducibility import sha256_file


def audit_ensemble_predictions(
    checkpoints: list[str | Path],
    input_csv: str | Path,
    predictions_csv: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
    train_csv: str | Path | None = None,
    min_action_count: int = 1,
    weights: list[float] | None = None,
    abstain_below: float | None = None,
) -> dict[str, Any]:
    """Return input hashes only when the saved CSV matches a full replay."""

    paths = [Path(path) for path in checkpoints]
    source = Path(input_csv)
    saved = Path(predictions_csv)
    train = Path(train_csv) if train_csv is not None else None
    inputs = {"input": source, "predictions": saved}
    if train is not None:
        inputs["train"] = train
    fingerprints = {name: sha256_file(path) for name, path in inputs.items()}
    checkpoint_hashes = [sha256_file(path) for path in paths]
    with tempfile.TemporaryDirectory(prefix="gridworld-ensemble-audit-") as directory:
        replay = Path(directory) / "replayed.csv"
        predict_ensemble_csv(
            paths,
            source,
            replay,
            device=device,
            batch_size=batch_size,
            train_csv=train,
            min_action_count=min_action_count,
            weights=weights,
            abstain_below=abstain_below,
        )
        try:
            with (
                replay.open(encoding="utf-8", newline="") as expected_file,
                saved.open(encoding="utf-8", newline="") as saved_file,
            ):
                expected_rows = csv.reader(expected_file, strict=True)
                saved_rows = csv.reader(saved_file, strict=True)
                expected_header = next(expected_rows)
                if next(saved_rows, None) != expected_header:
                    raise ValueError(
                        "Saved ensemble prediction CSV header does not match replay."
                    )
                row_count = 0
                for index, (expected, actual) in enumerate(
                    zip_longest(expected_rows, saved_rows)
                ):
                    if expected is None or actual is None:
                        raise ValueError(
                            f"Saved ensemble prediction row count differs at row {index}."
                        )
                    if expected != actual:
                        column = next(
                            (
                                name
                                for name, left, right in zip(
                                    expected_header, expected, actual, strict=False
                                )
                                if left != right
                            ),
                            "column count",
                        )
                        raise ValueError(
                            f"Saved ensemble prediction mismatch at row {index}, column {column}."
                        )
                    row_count += 1
        except (UnicodeError, csv.Error) as exc:
            raise ValueError(
                f"Invalid saved ensemble prediction CSV: {saved}."
            ) from exc
    for name, path in inputs.items():
        if sha256_file(path) != fingerprints[name]:
            raise ValueError(
                f"{name.capitalize()} changed during ensemble audit: {path}."
            )
    for path, fingerprint in zip(paths, checkpoint_hashes, strict=True):
        if sha256_file(path) != fingerprint:
            raise ValueError(f"Checkpoint changed during ensemble audit: {path}.")
    return {
        "schema_version": 1,
        "verified_rows": row_count,
        "matched": True,
        "inputs": {
            name: {"path": str(path), "sha256": fingerprints[name]}
            for name, path in inputs.items()
        },
        "checkpoints": [
            {"path": str(path), "sha256": fingerprint}
            for path, fingerprint in zip(paths, checkpoint_hashes, strict=True)
        ],
        "settings": {
            "weights": weights,
            "min_action_count": min_action_count,
            "abstain_below": abstain_below,
        },
    }
