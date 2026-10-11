"""Publish and semantically audit retained ensemble evaluation reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .ensemble_evaluation import evaluate_ensemble_csv
from .reproducibility import sha256_file


def save_ensemble_evaluation(
    output: str | Path,
    predictions_csv: str | Path,
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    train_csv: str | Path | None = None,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    target_coverage: float | None = None,
) -> Path:
    """Write a new self-describing report without replacing an existing file."""

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Ensemble evaluation report already exists: {destination}"
        )
    result = evaluate_ensemble_csv(
        Path(predictions_csv).resolve(),
        Path(challenge_csv).resolve(),
        Path(solution_csv).resolve(),
        train_csv=Path(train_csv).resolve() if train_csv is not None else None,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        target_coverage=target_coverage,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(result, temporary, sort_keys=True, indent=2, allow_nan=False)
            temporary.write("\n")
        for name, details in result["inputs"].items():
            if sha256_file(details["path"]) != details["sha256"]:
                raise ValueError(
                    f"{name.capitalize()} changed before report publication."
                )
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _first_difference(saved: Any, expected: Any, path: str = "report") -> str | None:
    if type(saved) is not type(expected):
        return path
    if isinstance(expected, dict):
        if saved.keys() != expected.keys():
            return path
        for key in expected:
            difference = _first_difference(saved[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(saved) != len(expected):
            return path
        for index, (left, right) in enumerate(zip(saved, expected, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return path if saved != expected else None


def audit_ensemble_evaluation_report(report: str | Path) -> dict[str, Any]:
    """Recompute a saved report and reject changed inputs or metrics."""

    report_path = Path(report)
    report_hash = sha256_file(report_path)
    try:
        with report_path.open(encoding="utf-8") as file:
            saved = json.load(file)
        if not isinstance(saved, dict) or saved.get("schema_version") != 1:
            raise ValueError("Invalid ensemble evaluation report schema.")
        inputs = saved["inputs"]
        bootstrap = saved.get("bootstrap", {})
        recommendation = saved.get("recommended_threshold", {})
        expected = evaluate_ensemble_csv(
            inputs["predictions"]["path"],
            inputs["challenge"]["path"],
            inputs["solution"]["path"],
            train_csv=inputs["train"]["path"] if "train" in inputs else None,
            bootstrap_replicates=bootstrap.get("replicates", 0),
            bootstrap_seed=bootstrap.get("seed", 0),
            confidence_level=bootstrap.get("confidence_level", 0.95),
            target_coverage=recommendation.get("target_coverage"),
        )
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid ensemble evaluation report structure.") from exc
    difference = _first_difference(saved, expected)
    if difference is not None:
        raise ValueError(f"Ensemble evaluation report mismatch at {difference}.")
    if sha256_file(report_path) != report_hash:
        raise ValueError("Ensemble evaluation report changed during audit.")
    return {
        "schema_version": 1,
        "valid": True,
        "report": {"path": str(report_path), "sha256": report_hash},
        "verified_rows": expected["num_rows"],
        "inputs": expected["inputs"],
    }
