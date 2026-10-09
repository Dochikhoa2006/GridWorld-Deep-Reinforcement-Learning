"""Integrity-checked comparison of two saved experiment runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .integrity import verify_artifacts

SCORES = ("accuracy", "balanced_accuracy", "macro_f1")
DATASET_KEYS = ("train", "eval_challenge", "eval_solution")


def _read_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_artifacts(directory)
    if not verification.valid:
        raise ValueError(
            f"Run failed integrity verification: {directory}: "
            f"missing={verification.missing}, modified={verification.modified}, "
            f"unexpected={verification.unexpected}."
        )
    config = ExperimentConfig.from_json(directory / "config.json").to_dict()
    try:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid run metrics in {directory}: {exc}") from exc
    if not isinstance(metrics, dict) or metrics.get("schema_version") != 1:
        raise ValueError(f"Unsupported run metrics in {directory}.")
    dataset = metrics.get("dataset")
    hashes = dataset.get("sha256") if isinstance(dataset, dict) else None
    if not isinstance(hashes, dict) or any(
        not isinstance(hashes.get(key), str) or len(hashes[key]) != 64
        for key in DATASET_KEYS
    ):
        raise ValueError(f"Run has missing dataset hashes: {directory}.")
    algorithms = metrics.get("algorithms")
    if not isinstance(algorithms, dict) or set(algorithms) != set(
        config["training"]["algorithms"]
    ):
        raise ValueError(f"Run algorithms disagree with configuration: {directory}.")
    if metrics.get("seed") != config["training"]["seed"]:
        raise ValueError(f"Run seed disagrees with configuration: {directory}.")
    for name, payload in algorithms.items():
        evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
        if not isinstance(evaluation, dict):
            raise ValueError(f"Missing evaluation for {name} in {directory}.")
        for score in SCORES:
            value = evaluation.get(score)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid {score} for {name} in {directory}.")
    return config, metrics


def compare_runs(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare two intact runs and identify whether scores share evaluation data."""

    paths = [Path(left), Path(right)]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("Run directories must be distinct.")
    (left_config, left_metrics), (right_config, right_metrics) = (
        _read_run(path) for path in paths
    )
    left_hashes = left_metrics["dataset"]["sha256"]
    right_hashes = right_metrics["dataset"]["sha256"]
    dataset_match = {key: left_hashes[key] == right_hashes[key] for key in DATASET_KEYS}
    dimensions_match = all(
        left_config["dataset"][key] == right_config["dataset"][key]
        for key in ("num_states", "num_actions")
    )
    same_evaluation = (
        dimensions_match
        and dataset_match["eval_challenge"]
        and dataset_match["eval_solution"]
    )
    settings = {
        "network": (left_config["network"], right_config["network"]),
        "dataset_dimensions": (
            {key: left_config["dataset"][key] for key in ("num_states", "num_actions")},
            {
                key: right_config["dataset"][key]
                for key in ("num_states", "num_actions")
            },
        ),
        "training": (
            {
                key: value
                for key, value in left_config["training"].items()
                if key not in {"algorithms", "seed"}
            },
            {
                key: value
                for key, value in right_config["training"].items()
                if key not in {"algorithms", "seed"}
            },
        ),
    }
    differences = {
        section: {"left": values[0], "right": values[1]}
        for section, values in settings.items()
        if values[0] != values[1]
    }
    common = sorted(set(left_metrics["algorithms"]) & set(right_metrics["algorithms"]))
    result = {
        "schema_version": 1,
        "runs": [
            {
                "path": str(path),
                "seed": metrics["seed"],
                "algorithms": sorted(metrics["algorithms"]),
            }
            for path, metrics in zip(paths, (left_metrics, right_metrics), strict=True)
        ],
        "dataset_match": dataset_match,
        "same_evaluation_split": same_evaluation,
        "settings_differences": differences,
        "common_algorithms": common,
        "left_only_algorithms": sorted(set(left_metrics["algorithms"]) - set(common)),
        "right_only_algorithms": sorted(set(right_metrics["algorithms"]) - set(common)),
        "metric_deltas": {},
    }
    if same_evaluation:
        result["metric_deltas"] = {
            name: {
                score: {
                    "left": left_metrics["algorithms"][name]["evaluation"][score],
                    "right": right_metrics["algorithms"][name]["evaluation"][score],
                    "right_minus_left": (
                        right_metrics["algorithms"][name]["evaluation"][score]
                        - left_metrics["algorithms"][name]["evaluation"][score]
                    ),
                }
                for score in SCORES
            }
            for name in common
        }
    return result
