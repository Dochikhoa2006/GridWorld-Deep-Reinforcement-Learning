"""Semantic verification of a saved multi-seed benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import aggregate_runs
from .config import ExperimentConfig
from .integrity import verify_artifacts
from .run_audit import audit_run


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def audit_benchmark(benchmark_dir: str | Path) -> dict[str, Any]:
    """Verify nested run semantics and reproduce the aggregate from saved runs."""

    directory = Path(benchmark_dir)
    verification = verify_artifacts(directory)
    issues: list[str] = []
    if not verification.valid:
        issues.extend(
            f"{category}: {name}"
            for category in ("missing", "modified", "unexpected")
            for name in getattr(verification, category)
        )
        return {
            "schema_version": 1,
            "valid": False,
            "checked_files": verification.checked_files,
            "issues": issues,
        }
    settings = _read_object(directory / "benchmark_config.json")
    saved = _read_object(directory / "aggregate_metrics.json")
    benchmark = settings.get("benchmark")
    experiment = settings.get("experiment")
    if not isinstance(benchmark, dict) or not isinstance(experiment, dict):
        raise ValueError(
            "Benchmark configuration must contain experiment and benchmark objects."
        )
    seeds = benchmark.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Benchmark seeds must be distinct non-negative integers.")
    declared_config = ExperimentConfig.from_dict(experiment).to_dict()
    expected_paths = [f"runs/seed-{seed}" for seed in seeds]
    if saved.get("run_directories") != expected_paths:
        issues.append("Aggregate run directories differ from the declared seeds.")
    if saved.get("seeds") != seeds:
        issues.append("Aggregate seeds differ from benchmark configuration.")
    run_dirs = [directory / path for path in expected_paths]
    for seed, run_dir in zip(seeds, run_dirs, strict=True):
        result = audit_run(run_dir)
        issues.extend(f"seed-{seed}: {issue}" for issue in result["issues"])
        if not result["valid"]:
            continue
        run_config = ExperimentConfig.from_json(run_dir / "config.json").to_dict()
        for section in ("dataset", "network"):
            if run_config[section] != declared_config[section]:
                issues.append(
                    f"seed-{seed}: {section} differs from benchmark configuration."
                )
        declared_training = {
            key: value
            for key, value in declared_config["training"].items()
            if key != "seed"
        }
        actual_training = {
            key: value for key, value in run_config["training"].items() if key != "seed"
        }
        if (
            actual_training != declared_training
            or run_config["training"]["seed"] != seed
        ):
            issues.append(
                f"seed-{seed}: training differs from benchmark configuration."
            )
    if not issues:
        recomputed = aggregate_runs(run_dirs, seeds)
        recomputed["run_directories"] = expected_paths
        for key in sorted(set(saved) | set(recomputed)):
            if saved.get(key) != recomputed.get(key):
                issues.append(f"Aggregate field differs from saved runs: {key}.")
    return {
        "schema_version": 1,
        "valid": not issues,
        "checked_files": verification.checked_files,
        "seeds": seeds,
        "issues": issues,
    }
