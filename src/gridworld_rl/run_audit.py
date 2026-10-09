"""Cross-check the contents of an intact saved experiment run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint
from .config import ExperimentConfig
from .evaluation import policy_agreement_matrix
from .integrity import verify_artifacts


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    """Check config, metrics, predictions, and checkpoints after hash verification."""

    directory = Path(run_dir)
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
    config = ExperimentConfig.from_json(directory / "config.json")
    metrics = _read_object(directory / "metrics.json")
    predictions_doc = _read_object(directory / "predictions.json")
    algorithms = metrics.get("algorithms")
    predictions = predictions_doc.get("predictions")
    if not isinstance(algorithms, dict) or not isinstance(predictions, dict):
        raise ValueError("Run metrics and predictions must contain algorithm objects.")
    expected = set(config.training.algorithms)
    if set(algorithms) != expected:
        issues.append("Metrics algorithm set differs from the run configuration.")
    if set(predictions) != expected:
        issues.append("Predictions algorithm set differs from the run configuration.")
    if metrics.get("seed") != config.training.seed:
        issues.append("Metrics seed differs from the run configuration.")
    dataset = metrics.get("dataset")
    eval_rows = dataset.get("eval_rows") if isinstance(dataset, dict) else None
    if type(eval_rows) is not int or eval_rows <= 0:
        raise ValueError("Run metrics must contain a positive dataset.eval_rows.")

    valid_predictions: dict[str, list[int]] = {}
    for name in config.training.algorithms:
        actions = predictions.get(name)
        if (
            not isinstance(actions, list)
            or len(actions) != eval_rows
            or any(
                type(action) is not int or not 0 <= action < config.dataset.num_actions
                for action in actions
            )
        ):
            issues.append(f"Invalid prediction rows for {name}.")
        else:
            valid_predictions[name] = actions
        details = algorithms.get(name)
        if not isinstance(details, dict):
            issues.append(f"Missing metrics for {name}.")
            continue
        expected_path = f"checkpoints/{name}.pt"
        if details.get("checkpoint") != expected_path:
            issues.append(f"Checkpoint path mismatch for {name}.")
        checkpoint_path = directory / expected_path
        try:
            model, payload = load_checkpoint(checkpoint_path)
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append(f"Invalid checkpoint for {name}: {exc}")
            continue
        if (
            payload["algorithm"] != name
            or payload.get("seed") != config.training.seed
            or payload.get("global_steps") != details.get("global_steps")
            or model.num_states != config.dataset.num_states
            or model.num_actions != config.dataset.num_actions
            or list(payload["network"]["hidden_sizes"])
            != list(config.network.hidden_sizes)
        ):
            issues.append(f"Checkpoint metadata differs from run settings for {name}.")
        evaluation = details.get("evaluation")
        counts = (
            evaluation.get("per_action_prediction_count")
            if isinstance(evaluation, dict)
            else None
        )
        if name in valid_predictions:
            actual_counts = {
                str(action): valid_predictions[name].count(action)
                for action in range(config.dataset.num_actions)
            }
            if counts != actual_counts:
                issues.append(f"Prediction counts disagree with metrics for {name}.")
            if (
                not isinstance(evaluation, dict)
                or evaluation.get("num_examples") != eval_rows
            ):
                issues.append(f"Evaluation row count differs for {name}.")

    if set(valid_predictions) == expected:
        agreement = policy_agreement_matrix(
            valid_predictions, num_actions=config.dataset.num_actions
        )
        if metrics.get("policy_agreement") != agreement:
            issues.append("Policy agreement matrix differs from saved predictions.")
    return {
        "schema_version": 1,
        "valid": not issues,
        "checked_files": verification.checked_files,
        "issues": issues,
    }
