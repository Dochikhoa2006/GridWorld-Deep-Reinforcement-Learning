"""Cross-check the contents of an intact saved experiment run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .config import ExperimentConfig
from .data import (
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
    transition_diagnostics,
)
from .evaluation import (
    classification_metrics,
    overlap_sliced_agreement,
    policy_agreement_matrix,
    predict_actions,
)
from .integrity import verify_artifacts
from .reproducibility import sha256_file
from .support import policy_support_metrics


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def audit_run(
    run_dir: str | Path, *, data_dir: str | Path | None = None
) -> dict[str, Any]:
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

    loaded_data = None
    data_paths: dict[str, Path] = {}
    data_hashes: dict[str, str] = {}
    if data_dir is not None:
        root = Path(data_dir)
        data_paths = {
            "train": root / "train.csv",
            "eval_challenge": root / "eval_challenge.csv",
            "eval_solution": root / "eval_solution.csv",
        }
        data_hashes = {name: sha256_file(path) for name, path in data_paths.items()}
        saved_hashes = dataset.get("sha256")
        if not isinstance(saved_hashes, dict):
            raise ValueError("Run metrics must contain dataset.sha256 for data audit.")
        mismatched_data = []
        for name, fingerprint in data_hashes.items():
            if saved_hashes.get(name) != fingerprint:
                mismatched_data.append(name)
                issues.append(f"Dataset fingerprint differs for {name}.")
        if mismatched_data:
            return {
                "schema_version": 1,
                "valid": False,
                "checked_files": verification.checked_files,
                "issues": issues,
            }
        train = load_transition_csv(
            data_paths["train"],
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        )
        challenge, solution = load_evaluation_data(
            data_paths["eval_challenge"],
            data_paths["eval_solution"],
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        )
        loaded_data = (train, challenge, solution)
        if len(train) != dataset.get("train_rows") or len(challenge) != eval_rows:
            issues.append("Dataset row counts differ from saved metrics.")
        current_diagnostics = transition_diagnostics(
            train,
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        )
        saved_diagnostics = dataset.get("diagnostics")
        if not isinstance(saved_diagnostics, dict) or any(
            current_diagnostics.get(key) != value
            for key, value in saved_diagnostics.items()
        ):
            issues.append("Training diagnostics differ from saved metrics.")
        if evaluation_split_diagnostics(
            train,
            solution,
            num_states=config.dataset.num_states,
            num_actions=config.dataset.num_actions,
        ) != metrics.get("evaluation_split_diagnostics"):
            issues.append("Evaluation split diagnostics differ from saved metrics.")

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
        if loaded_data is not None:
            train, challenge, solution = loaded_data
            recomputed = predict_actions(
                model,
                challenge["state"].to_numpy(),
                device=torch.device("cpu"),
            )
            if (
                name in valid_predictions
                and recomputed.tolist() != valid_predictions[name]
            ):
                issues.append(
                    f"Saved predictions differ from checkpoint inference for {name}."
                )
            targets = solution["action"].to_numpy()
            expected_evaluation = classification_metrics(
                targets, recomputed, num_actions=config.dataset.num_actions
            )
            expected_evaluation["overlap_slices"] = overlap_sliced_agreement(
                train, solution, targets, recomputed
            )
            if evaluation != expected_evaluation:
                issues.append(
                    f"Evaluation metrics differ from checkpoint inference for {name}."
                )
            expected_support = policy_support_metrics(
                model, train, device=torch.device("cpu")
            )
            if details.get("policy_support") != expected_support:
                issues.append(
                    f"Policy support differs from checkpoint inference for {name}."
                )

    if set(valid_predictions) == expected:
        agreement = policy_agreement_matrix(
            valid_predictions, num_actions=config.dataset.num_actions
        )
        if metrics.get("policy_agreement") != agreement:
            issues.append("Policy agreement matrix differs from saved predictions.")
    for name, path in data_paths.items():
        if sha256_file(path) != data_hashes[name]:
            raise ValueError(f"Dataset file changed during audit: {path}.")
    return {
        "schema_version": 1,
        "valid": not issues,
        "checked_files": verification.checked_files,
        "issues": issues,
    }
