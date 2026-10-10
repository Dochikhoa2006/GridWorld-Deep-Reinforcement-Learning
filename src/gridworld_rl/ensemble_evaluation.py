"""Evaluate a saved ensemble prediction CSV on aligned labeled rows."""

from __future__ import annotations

import csv
import math
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_evaluation_data
from .evaluation import classification_metrics
from .reproducibility import sha256_file
from .uncertainty import validate_bootstrap_options


def _action(value: str, num_actions: int, row: int, field: str) -> int:
    if not value.isdecimal():
        raise ValueError(f"Prediction row {row} has invalid {field}.")
    action = int(value)
    if action >= num_actions:
        raise ValueError(f"Prediction row {row} has out-of-range {field}.")
    return action


def _fraction(value: str, row: int, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Prediction row {row} has invalid {field}.") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"Prediction row {row} has invalid {field}.")
    return result


def _agreement_curve(
    agreements: list[float], predictions: np.ndarray, targets: np.ndarray
) -> list[dict[str, float | int]]:
    """List every attainable coverage point in descending agreement order."""

    ordered = sorted(zip(agreements, predictions == targets, strict=True), reverse=True)
    accepted = 0
    correct = 0
    curve = []
    for threshold, rows in groupby(ordered, key=lambda item: item[0]):
        for _, is_correct in rows:
            accepted += 1
            correct += int(is_correct)
        curve.append(
            {
                "threshold": threshold,
                "accepted_rows": accepted,
                "coverage": accepted / len(targets),
                "selective_accuracy": correct / accepted,
            }
        )
    return curve


def _bootstrap_intervals(
    states: np.ndarray,
    accepted: np.ndarray,
    correct: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Resample state clusters so duplicate states stay together."""

    unique_states, inverse = np.unique(states, return_inverse=True)
    cluster_count = len(unique_states)
    sizes = np.bincount(inverse)
    accepted_counts = np.bincount(inverse, weights=accepted.astype(np.int64))
    accepted_correct = np.bincount(
        inverse, weights=(accepted & correct).astype(np.int64)
    )
    correct_counts = np.bincount(inverse, weights=correct.astype(np.int64))
    coverage_draws = np.empty(replicates)
    suggested_draws = np.empty(replicates)
    selective_draws = []
    generator = np.random.default_rng(seed)
    for index in range(replicates):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        total = sizes[sampled].sum()
        accepted_total = accepted_counts[sampled].sum()
        coverage_draws[index] = accepted_total / total
        suggested_draws[index] = correct_counts[sampled].sum() / total
        if accepted_total:
            selective_draws.append(accepted_correct[sampled].sum() / accepted_total)
    tail = (1 - confidence_level) / 2

    def interval(estimate, draws):
        return (
            None
            if not len(draws)
            else {
                "estimate": estimate,
                "lower": float(np.quantile(draws, tail)),
                "upper": float(np.quantile(draws, 1 - tail)),
            }
        )

    accepted_total = int(accepted.sum())
    return {
        "method": "percentile state-cluster bootstrap",
        "resampling_unit": "state",
        "unique_states": cluster_count,
        "replicates": replicates,
        "selective_replicates": len(selective_draws),
        "seed": seed,
        "confidence_level": confidence_level,
        "coverage_interval": interval(accepted_total / len(states), coverage_draws),
        "suggested_accuracy_interval": interval(float(correct.mean()), suggested_draws),
        "selective_accuracy_interval": interval(
            float(correct[accepted].mean()) if accepted_total else None,
            selective_draws,
        ),
    }


def evaluate_ensemble_csv(
    predictions_csv: str | Path,
    challenge_csv: str | Path,
    solution_csv: str | Path,
    *,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Report coverage and accuracy on accepted rows, checking row alignment."""

    validate_bootstrap_options(bootstrap_replicates, bootstrap_seed, confidence_level)
    paths = {
        "predictions": Path(predictions_csv),
        "challenge": Path(challenge_csv),
        "solution": Path(solution_csv),
    }
    fingerprints = {name: sha256_file(path) for name, path in paths.items()}
    with paths["predictions"].open(encoding="utf-8", newline="") as file:
        header = next(csv.reader(file, strict=True), None)
    if header is None or len(header) != len(set(header)):
        raise ValueError("Invalid ensemble prediction CSV header.")
    vote_columns = sorted(
        (
            column
            for column in header
            if column.startswith("votes_") and column[6:].isdecimal()
        ),
        key=lambda column: int(column[6:]),
    )
    num_actions = len(vote_columns)
    if num_actions < 2 or vote_columns != [
        f"votes_{action}" for action in range(num_actions)
    ]:
        raise ValueError(
            "Ensemble prediction CSV needs contiguous votes_<action> columns."
        )
    required = {
        "state",
        "action",
        "votes",
        "agreement_fraction",
        "unanimous",
        "member_actions",
        *vote_columns,
    }
    if not required.issubset(header):
        raise ValueError("Ensemble prediction CSV is missing required columns.")
    weight_columns = [f"weight_{action}" for action in range(num_actions)]
    weighted = all(column in header for column in weight_columns)
    if any(column.startswith("weight_") for column in header) and not weighted:
        raise ValueError("Ensemble prediction CSV has incomplete weight columns.")
    if weighted and not {"vote_weight", "agreement_weight_fraction"}.issubset(header):
        raise ValueError(
            "Ensemble prediction CSV is missing weighted agreement columns."
        )
    abstention_columns = {"suggested_action", "abstained"}
    if bool(abstention_columns & set(header)) != abstention_columns.issubset(header):
        raise ValueError("Ensemble prediction CSV has incomplete abstention columns.")
    challenge, solution = load_evaluation_data(
        paths["challenge"], paths["solution"], num_actions=num_actions
    )
    states = challenge["state"].to_numpy(dtype=np.int64)
    targets = solution["action"].to_numpy(dtype=np.int64)
    accepted: list[bool] = []
    predictions: list[int] = []
    agreements: list[float] = []
    decision_agreements: list[float] = []
    with paths["predictions"].open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        for index, row in enumerate(reader):
            if index >= len(states):
                raise ValueError(
                    "Ensemble predictions have more rows than the challenge."
                )
            if row["state"] != str(states[index]):
                raise ValueError(f"Ensemble prediction state mismatch at row {index}.")
            members = row["member_actions"].split(";")
            actions = [
                _action(value, num_actions, index, "member_actions")
                for value in members
            ]
            votes = [
                _action(row[column], len(actions) + 1, index, column)
                for column in vote_columns
            ]
            if votes != [actions.count(action) for action in range(num_actions)]:
                raise ValueError(f"Ensemble vote counts mismatch at row {index}.")
            if weighted:
                try:
                    weight_totals = [float(row[column]) for column in weight_columns]
                except ValueError as exc:
                    raise ValueError(f"Invalid weighted votes at row {index}.") from exc
                if any(
                    not math.isfinite(value) or value < 0 for value in weight_totals
                ):
                    raise ValueError(f"Invalid weighted votes at row {index}.")
                if sum(weight_totals) <= 0:
                    raise ValueError(f"Invalid weighted votes at row {index}.")
                ranking = weight_totals
            else:
                ranking = votes
            winner = _action(
                row["suggested_action"]
                if abstention_columns.issubset(header)
                else row["action"],
                num_actions,
                index,
                "suggested_action",
            )
            if winner != max(range(num_actions), key=ranking.__getitem__):
                raise ValueError(f"Ensemble winner mismatch at row {index}.")
            if row["votes"] != str(votes[winner]):
                raise ValueError(f"Winning vote count mismatch at row {index}.")
            if weighted:
                try:
                    winner_weight = float(row["vote_weight"])
                except ValueError as exc:
                    raise ValueError(
                        f"Winning vote weight mismatch at row {index}."
                    ) from exc
                weight_share = _fraction(
                    row["agreement_weight_fraction"], index, "agreement_weight_fraction"
                )
                if not math.isclose(
                    winner_weight, ranking[winner], rel_tol=1e-12
                ) or not math.isclose(
                    weight_share, ranking[winner] / sum(ranking), abs_tol=1e-12
                ):
                    raise ValueError(f"Weighted agreement mismatch at row {index}.")
            agreement = _fraction(
                row["agreement_fraction"], index, "agreement_fraction"
            )
            if not math.isclose(agreement, votes[winner] / len(actions), abs_tol=1e-12):
                raise ValueError(f"Agreement fraction mismatch at row {index}.")
            if row["unanimous"] != str(votes[winner] == len(actions)):
                raise ValueError(f"Unanimity mismatch at row {index}.")
            abstained = row["action"] == ""
            if abstention_columns.issubset(header):
                if row["abstained"] != str(abstained):
                    raise ValueError(f"Abstention mismatch at row {index}.")
            elif abstained:
                raise ValueError(
                    f"Prediction row {index} has a blank action without abstention columns."
                )
            if (
                not abstained
                and _action(row["action"], num_actions, index, "action") != winner
            ):
                raise ValueError(f"Selected action mismatch at row {index}.")
            accepted.append(not abstained)
            predictions.append(winner)
            agreements.append(agreement)
            decision_agreements.append(weight_share if weighted else agreement)
    if len(accepted) != len(states):
        raise ValueError("Ensemble predictions have fewer rows than the challenge.")
    accepted_mask = np.asarray(accepted, dtype=bool)
    predicted = np.asarray(predictions, dtype=np.int64)
    accepted_count = int(accepted_mask.sum())
    agreement_curve = _agreement_curve(decision_agreements, predicted, targets)
    previous_coverage = 0.0
    area_under_risk_coverage_curve = 0.0
    for point in agreement_curve:
        area_under_risk_coverage_curve += (point["coverage"] - previous_coverage) * (
            1 - point["selective_accuracy"]
        )
        previous_coverage = point["coverage"]
    result = {
        "schema_version": 1,
        "inputs": {
            name: {"path": str(path), "sha256": fingerprints[name]}
            for name, path in paths.items()
        },
        "num_actions": num_actions,
        "num_rows": len(states),
        "accepted_rows": accepted_count,
        "abstained_rows": len(states) - accepted_count,
        "coverage": accepted_count / len(states),
        "selective_accuracy": float(
            (predicted[accepted_mask] == targets[accepted_mask]).mean()
        )
        if accepted_count
        else None,
        "suggested_accuracy": float((predicted == targets).mean()),
        "agreement_basis": "weight_share" if weighted else "member_share",
        "agreement_curve": agreement_curve,
        "area_under_risk_coverage_curve": area_under_risk_coverage_curve,
        "mean_agreement_accepted": float(np.mean(np.asarray(agreements)[accepted_mask]))
        if accepted_count
        else None,
        "mean_agreement_abstained": float(
            np.mean(np.asarray(agreements)[~accepted_mask])
        )
        if accepted_count < len(states)
        else None,
        "accepted_metrics": classification_metrics(
            targets[accepted_mask], predicted[accepted_mask], num_actions=num_actions
        )
        if accepted_count
        else None,
    }
    if bootstrap_replicates:
        result["bootstrap"] = _bootstrap_intervals(
            states,
            accepted_mask,
            predicted == targets,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=float(confidence_level),
        )
    for name, path in paths.items():
        if sha256_file(path) != fingerprints[name]:
            raise ValueError(
                f"{name.capitalize()} changed during ensemble evaluation: {path}."
            )
    return result
