from __future__ import annotations

import json
import math

import pytest

from gridworld_rl.benchmark import aggregate_runs, generate_benchmark_report
from gridworld_rl.config import ExperimentConfig


def _runs(tmp_path, scores, seeds):
    paths = []
    for index, seed in enumerate(seeds):
        path = tmp_path / f"run-{index}"
        path.mkdir()
        config = ExperimentConfig()
        config.training.seed = seed
        config.training.algorithms = list(scores)
        config.to_json(path / "config.json")
        metrics = {
            "seed": seed,
            "device": "cpu",
            "runtime": {},
            "source": {},
            "dataset": {"sha256": {"train": "same"}},
            "evaluation_split_diagnostics": {
                "training_majority": {"accuracy": 0.25},
                "training_state_mode_accuracy": 0.5,
                "evaluation_state_mode_ceiling": 0.75,
                "exact_training_overlap_fraction": 0.5,
            },
            "algorithms": {
                name: {
                    "evaluation": {
                        "accuracy": values[index],
                        "per_action_recall": {"0": 0.5},
                    },
                    "training_history": {"total_loss": [1.0]},
                }
                for name, values in scores.items()
            },
        }
        (path / "metrics.json").write_text(json.dumps(metrics))
        paths.append(path)
    return paths


def test_paired_comparisons_preserve_seed_pairing_and_direction(tmp_path):
    seeds = [9, 2, 7]
    # Identical marginal score distributions, different per-seed outcomes.
    scores = {"dqn": [0.25, 0.75, 0.5], "cql": [0.75, 0.25, 0.5]}
    aggregate = aggregate_runs(_runs(tmp_path, scores, seeds), seeds)
    (pair,) = aggregate["paired_comparisons"]
    assert pair["first_algorithm"] == "cql"
    assert pair["second_algorithm"] == "dqn"
    assert pair["per_seed"] == [
        {"seed": 9, "accuracy_difference": 0.5},
        {"seed": 2, "accuracy_difference": -0.5},
        {"seed": 7, "accuracy_difference": 0.0},
    ]
    assert pair["accuracy_difference"] == {
        "mean": 0.0,
        "std": 0.5,
        "minimum": -0.5,
        "maximum": 0.5,
    }
    assert (pair["wins"], pair["ties"], pair["losses"]) == (1, 1, 1)
    generate_benchmark_report(tmp_path, aggregate)
    summary = (tmp_path / "benchmark.md").read_text()
    assert "| CQL | DQN | +0.00 | 50.00 | -50.00 | +50.00 | 1 / 1 / 1 |" in summary
    assert "not significance tests" in summary


def test_every_pair_is_included_once_for_single_seed(tmp_path):
    scores = {"expected_sarsa": [0.25], "dqn": [0.75], "cql": [0.5]}
    aggregate = aggregate_runs(_runs(tmp_path, scores, [1]), [1])
    pairs = aggregate["paired_comparisons"]
    assert [(p["first_algorithm"], p["second_algorithm"]) for p in pairs] == [
        ("cql", "dqn"),
        ("cql", "expected_sarsa"),
        ("dqn", "expected_sarsa"),
    ]
    assert all(p["accuracy_difference"]["std"] == 0 for p in pairs)
    assert pairs[0]["accuracy_difference"]["mean"] == -0.25
    assert pairs[0]["losses"] == 1
    assert pairs[1]["wins"] == 1


def test_single_algorithm_and_legacy_reports(tmp_path):
    aggregate = aggregate_runs(_runs(tmp_path, {"dqn": [0.5]}, [1]), [1])
    assert aggregate["paired_comparisons"] == []
    del aggregate["paired_comparisons"]
    generate_benchmark_report(tmp_path, aggregate)
    assert "Paired seed comparisons" not in (tmp_path / "benchmark.md").read_text()


def test_aggregate_overlap_slices_and_report_nonempty_groups(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.75]}, [1, 2])
    for path, score in zip(paths, [0.5, 0.75], strict=True):
        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        evaluation = metrics["algorithms"]["dqn"]["evaluation"]
        evaluation["num_examples"] = 4
        evaluation["overlap_slices"] = {
            "exact_training_transition": {"rows": 2, "accuracy": score},
            "seen_state_new_transition": {"rows": 2, "accuracy": score},
            "unseen_state": {"rows": 0, "accuracy": None},
        }
        metrics_path.write_text(json.dumps(metrics))
    aggregate = aggregate_runs(paths, [1, 2])
    slices = aggregate["algorithms"]["dqn"]["overlap_slices"]
    assert slices["exact_training_transition"]["accuracy"]["mean"] == 0.625
    assert slices["exact_training_transition"]["accuracy"]["std"] == pytest.approx(
        0.25 / 2**0.5
    )
    assert slices["unseen_state"] == {"rows": 0, "accuracy": None}
    generate_benchmark_report(tmp_path, aggregate)
    report = (tmp_path / "benchmark.md").read_text()
    assert "| DQN | Exact training transition | 2 | 62.50% |" in report
    assert "| DQN | Unseen state | 0 | N/A | N/A |" in report


@pytest.mark.parametrize("corruption", ["missing", "changed_rows", "invalid_accuracy"])
def test_rejects_inconsistent_overlap_slices(tmp_path, corruption):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5]}, [1, 2])
    for path in paths:
        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        evaluation = metrics["algorithms"]["dqn"]["evaluation"]
        evaluation["num_examples"] = 2
        evaluation["overlap_slices"] = {
            "exact_training_transition": {"rows": 2, "accuracy": 0.5},
            "seen_state_new_transition": {"rows": 0, "accuracy": None},
            "unseen_state": {"rows": 0, "accuracy": None},
        }
        metrics_path.write_text(json.dumps(metrics))
    second_path = paths[1] / "metrics.json"
    metrics = json.loads(second_path.read_text())
    evaluation = metrics["algorithms"]["dqn"]["evaluation"]
    if corruption == "missing":
        del evaluation["overlap_slices"]
    elif corruption == "changed_rows":
        evaluation["overlap_slices"]["exact_training_transition"]["rows"] = 1
    else:
        evaluation["overlap_slices"]["exact_training_transition"]["accuracy"] = 1.5
    second_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="overlap slice"):
        aggregate_runs(paths, [1, 2])


@pytest.mark.parametrize("seed", [5, True, "1", None])
def test_rejects_mislabeled_metric_seeds(tmp_path, seed):
    paths = _runs(tmp_path, {"dqn": [0.5]}, [1])
    path = paths[0] / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["seed"] = seed
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="metrics seeds"):
        aggregate_runs(paths, [1])


@pytest.mark.parametrize("seeds", [[1, 1], [True, 2], [-1, 2], [1.5, 2]])
def test_rejects_invalid_seed_schedule(tmp_path, seeds):
    with pytest.raises(ValueError, match="distinct non-negative integers"):
        aggregate_runs([tmp_path, tmp_path], seeds)


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, True, "0.5", None])
def test_rejects_invalid_accuracy(tmp_path, score):
    paths = _runs(tmp_path, {"dqn": [score], "cql": [0.5]}, [1])
    with pytest.raises(ValueError, match="accuracy must be finite"):
        aggregate_runs(paths, [1])
