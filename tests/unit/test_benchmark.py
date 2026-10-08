from __future__ import annotations

import json
import math

import pytest

from gridworld_rl.benchmark import aggregate_runs, generate_benchmark_report
from gridworld_rl.config import ExperimentConfig
from gridworld_rl.reproducibility import sha256_file


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


def test_policy_support_aggregates_rates_and_rejects_inconsistent_runs(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.75]}, [1, 2])
    for path, supported, row_rate in zip(paths, [2, 3], [0.4, 0.8], strict=True):
        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["algorithms"]["dqn"]["policy_support"] = {
            "training_rows": 10,
            "observed_states": 4,
            "supported_observed_states": supported,
            "observed_state_support_rate": supported / 4,
            "logged_row_support_rate": row_rate,
        }
        metrics_path.write_text(json.dumps(metrics))
    aggregate = aggregate_runs(paths, [1, 2])
    support = aggregate["algorithms"]["dqn"]["policy_support"]
    assert support["observed_states"] == 4
    assert support["observed_state_support_rate"]["mean"] == 0.625
    assert support["logged_row_support_rate"]["mean"] == pytest.approx(0.6)
    generate_benchmark_report(tmp_path, aggregate)
    assert (
        "Logged action support across seeds" in (tmp_path / "benchmark.md").read_text()
    )

    metrics_path = paths[1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["algorithms"]["dqn"]["policy_support"]["observed_state_support_rate"] = 1.0
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Invalid benchmark policy support"):
        aggregate_runs(paths, [1, 2])

    del metrics["algorithms"]["dqn"]["policy_support"]
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="policy support is inconsistent"):
        aggregate_runs(paths, [1, 2])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("recall", float("nan"), "per-action recall"),
        ("recall", 1.2, "per-action recall"),
        ("recall", True, "per-action recall"),
        ("objective", float("inf"), "final training objective"),
        ("objective", True, "final training objective"),
        ("objective", [], "final training objective"),
    ],
)
def test_rejects_invalid_recall_and_training_objective(tmp_path, field, value, message):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5]}, [1, 2])
    metrics_path = paths[1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    algorithm = metrics["algorithms"]["dqn"]
    if field == "recall":
        algorithm["evaluation"]["per_action_recall"]["0"] = value
    else:
        algorithm["training_history"]["total_loss"] = (
            value if isinstance(value, list) else [value]
        )
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match=message):
        aggregate_runs(paths, [1, 2])


def test_benchmark_refuses_mixed_manifest_presence(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5]}, [1, 2])
    hashes = {
        name: sha256_file(paths[0] / name) for name in ("config.json", "metrics.json")
    }
    (paths[0] / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    with pytest.raises(ValueError, match="inconsistent integrity manifests"):
        aggregate_runs(paths, [1, 2])


def test_action_stability_uses_aligned_rows_and_seed_pairs(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5, 0.5]}, [9, 2, 7])
    for path, values in zip(
        paths, [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 2, 3]], strict=True
    ):
        (path / "predictions.json").write_text(
            json.dumps({"predictions": {"dqn": values}})
        )
    aggregate = aggregate_runs(paths, [9, 2, 7])
    stability = aggregate["action_stability"]["dqn"]
    assert stability["evaluation_rows"] == 4
    assert stability["seed_pairs"] == [
        {"first_seed": 9, "second_seed": 2, "agreement": 0.5},
        {"first_seed": 9, "second_seed": 7, "agreement": 0.75},
        {"first_seed": 2, "second_seed": 7, "agreement": 0.25},
    ]
    assert stability["pairwise_agreement"]["mean"] == 0.5
    generate_benchmark_report(tmp_path, aggregate)
    assert (
        "| DQN | 4 | 3 | 50.00% | 25.00% |" in (tmp_path / "benchmark.md").read_text()
    )


@pytest.mark.parametrize("corruption", ["missing", "rows", "action", "algorithm"])
def test_action_stability_rejects_invalid_prediction_artifacts(tmp_path, corruption):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5]}, [1, 2])
    (paths[0] / "predictions.json").write_text(
        json.dumps({"predictions": {"dqn": [0, 1]}})
    )
    if corruption != "missing":
        values = {
            "rows": {"dqn": [0]},
            "action": {"dqn": [0, 4]},
            "algorithm": {"cql": [0, 1]},
        }[corruption]
        (paths[1] / "predictions.json").write_text(json.dumps({"predictions": values}))
    with pytest.raises(ValueError, match="prediction"):
        aggregate_runs(paths, [1, 2])


def _write_policy_agreement(path, agreement, differing):
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["policy_agreement"] = {
        "algorithms": ["dqn", "cql"],
        "num_examples": 4,
        "agreement": [[1.0, agreement], [agreement, 1.0]],
        "disagreements": [[0, differing], [differing, 0]],
    }
    metrics_path.write_text(json.dumps(metrics))


def test_benchmark_aggregates_cross_algorithm_policy_agreement(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5], "cql": [0.5, 0.5]}, [9, 2])
    _write_policy_agreement(paths[0], 0.5, 2)
    _write_policy_agreement(paths[1], 0.75, 1)
    aggregate = aggregate_runs(paths, [9, 2])
    comparison = aggregate["policy_agreement"]
    assert comparison["num_examples"] == 4
    (pair,) = comparison["pairs"]
    assert (pair["first_algorithm"], pair["second_algorithm"]) == ("cql", "dqn")
    assert pair["agreement"]["mean"] == 0.625
    assert pair["disagreements"]["mean"] == 1.5
    assert pair["per_seed"] == [
        {"seed": 9, "agreement": 0.5, "disagreements": 2},
        {"seed": 2, "agreement": 0.75, "disagreements": 1},
    ]
    generate_benchmark_report(tmp_path, aggregate)
    report = (tmp_path / "benchmark.md").read_text()
    assert "Agreement between algorithms across seeds" in report
    assert "| CQL | DQN | 62.50% | 17.68% | 1.50 |" in report


@pytest.mark.parametrize("damage", ["missing", "asymmetric", "count", "rows"])
def test_benchmark_rejects_inconsistent_policy_agreement(tmp_path, damage):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5], "cql": [0.5, 0.5]}, [1, 2])
    _write_policy_agreement(paths[0], 0.5, 2)
    if damage != "missing":
        _write_policy_agreement(paths[1], 0.5, 2)
        path = paths[1] / "metrics.json"
        metrics = json.loads(path.read_text())
        comparison = metrics["policy_agreement"]
        if damage == "asymmetric":
            comparison["agreement"][1][0] = 0.75
        elif damage == "count":
            comparison["disagreements"][0][1] = 1
        else:
            comparison["num_examples"] = 5
        path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="policy agreement"):
        aggregate_runs(paths, [1, 2])


def test_single_algorithm_and_legacy_reports(tmp_path):
    aggregate = aggregate_runs(_runs(tmp_path, {"dqn": [0.5]}, [1]), [1])
    assert aggregate["paired_comparisons"] == []
    del aggregate["paired_comparisons"]
    generate_benchmark_report(tmp_path, aggregate)
    assert "Paired seed comparisons" not in (tmp_path / "benchmark.md").read_text()
    assert "Macro F1 mean" not in (tmp_path / "benchmark.md").read_text()


def test_macro_f1_is_aggregated_and_invalid_values_are_rejected(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.75]}, [1, 2])
    for path, score in zip(paths, [0.25, 0.75], strict=True):
        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["algorithms"]["dqn"]["evaluation"]["macro_f1"] = score
        metrics_path.write_text(json.dumps(metrics))
    aggregate = aggregate_runs(paths, [1, 2])
    assert aggregate["algorithms"]["dqn"]["macro_f1"]["mean"] == 0.5
    assert aggregate["algorithms"]["dqn"]["macro_f1"]["std"] == pytest.approx(
        0.5 / 2**0.5
    )
    generate_benchmark_report(tmp_path, aggregate)
    assert "Macro F1 mean | Macro F1 std" in (tmp_path / "benchmark.md").read_text()
    second_path = paths[1] / "metrics.json"
    metrics = json.loads(second_path.read_text())
    metrics["algorithms"]["dqn"]["evaluation"]["macro_f1"] = 1.5
    second_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="macro F1"):
        aggregate_runs(paths, [1, 2])


def test_balanced_accuracy_is_aggregated_and_validated(tmp_path):
    paths = _runs(tmp_path, {"dqn": [0.5, 0.5]}, [1, 2])
    for path, score in zip(paths, [0.25, 0.75], strict=True):
        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["algorithms"]["dqn"]["evaluation"]["balanced_accuracy"] = score
        metrics_path.write_text(json.dumps(metrics))
    aggregate = aggregate_runs(paths, [1, 2])
    assert aggregate["algorithms"]["dqn"]["balanced_accuracy"]["mean"] == 0.5
    generate_benchmark_report(tmp_path, aggregate)
    assert "Balanced accuracy mean" in (tmp_path / "benchmark.md").read_text()

    metrics_path = paths[1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["algorithms"]["dqn"]["evaluation"]["balanced_accuracy"] = 1.5
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="balanced accuracy"):
        aggregate_runs(paths, [1, 2])


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
