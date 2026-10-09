from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.config import ExperimentConfig
from gridworld_rl.reproducibility import sha256_file
from gridworld_rl.run_comparison import compare_runs


def _run(tmp_path, name, *, seed, score, train_hash="a" * 64, solution_hash="c" * 64):
    directory = tmp_path / name
    directory.mkdir()
    config = ExperimentConfig()
    config.training.algorithms = ["dqn"]
    config.training.seed = seed
    config.output.run_name = name
    config.to_json(directory / "config.json")
    metrics = {
        "schema_version": 1,
        "seed": seed,
        "dataset": {
            "sha256": {
                "train": train_hash,
                "eval_challenge": "b" * 64,
                "eval_solution": solution_hash,
            }
        },
        "algorithms": {
            "dqn": {
                "evaluation": {
                    "accuracy": score,
                    "balanced_accuracy": score,
                    "macro_f1": score,
                }
            }
        },
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    _manifest(directory)
    return directory


def _manifest(directory):
    hashes = {
        name: sha256_file(directory / name) for name in ("config.json", "metrics.json")
    }
    (directory / "manifest.json").write_text(json.dumps({"sha256": hashes}))


def test_compare_runs_verifies_and_reports_deltas(tmp_path, capsys):
    left = _run(tmp_path, "left", seed=11, score=0.5)
    right = _run(tmp_path, "right", seed=22, score=0.75)
    result = compare_runs(left, right)
    assert result["same_evaluation_split"] is True
    assert all(result["dataset_match"].values())
    assert result["settings_differences"] == {}
    assert result["metric_deltas"]["dqn"]["accuracy"] == {
        "left": 0.5,
        "right": 0.75,
        "right_minus_left": 0.25,
    }
    assert main(["compare-runs", "--left", str(left), "--right", str(right)]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_compare_runs_flags_changed_inputs_and_settings(tmp_path):
    left = _run(tmp_path, "left", seed=11, score=0.5)
    right = _run(
        tmp_path,
        "right",
        seed=22,
        score=0.75,
        train_hash="d" * 64,
        solution_hash="e" * 64,
    )
    config = ExperimentConfig.from_json(right / "config.json")
    config.training.learning_rate = 0.02
    config.to_json(right / "config.json")
    _manifest(right)
    result = compare_runs(left, right)
    assert result["dataset_match"] == {
        "train": False,
        "eval_challenge": True,
        "eval_solution": False,
    }
    assert result["same_evaluation_split"] is False
    assert result["metric_deltas"] == {}
    assert result["settings_differences"]["training"]["right"]["learning_rate"] == 0.02


def test_compare_runs_rejects_damaged_or_inconsistent_runs(tmp_path):
    left = _run(tmp_path, "left", seed=11, score=0.5)
    right = _run(tmp_path, "right", seed=22, score=0.75)
    with pytest.raises(ValueError, match="distinct"):
        compare_runs(left, left)
    (right / "metrics.json").write_text("tampered")
    with pytest.raises(ValueError, match="integrity verification"):
        compare_runs(left, right)
    metrics = json.loads((left / "metrics.json").read_text())
    metrics["seed"] = 99
    (left / "metrics.json").write_text(json.dumps(metrics))
    _manifest(left)
    with pytest.raises(ValueError, match="seed disagrees"):
        compare_runs(left, _run(tmp_path, "third", seed=33, score=0.6))


def test_compare_runs_rejects_nonfinite_or_out_of_range_scores(tmp_path):
    left = _run(tmp_path, "left", seed=11, score=0.5)
    right = _run(tmp_path, "right", seed=22, score=0.75)
    metrics = json.loads((right / "metrics.json").read_text())
    metrics["algorithms"]["dqn"]["evaluation"]["accuracy"] = 1.2
    (right / "metrics.json").write_text(json.dumps(metrics))
    _manifest(right)
    with pytest.raises(ValueError, match="Invalid accuracy"):
        compare_runs(left, right)
