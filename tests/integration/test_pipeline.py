from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from gridworld_rl.benchmark import aggregate_runs, run_benchmark
from gridworld_rl.checkpoints import load_checkpoint
from gridworld_rl.cli import main
from gridworld_rl.config import ExperimentConfig
from gridworld_rl.report import generate_report
from gridworld_rl.reproducibility import sha256_file
from gridworld_rl.trainer import run_experiment


def _write_tiny_dataset(directory: Path) -> tuple[Path, Path, Path]:
    train = pd.DataFrame(
        {
            "state": [0, 1, 2, 3, 4, 5, 6, 7],
            "action": [0, 1, 2, 3, 0, 1, 2, 3],
            "reward": [0.4, -0.3, 0.4, -1.3, 0.4, -0.3, 0.4, 20.0],
            "next_state": [1, 2, 3, 4, 5, 6, 7, 99],
            "done": [False, False, False, False, False, False, False, True],
        }
    )
    solution = train.iloc[:4].copy()
    challenge = solution.copy()
    challenge["action"] = -1
    challenge["reward"] = -999
    paths = (
        directory / "train.csv",
        directory / "eval_challenge.csv",
        directory / "eval_solution.csv",
    )
    train.to_csv(paths[0], index=False)
    challenge.to_csv(paths[1], index=False)
    solution.to_csv(paths[2], index=False)
    return paths


def _config(tmp_path: Path, run_name: str = "test") -> ExperimentConfig:
    train, challenge, solution = _write_tiny_dataset(tmp_path)
    return ExperimentConfig.from_dict(
        {
            "dataset": {
                "train": str(train),
                "eval_challenge": str(challenge),
                "eval_solution": str(solution),
            },
            "network": {"hidden_sizes": [8]},
            "training": {
                "algorithms": ["dqn"],
                "epochs": 2,
                "learning_rate": 0.01,
                "batch_size": 4,
                "gamma": 0.9,
                "epsilon": 0.1,
                "cql_alpha": 0.5,
                "target_update_interval": 1,
                "gradient_clip_norm": 1.0,
                "seed": 7,
                "device": "cpu",
            },
            "output": {"directory": str(tmp_path / "artifacts"), "run_name": run_name},
        }
    )


def test_config_rejects_invalid_hyperparameters_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="gamma"):
        ExperimentConfig.from_dict({"training": {"gamma": 1.1}})
    with pytest.raises(ValueError, match="Invalid configuration field"):
        ExperimentConfig.from_dict({"training": {"not_a_setting": 3}})
    for field in ("learning_rate", "cql_alpha", "gradient_clip_norm"):
        with pytest.raises(ValueError, match="finite"):
            ExperimentConfig.from_dict({"training": {field: float("nan")}})
        with pytest.raises(ValueError, match="finite"):
            ExperimentConfig.from_dict({"training": {field: float("inf")}})


def test_end_to_end_run_writes_loadable_reproducible_artifacts(tmp_path) -> None:
    config = _config(tmp_path)
    run_dir = run_experiment(config)

    expected = {
        "config.json",
        "metrics.json",
        "predictions.json",
        "report.png",
        "confusion_matrices.png",
        "summary.md",
        "manifest.json",
        "checkpoints/dqn.pt",
    }
    assert expected.issubset(
        {
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        }
    )
    predictions = json.loads((run_dir / "predictions.json").read_text())
    assert "targets" not in predictions
    assert len(predictions["predictions"]["dqn"]) == 4
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["algorithms"]["dqn"]["target_synchronizations"] == 5
    assert {
        "python",
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "gridworld_offline_rl",
    } == set(metrics["runtime"])
    assert {"git_commit", "git_dirty"} == set(metrics["source"])
    assert "state_action_coverage" in metrics["dataset"]["diagnostics"]
    summary = (run_dir / "summary.md").read_text()
    assert "| Algorithm | Agreement | Recall a0" in summary
    assert "| DQN |" in summary and "%" in summary.split("| DQN |", maxsplit=1)[1]

    model, payload = load_checkpoint(run_dir / "checkpoints/dqn.pt")
    assert payload["algorithm"] == "dqn"
    assert model(torch.tensor([0])).shape == (1, 4)

    regenerated = generate_report(run_dir)
    assert all(path.is_file() and path.stat().st_size > 0 for path in regenerated)
    manifest = json.loads((run_dir / "manifest.json").read_text())["sha256"]
    assert all(
        sha256_file(run_dir / relative_path) == digest
        for relative_path, digest in manifest.items()
    )


def test_seeded_runs_produce_identical_histories_and_predictions(tmp_path) -> None:
    first_dir = run_experiment(_config(tmp_path, "first"))
    second_dir = run_experiment(_config(tmp_path, "second"))

    first_metrics = json.loads((first_dir / "metrics.json").read_text())
    second_metrics = json.loads((second_dir / "metrics.json").read_text())
    assert (
        first_metrics["algorithms"]["dqn"]["training_history"]
        == second_metrics["algorithms"]["dqn"]["training_history"]
    )
    assert json.loads((first_dir / "predictions.json").read_text()) == json.loads(
        (second_dir / "predictions.json").read_text()
    )


def test_existing_run_requires_explicit_atomic_overwrite(tmp_path) -> None:
    config = _config(tmp_path, "replace-me")
    run_dir = run_experiment(config)
    marker = run_dir / "stale-marker.txt"
    marker.write_text("preserve until replacement")

    with pytest.raises(FileExistsError, match="Choose a new run name"):
        run_experiment(config)
    assert marker.is_file()

    config.output.overwrite = True
    replaced_dir = run_experiment(config)
    assert replaced_dir == run_dir
    assert not marker.exists()
    assert (run_dir / "manifest.json").is_file()


def test_cli_reports_missing_dataset_without_traceback(tmp_path, capsys) -> None:
    config = ExperimentConfig()
    config.dataset.train = str(tmp_path / "missing.csv")
    config_path = tmp_path / "config.json"
    config.to_json(config_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--config", str(config_path)])

    assert exit_info.value.code == 2
    assert "Dataset file not found" in capsys.readouterr().err


def test_multi_seed_benchmark_writes_aggregate_metrics_and_manifest(tmp_path) -> None:
    benchmark_dir = run_benchmark(
        _config(tmp_path, "unused"),
        seeds=[3, 5],
        output_directory=tmp_path / "benchmarks",
        name="comparison",
    )

    aggregate = json.loads((benchmark_dir / "aggregate_metrics.json").read_text())
    assert aggregate["seeds"] == [3, 5]
    assert aggregate["num_runs"] == 2
    assert set(aggregate["algorithms"]["dqn"]["accuracy"]) == {
        "mean",
        "std",
        "minimum",
        "maximum",
    }
    assert (benchmark_dir / "benchmark.png").stat().st_size > 0
    assert (benchmark_dir / "benchmark.md").is_file()
    benchmark_summary = (benchmark_dir / "benchmark.md").read_text()
    assert "| Algorithm | Agreement mean | Agreement std" in benchmark_summary
    assert "| DQN |" in benchmark_summary
    assert "%" in benchmark_summary.split("| DQN |", maxsplit=1)[1]

    manifest = json.loads((benchmark_dir / "manifest.json").read_text())["sha256"]
    assert all(
        sha256_file(benchmark_dir / relative_path) == digest
        for relative_path, digest in manifest.items()
    )

    second_metrics_path = benchmark_dir / "runs/seed-5/metrics.json"
    second_metrics = json.loads(second_metrics_path.read_text())
    second_metrics["dataset"]["sha256"]["train"] = "different"
    second_metrics_path.write_text(json.dumps(second_metrics))
    with pytest.raises(ValueError, match="different dataset fingerprints"):
        aggregate_runs(
            [
                benchmark_dir / "runs/seed-3",
                benchmark_dir / "runs/seed-5",
            ],
            [3, 5],
        )
    with pytest.raises(FileExistsError, match="immutable evidence"):
        run_benchmark(
            _config(tmp_path, "unused-again"),
            seeds=[7],
            output_directory=tmp_path / "benchmarks",
            name="comparison",
        )


def test_cli_train_report_and_benchmark_commands(tmp_path, capsys) -> None:
    config = _config(tmp_path, "cli-run")
    config.training.epochs = 1
    config_path = tmp_path / "cli-config.json"
    config.to_json(config_path)

    assert main(["train", "--config", str(config_path)]) == 0
    run_dir = Path(config.output.directory) / config.output.run_name
    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_path),
                "--seeds",
                "13",
                "17",
                "--output-dir",
                str(tmp_path / "cli-benchmarks"),
                "--name",
                "cli-comparison",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Experiment complete:" in output
    assert "Generated report files:" in output
    assert "Benchmark complete:" in output


def test_cli_option_only_invocation_defaults_to_train(tmp_path, capsys) -> None:
    _write_tiny_dataset(tmp_path)

    assert (
        main(
            [
                "--data-dir",
                str(tmp_path),
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "direct-artifacts"),
                "--run-name",
                "direct-run",
            ]
        )
        == 0
    )

    run_dir = tmp_path / "direct-artifacts/direct-run"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["dataset"]["train"] == str(tmp_path / "train.csv")
    assert config["training"]["epochs"] == 1
    assert "Experiment complete:" in capsys.readouterr().out
