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
from gridworld_rl.integrity import verify_artifacts
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
    slices = metrics["algorithms"]["dqn"]["evaluation"]["overlap_slices"]
    assert slices["exact_training_transition"]["rows"] == 4
    assert slices["seen_state_new_transition"] == {"rows": 0, "accuracy": None}
    assert slices["unseen_state"] == {"rows": 0, "accuracy": None}
    evaluation = metrics["algorithms"]["dqn"]["evaluation"]
    assert 0 <= evaluation["macro_f1"] <= 1
    assert set(evaluation["per_action_precision"]) == {"0", "1", "2", "3"}
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
    assert "| Algorithm | Agreement | Macro F1 | Recall a0" in summary
    assert "Per-action precision and F1" in summary
    assert "Agreement by training overlap" in summary
    assert "| DQN | Unseen state | 0 | N/A |" in summary
    assert "| DQN |" in summary and "%" in summary.split("| DQN |", maxsplit=1)[1]

    model, payload = load_checkpoint(run_dir / "checkpoints/dqn.pt")
    assert payload["algorithm"] == "dqn"
    assert model(torch.tensor([0])).shape == (1, 4)

    regenerated = generate_report(run_dir)
    assert verify_artifacts(run_dir).valid
    assert all(path.is_file() and path.stat().st_size > 0 for path in regenerated)
    manifest = json.loads((run_dir / "manifest.json").read_text())["sha256"]
    assert all(
        sha256_file(run_dir / relative_path) == digest
        for relative_path, digest in manifest.items()
    )


@pytest.mark.parametrize("damage", ["modified", "missing", "unexpected"])
def test_report_refresh_refuses_damaged_run_without_rehashing(tmp_path, damage) -> None:
    run_dir = run_experiment(_config(tmp_path))
    report_files = [
        "report.png",
        "confusion_matrices.png",
        "summary.md",
        "manifest.json",
    ]
    before = {name: (run_dir / name).read_bytes() for name in report_files}
    checkpoint = run_dir / "checkpoints/dqn.pt"
    if damage == "modified":
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    elif damage == "missing":
        checkpoint.rename(run_dir / "checkpoints/saved.pt")
    else:
        (run_dir / "extra.txt").write_text("unexpected")

    with pytest.raises(ValueError, match="invalid artifact integrity"):
        generate_report(run_dir)
    assert {name: (run_dir / name).read_bytes() for name in report_files} == before


def test_step_limited_run_saves_partial_epoch_metadata(tmp_path) -> None:
    config = _config(tmp_path)
    config.training.max_optimizer_steps = 1
    run_dir = run_experiment(config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    training = metrics["algorithms"]["dqn"]
    assert training["global_steps"] == 1
    assert training["planned_global_steps"] == 4
    assert training["completed_epochs"] == 0
    assert training["partial_epoch_batches"] == 1
    assert len(training["training_history"]["total_loss"]) == 1
    assert verify_artifacts(run_dir).valid


def test_multi_algorithm_run_reports_policy_agreement(tmp_path) -> None:
    config = _config(tmp_path, "two-algorithms")
    config.training.algorithms = ["dqn", "cql"]
    run_dir = run_experiment(config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    predictions = json.loads((run_dir / "predictions.json").read_text())["predictions"]
    comparison = metrics["policy_agreement"]
    assert comparison["algorithms"] == ["dqn", "cql"]
    assert comparison["num_examples"] == 4
    differences = sum(
        first != second
        for first, second in zip(predictions["dqn"], predictions["cql"], strict=True)
    )
    assert comparison["disagreements"] == [[0, differences], [differences, 0]]
    assert comparison["agreement"][0][1] == (4 - differences) / 4
    assert "Agreement between algorithms" in (run_dir / "summary.md").read_text()
    assert verify_artifacts(run_dir).valid


@pytest.mark.parametrize("stage", ["after_load", "before_publish"])
def test_run_refuses_dataset_changes_and_cleans_staging(tmp_path, monkeypatch, stage):
    import gridworld_rl.trainer as trainer

    config = _config(tmp_path, "changing")
    train_path = Path(config.dataset.train)

    if stage == "after_load":
        original = trainer.load_transition_csv

        def load_then_change(*args, **kwargs):
            frame = original(*args, **kwargs)
            train_path.write_bytes(train_path.read_bytes() + b"\n")
            return frame

        monkeypatch.setattr(trainer, "load_transition_csv", load_then_change)
    else:
        original = trainer.generate_report

        def report_then_change(directory):
            files = original(directory)
            train_path.write_bytes(train_path.read_bytes() + b"\n")
            return files

        monkeypatch.setattr(trainer, "generate_report", report_then_change)

    with pytest.raises(RuntimeError, match=r"Dataset file.*changed.*train"):
        run_experiment(config)
    assert not (Path(config.output.directory) / config.output.run_name).exists()
    assert not list(Path(config.output.directory).glob(".changing.in-progress-*"))


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


@pytest.mark.parametrize("existing_run", [False, True])
def test_divergence_preserves_published_artifacts(
    tmp_path, monkeypatch, capsys, existing_run
):
    import gridworld_rl.trainer as trainer

    config = _config(tmp_path, "protected")
    run_dir = Path(config.output.directory) / config.output.run_name
    before = {}
    if existing_run:
        run_experiment(config)
        before = {
            p.relative_to(run_dir): p.read_bytes()
            for p in run_dir.rglob("*")
            if p.is_file()
        }
        config.output.overwrite = True
    original_loss = trainer.calculate_loss
    calls = 0

    def diverging_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        loss, parts = original_loss(*args, **kwargs)
        return (loss * float("nan") if calls == 2 else loss), parts

    monkeypatch.setattr(trainer, "calculate_loss", diverging_loss)
    config_path = tmp_path / "failure.json"
    config.to_json(config_path)
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config_path)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Non-finite loss" in error and "batch 2" in error
    assert "Traceback" not in error
    if existing_run:
        assert before == {
            p.relative_to(run_dir): p.read_bytes()
            for p in run_dir.rglob("*")
            if p.is_file()
        }
        assert verify_artifacts(run_dir).valid
    else:
        assert not run_dir.exists()
    assert not list(run_dir.parent.glob(".protected.in-progress-*"))


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
    assert "macro_f1" in aggregate["algorithms"]["dqn"]
    assert aggregate["algorithms"]["dqn"]["overlap_slices"]["unseen_state"] == {
        "rows": 0,
        "accuracy": None,
    }
    assert set(aggregate["algorithms"]["dqn"]["accuracy"]) == {
        "mean",
        "std",
        "minimum",
        "maximum",
    }
    stability = aggregate["action_stability"]["dqn"]
    assert stability["evaluation_rows"] == 4
    assert [
        (pair["first_seed"], pair["second_seed"]) for pair in stability["seed_pairs"]
    ] == [(3, 5)]
    assert 0 <= stability["pairwise_agreement"]["mean"] <= 1
    assert (benchmark_dir / "benchmark.png").stat().st_size > 0
    assert (benchmark_dir / "benchmark.md").is_file()
    assert verify_artifacts(benchmark_dir).valid
    assert not list(benchmark_dir.parent.glob(".comparison.in-progress-*"))
    assert json.loads((benchmark_dir / "aggregate_metrics.json").read_text())[
        "run_directories"
    ] == ["runs/seed-3", "runs/seed-5"]
    benchmark_summary = (benchmark_dir / "benchmark.md").read_text()
    assert "| Algorithm | Agreement mean | Agreement std" in benchmark_summary
    assert "Macro F1 mean | Macro F1 std" in benchmark_summary
    assert "Action stability across seeds" in benchmark_summary
    assert "| DQN | Unseen state | 0 | N/A | N/A |" in benchmark_summary
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


@pytest.mark.parametrize("failure", ["seed", "report", "late_destination"])
def test_benchmark_failure_never_publishes_partial_results(
    tmp_path, monkeypatch, failure
):
    import gridworld_rl.benchmark as benchmark

    config = _config(tmp_path, "unused")
    parent = tmp_path / "benchmarks"
    destination = parent / "unfinished"
    original_run = benchmark.run_experiment
    original_report = benchmark.generate_benchmark_report
    calls = 0

    def run_with_failure(run_config):
        nonlocal calls
        calls += 1
        assert not destination.exists()
        if failure == "seed" and calls == 2:
            raise RuntimeError("seed failed")
        return original_run(run_config)

    def report_with_failure(directory, aggregate):
        assert not destination.exists()
        if failure == "report":
            raise RuntimeError("report failed")
        result = original_report(directory, aggregate)
        if failure == "late_destination":
            destination.mkdir()
            (destination / "marker.txt").write_text("keep me")
        return result

    monkeypatch.setattr(benchmark, "run_experiment", run_with_failure)
    monkeypatch.setattr(benchmark, "generate_benchmark_report", report_with_failure)
    error = FileExistsError if failure == "late_destination" else RuntimeError
    with pytest.raises(error):
        run_benchmark(config, seeds=[3, 5], output_directory=parent, name="unfinished")
    assert calls == 2
    if failure == "late_destination":
        assert (destination / "marker.txt").read_text() == "keep me"
    else:
        assert not destination.exists()
    assert not list(parent.glob(".unfinished.in-progress-*"))


def test_cli_train_report_and_benchmark_commands(tmp_path, capsys) -> None:
    config = _config(tmp_path, "cli-run")
    config.training.epochs = 1
    config_path = tmp_path / "cli-config.json"
    config.to_json(config_path)

    assert main(["train", "--config", str(config_path), "--algorithms", "cql"]) == 0
    run_dir = Path(config.output.directory) / config.output.run_name
    assert {path.name for path in (run_dir / "checkpoints").iterdir()} == {"cql.pt"}
    assert json.loads((run_dir / "config.json").read_text())["training"][
        "algorithms"
    ] == ["cql"]
    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_path),
                "--algorithms",
                "double_dqn",
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
    benchmark_dir = tmp_path / "cli-benchmarks/cli-comparison"
    aggregate = json.loads((benchmark_dir / "aggregate_metrics.json").read_text())
    assert list(aggregate["algorithms"]) == ["double_dqn"]
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
