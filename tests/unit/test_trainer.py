from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
import torch

from gridworld_rl.cli import _apply_training_overrides, build_parser
from gridworld_rl.config import SUPPORTED_ALGORITHMS, ExperimentConfig
from gridworld_rl.data import TransitionDataset
from gridworld_rl.models import QNetwork
from gridworld_rl.trainer import _mean_history, train_model


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize("rows,accumulation", [(11, 2), (5, 4), (12, 2), (7, 1)])
def test_accumulation_matches_full_batches(algorithm, schedule, rows, accumulation):
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [i % 8 for i in range(rows)],
                "action": [i % 4 for i in range(rows)],
                "reward": [0.07 * (i - 3) for i in range(rows)],
                "next_state": [(i + 1) % 8 for i in range(rows)],
                "done": [i % 3 == 0 for i in range(rows)],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [8]},
            "training": {
                "epochs": 3,
                "learning_rate_schedule": schedule,
                "warmup_steps": 1 if schedule == "cosine" else 0,
                "min_learning_rate_ratio": 0.1,
                "batch_size": 3,
                "gradient_accumulation_steps": accumulation,
                "target_update_interval": 2,
                "gradient_clip_norm": 0.1,
                "device": "cpu",
            },
        }
    )
    full_config = copy.deepcopy(config)
    full_config.training.batch_size *= accumulation
    full_config.training.gradient_accumulation_steps = 1
    model, metrics = train_model(algorithm, dataset, config, torch.device("cpu"))
    full_model, full_metrics = train_model(
        algorithm, dataset, full_config, torch.device("cpu")
    )

    steps = 3 * math.ceil(rows / (3 * accumulation))
    assert metrics["global_steps"] == full_metrics["global_steps"] == steps
    assert metrics["target_synchronizations"] == 1 + steps // 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, full_model.state_dict()[name])
    for key, values in metrics["training_history"].items():
        assert values == pytest.approx(
            full_metrics["training_history"][key], rel=1e-5, abs=1e-7
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_accumulation_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        ExperimentConfig.from_dict({"training": {"gradient_accumulation_steps": value}})


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_accumulation_cli_override(command):
    argv = [command, "--gradient-accumulation-steps", "4"]
    if command == "benchmark":
        argv.extend(["--seeds", "1", "2"])
    args = build_parser().parse_args(argv)
    config = _apply_training_overrides(ExperimentConfig(), args)
    assert config.to_dict()["training"]["gradient_accumulation_steps"] == 4


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_cli_algorithm_selection_validates_and_preserves_order(command):
    argv = [command, "--algorithms", "cql", "dqn"]
    if command == "benchmark":
        argv += ["--seeds", "1", "2"]
    config = _apply_training_overrides(
        ExperimentConfig(), build_parser().parse_args(argv)
    )
    assert config.training.algorithms == ["cql", "dqn"]

    duplicate = [command, "--algorithms", "dqn", "dqn"]
    if command == "benchmark":
        duplicate += ["--seeds", "1", "2"]
    with pytest.raises(ValueError, match="duplicates"):
        _apply_training_overrides(
            ExperimentConfig(), build_parser().parse_args(duplicate)
        )

    invalid = [command, "--algorithms", "unknown"]
    if command == "benchmark":
        invalid += ["--seeds", "1", "2"]
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(invalid)
    assert exc.value.code == 2


def test_epoch_metrics_weight_transitions():
    first = dict.fromkeys(("td_loss", "cql_loss", "total_loss"), 2.0)
    last = dict.fromkeys(first, 10.0)
    assert _mean_history([(3, first), (1, last)]) == dict.fromkeys(first, 4.0)
    with pytest.raises(RuntimeError, match="no batches"):
        _mean_history([])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_max_optimizer_steps_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_optimizer_steps"):
        ExperimentConfig.from_dict({"training": {"max_optimizer_steps": value}})


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_max_optimizer_steps_cli_override(command):
    argv = [command, "--max-optimizer-steps", "4"]
    if command == "benchmark":
        argv += ["--seeds", "1", "2"]
    config = _apply_training_overrides(
        ExperimentConfig(), build_parser().parse_args(argv)
    )
    assert config.training.max_optimizer_steps == 4


def test_step_budget_stops_after_full_accumulation_window_in_partial_epoch():
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [i % 8 for i in range(10)],
                "action": [i % 4 for i in range(10)],
                "reward": [0.1 * i for i in range(10)],
                "next_state": [(i + 1) % 8 for i in range(10)],
                "done": [i % 3 == 0 for i in range(10)],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [8]},
            "training": {
                "epochs": 5,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "max_optimizer_steps": 4,
                "learning_rate_schedule": "cosine",
                "target_update_interval": 2,
            },
        }
    )
    shorter = copy.deepcopy(config)
    shorter.training.epochs = 2
    model, metrics = train_model("dqn", dataset, config, torch.device("cpu"))
    short_model, short_metrics = train_model(
        "dqn", dataset, shorter, torch.device("cpu")
    )
    assert metrics["global_steps"] == short_metrics["global_steps"] == 4
    assert metrics["planned_global_steps"] == 15
    assert metrics["completed_epochs"] == 1
    assert metrics["partial_epoch_batches"] == 2
    assert len(metrics["training_history"]["total_loss"]) == 2
    assert metrics["target_synchronizations"] == 3
    assert metrics["training_history"] == short_metrics["training_history"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, short_model.state_dict()[name])


def test_step_budget_rejects_warmup_that_consumes_every_update():
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [0, 1],
                "action": [0, 1],
                "reward": [0.0, 0.0],
                "next_state": [1, 0],
                "done": [True, True],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {"training": {"max_optimizer_steps": 1, "warmup_steps": 1}}
    )
    with pytest.raises(ValueError, match="warmup_steps"):
        train_model("dqn", dataset, config, torch.device("cpu"))


@pytest.mark.parametrize(
    "algorithm,expected_forwards",
    [("dqn", 4), ("cql", 4), ("double_dqn", 6), ("expected_sarsa", 6)],
)
def test_next_online_network_runs_only_when_target_requires_it(
    monkeypatch, algorithm, expected_forwards
):
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [0, 1, 2, 3],
                "action": [0, 1, 2, 3],
                "reward": [0.1, 0.2, 0.3, 0.4],
                "next_state": [1, 2, 3, 4],
                "done": [False, False, False, True],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [8]},
            "training": {"algorithms": [algorithm], "epochs": 1, "batch_size": 2},
        }
    )
    original_forward = QNetwork.forward
    calls = 0

    def recording_forward(model, states):
        nonlocal calls
        calls += 1
        return original_forward(model, states)

    monkeypatch.setattr(QNetwork, "forward", recording_forward)
    _, metrics = train_model(algorithm, dataset, config, torch.device("cpu"))
    assert metrics["global_steps"] == 2
    assert calls == expected_forwards


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("terminal_rows", [2, 4])
def test_terminal_transitions_skip_next_state_inference(
    monkeypatch, algorithm, terminal_rows
):
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [0, 1, 2, 3],
                "action": [0, 1, 2, 3],
                "reward": [0.1, 0.2, 0.3, 0.4],
                "next_state": [4, 5, 6, 7],
                "done": [i < terminal_rows for i in range(4)],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [8]},
            "training": {"algorithms": [algorithm], "epochs": 1, "batch_size": 4},
        }
    )
    original_forward = QNetwork.forward
    batches = []

    def recording_forward(model, states):
        batches.append(states.tolist())
        return original_forward(model, states)

    monkeypatch.setattr(QNetwork, "forward", recording_forward)
    _, metrics = train_model(algorithm, dataset, config, torch.device("cpu"))
    assert metrics["global_steps"] == 1
    assert sorted(batches[0]) == [0, 1, 2, 3]
    expected_next = [i for i in range(4 + terminal_rows, 8)]
    if terminal_rows == 4:
        assert len(batches) == 1
    else:
        assert len(batches) == (
            3 if algorithm in {"double_dqn", "expected_sarsa"} else 2
        )
        assert all(sorted(batch) == expected_next for batch in batches[1:])
