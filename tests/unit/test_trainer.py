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


def test_epoch_metrics_weight_transitions():
    first = dict.fromkeys(("td_loss", "cql_loss", "total_loss"), 2.0)
    last = dict.fromkeys(first, 10.0)
    assert _mean_history([(3, first), (1, last)]) == dict.fromkeys(first, 4.0)
    with pytest.raises(RuntimeError, match="no batches"):
        _mean_history([])


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
