from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
import torch

from gridworld_rl.cli import _apply_training_overrides, build_parser
from gridworld_rl.config import ExperimentConfig
from gridworld_rl.data import TransitionDataset
from gridworld_rl.trainer import train_model, update_target_model


class _TinyModel(torch.nn.Module):
    def __init__(self, value: float, count: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))
        self.register_buffer("count", torch.tensor(count))


@pytest.mark.parametrize(
    "tau,expected",
    [(1.0, 10.0), (0.5, 6.0), (0.25, 4.0)],
)
def test_target_update_blends_parameters_and_copies_buffers(tau, expected):
    online = _TinyModel(10.0, 9)
    target = _TinyModel(2.0, 1)
    update_target_model(target, online, tau)
    assert target.weight.item() == pytest.approx(expected)
    assert target.count.item() == 9
    assert online.weight.item() == 10.0
    assert online.count.item() == 9
    assert target.weight.grad is None
    assert target.weight.requires_grad


def test_repeated_blending_converges_without_mutating_online():
    online = _TinyModel(10.0, 0)
    target = _TinyModel(2.0, 0)
    for _ in range(3):
        update_target_model(target, online, 0.5)
    assert target.weight.item() == pytest.approx(9.0)
    assert online.weight.item() == 10.0


@pytest.mark.parametrize(
    "invalid", [0, -0.1, 1.1, True, None, "0.5", math.nan, math.inf]
)
def test_config_rejects_invalid_tau(invalid):
    with pytest.raises(ValueError, match="target_update_tau"):
        ExperimentConfig.from_dict({"training": {"target_update_tau": invalid}})
    with pytest.raises(ValueError, match="tau must"):
        update_target_model(_TinyModel(2.0, 1), _TinyModel(10.0, 9), invalid)


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_cli_exposes_tau_for_training_and_benchmarks(command):
    argv = [command, "--target-update-tau", "0.25"]
    if command == "benchmark":
        argv += ["--seeds", "1", "2"]
    config = _apply_training_overrides(
        ExperimentConfig(), build_parser().parse_args(argv)
    )
    assert config.training.target_update_tau == 0.25
    assert config.to_dict()["training"]["target_update_tau"] == 0.25


def test_tau_affects_training_only_after_scheduled_updates():
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": list(range(8)),
                "action": [0, 1, 2, 3] * 2,
                "reward": [0.1, 1.0, -0.5, 2.0] * 2,
                "next_state": list(range(1, 9)),
                "done": [False] * 7 + [True],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [8]},
            "training": {
                "epochs": 3,
                "batch_size": 2,
                "target_update_interval": 1,
                "learning_rate": 0.01,
                "target_update_tau": 1.0,
            },
        }
    )
    hard, hard_metrics = train_model("dqn", dataset, config, torch.device("cpu"))
    soft_config = copy.deepcopy(config)
    soft_config.training.target_update_tau = 0.1
    soft, soft_metrics = train_model("dqn", dataset, soft_config, torch.device("cpu"))
    assert hard_metrics["global_steps"] == soft_metrics["global_steps"] == 12
    assert (
        hard_metrics["target_synchronizations"]
        == soft_metrics["target_synchronizations"]
        == 13
    )
    assert any(
        not torch.equal(hard.state_dict()[name], soft.state_dict()[name])
        for name in hard.state_dict()
    )
