from __future__ import annotations

import pandas as pd
import pytest
import torch

from gridworld_rl.cli import _apply_training_overrides, build_parser
from gridworld_rl.config import ExperimentConfig, TrainingConfig
from gridworld_rl.data import TransitionDataset
from gridworld_rl.schedules import learning_rate_for_step
from gridworld_rl.trainer import train_model


@pytest.mark.parametrize(
    "schedule,warmup,floor,total,expected",
    [
        ("constant", 0, 0, 3, [1, 1, 1]),
        ("constant", 2, 0, 4, [0.5, 1, 1, 1]),
        ("cosine", 0, 0, 3, [1, 0.5, 0]),
        ("cosine", 0, 0.2, 3, [1, 0.6, 0.2]),
        ("cosine", 2, 0.2, 4, [0.5, 1, 0.6, 0.2]),
        ("cosine", 0, 0.2, 1, [1]),
        ("cosine", 0, 1, 3, [1, 1, 1]),
    ],
)
def test_hand_calculated_schedule(schedule, warmup, floor, total, expected):
    config = TrainingConfig(
        learning_rate=1.0,
        learning_rate_schedule=schedule,
        warmup_steps=warmup,
        min_learning_rate_ratio=floor,
    )
    assert [
        learning_rate_for_step(config, i, total) for i in range(total)
    ] == pytest.approx(expected)


@pytest.mark.parametrize("step,total", [(-1, 3), (3, 3), (0, 0)])
def test_invalid_update_index(step, total):
    with pytest.raises(ValueError, match="step must"):
        learning_rate_for_step(TrainingConfig(), step, total)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate_schedule", "unknown"),
        ("warmup_steps", True),
        ("warmup_steps", -1),
        ("warmup_steps", 1.5),
        ("min_learning_rate_ratio", -0.1),
        ("min_learning_rate_ratio", 1.1),
        ("min_learning_rate_ratio", float("nan")),
        ("min_learning_rate_ratio", True),
    ],
)
def test_invalid_schedule_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.from_dict({"training": {field: value}})


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_cli_schedule_controls(command):
    argv = [
        command,
        "--learning-rate-schedule",
        "cosine",
        "--warmup-steps",
        "2",
        "--min-learning-rate-ratio",
        "0.2",
    ]
    if command == "benchmark":
        argv.extend(["--seeds", "1", "2"])
    config = _apply_training_overrides(
        ExperimentConfig(), build_parser().parse_args(argv)
    )
    assert config.training.learning_rate_schedule == "cosine"
    assert config.training.warmup_steps == 2
    assert config.training.min_learning_rate_ratio == 0.2


def _dataset():
    return TransitionDataset(
        pd.DataFrame(
            {
                "state": [0, 1, 2, 3, 4],
                "action": [0, 1, 2, 3, 0],
                "reward": [0.1] * 5,
                "next_state": [1, 2, 3, 4, 5],
                "done": [False] * 5,
            }
        )
    )


def test_schedule_is_applied_at_optimizer_updates_and_recorded(tmp_path, monkeypatch):
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [4]},
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.01,
                "learning_rate_schedule": "cosine",
                "warmup_steps": 2,
                "min_learning_rate_ratio": 0.2,
            },
        }
    )
    rates = []
    original_step = torch.optim.Adam.step

    def recording_step(optimizer, *args, **kwargs):
        rates.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    _, metrics = train_model("dqn", _dataset(), config, torch.device("cpu"))
    assert rates == pytest.approx([0.005, 0.01, 0.006, 0.002])
    assert metrics["training_history"]["learning_rate"] == pytest.approx([0.01, 0.002])
    path = tmp_path / "config.json"
    config.to_json(path)
    assert ExperimentConfig.from_json(path).to_dict() == config.to_dict()


def test_excess_warmup_fails_before_model_creation(monkeypatch):
    config = ExperimentConfig()
    config.training.epochs = 1
    config.training.warmup_steps = 1

    def unexpected_model(*args, **kwargs):
        pytest.fail("Model constructed before schedule validation")

    monkeypatch.setattr("gridworld_rl.trainer.QNetwork", unexpected_model)
    with pytest.raises(ValueError, match="smaller than total"):
        train_model("dqn", _dataset(), config, torch.device("cpu"))
