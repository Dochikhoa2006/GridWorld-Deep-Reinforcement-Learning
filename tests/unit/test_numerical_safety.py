from __future__ import annotations

import pandas as pd
import pytest
import torch

import gridworld_rl.trainer as trainer
from gridworld_rl.config import ExperimentConfig
from gridworld_rl.data import TransitionDataset
from gridworld_rl.evaluation import predict_actions
from gridworld_rl.models import QNetwork


def _training_inputs():
    dataset = TransitionDataset(
        pd.DataFrame(
            {
                "state": [0, 1, 2],
                "action": [0, 1, 2],
                "reward": [0.1, 0.2, 0.3],
                "next_state": [1, 2, 3],
                "done": [False, False, True],
            }
        )
    )
    config = ExperimentConfig.from_dict(
        {
            "network": {"hidden_sizes": [4]},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
            },
        }
    )
    return dataset, config


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "stage", ["Q-values", "loss", "loss metrics", "gradient", "parameters"]
)
def test_training_stops_on_nonfinite_values(monkeypatch, invalid, stage):
    dataset, config = _training_inputs()
    original_loss = trainer.calculate_loss
    original_forward = QNetwork.forward
    original_step = torch.optim.Adam.step
    updates = []

    def corrupt_forward(model, states):
        result = original_forward(model, states)
        return result * invalid

    def corrupt_loss(*args, **kwargs):
        loss, parts = original_loss(*args, **kwargs)
        if stage == "loss":
            loss = loss * invalid
        elif stage == "loss metrics":
            parts["td_loss"] = invalid
        elif stage == "gradient":
            loss.register_hook(lambda grad: torch.full_like(grad, invalid))
        return loss, parts

    def recording_step(optimizer, *args, **kwargs):
        updates.append(1)
        result = original_step(optimizer, *args, **kwargs)
        if stage == "parameters":
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(invalid)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    if stage == "Q-values":
        monkeypatch.setattr(QNetwork, "forward", corrupt_forward)
    else:
        monkeypatch.setattr(trainer, "calculate_loss", corrupt_loss)
    with pytest.raises(RuntimeError) as exc:
        trainer.train_model("dqn", dataset, config, torch.device("cpu"))
    message = str(exc.value)
    assert "dqn training (epoch 1, batch" in message
    assert "optimizer update 1" in message
    if stage == "gradient":
        assert "Gradient clipping failed" in message
        assert "batch 2" in message  # Accumulated gradients checked before the update.
    else:
        assert "Non-finite" in message
    assert len(updates) == (1 if stage == "parameters" else 0)


@pytest.mark.parametrize("training", [True, False])
def test_prediction_rejects_nonfinite_values_and_restores_mode(training):
    class InvalidModel(torch.nn.Module):
        def forward(self, states):
            return torch.full((len(states), 4), float("nan"))

    model = InvalidModel().train(training)
    with pytest.raises(ValueError, match=r"rows 0\.\.1"):
        predict_actions(
            model, torch.tensor([0, 1, 2]), device=torch.device("cpu"), batch_size=2
        )
    assert model.training is training


def test_json_rejects_nonfinite_metrics_without_touching_existing_file(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("preserve")
    with pytest.raises(ValueError):
        trainer._write_json(path, {"loss": float("nan")})
    assert path.read_text() == "preserve"
