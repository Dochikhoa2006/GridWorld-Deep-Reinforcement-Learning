from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gridworld_rl.evaluation import classification_metrics, predict_actions


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_lengths: list[int] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batch_lengths.append(len(states))
        return torch.stack((states, -states), dim=1)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("tensor_input", [True, False])
def test_predictions_preserve_mode_and_batch_order(
    training: bool, tensor_input: bool
) -> None:
    model = RecordingModel()
    model.train(training)
    states = np.array([1, -1, 2, -2, 0])
    inputs = torch.tensor(states) if tensor_input else states

    result = predict_actions(model, inputs, device=torch.device("cpu"), batch_size=2)

    np.testing.assert_array_equal(result, [0, 1, 0, 1, 0])
    assert result.dtype == np.int64
    assert model.batch_lengths == [2, 2, 1]
    assert model.training is training


def test_empty_predictions_preserve_training_mode() -> None:
    model = RecordingModel()
    result = predict_actions(model, np.array([]), device=torch.device("cpu"))
    assert result.shape == (0,)
    assert result.dtype == np.int64
    assert model.training
    assert model.batch_lengths == []


@pytest.mark.parametrize("field", ["targets", "predictions"])
@pytest.mark.parametrize("labels", [[0.5], [np.nan], [np.inf], ["0"], [True]])
def test_metrics_reject_invalid_labels(field: str, labels: list) -> None:
    inputs = {"targets": np.array([0]), "predictions": np.array([0])}
    inputs[field] = np.array(labels)
    with pytest.raises(ValueError, match=field):
        classification_metrics(**inputs)


@pytest.mark.parametrize("num_actions", [0, -1, True, 2.5])
def test_metrics_reject_invalid_action_count(num_actions: int) -> None:
    with pytest.raises(ValueError, match="num_actions must be a positive integer"):
        classification_metrics(np.array([0]), np.array([0]), num_actions=num_actions)


def test_metrics_accept_integer_valued_floats() -> None:
    result = classification_metrics(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert result["accuracy"] == 0.5
    assert result["per_action_support"] == {"0": 1, "1": 1, "2": 0, "3": 0}


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_predictions_reject_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        predict_actions(
            RecordingModel(),
            np.array([0]),
            device=torch.device("cpu"),
            batch_size=batch_size,
        )


def test_predictions_restore_mode_after_inference_failure() -> None:
    class FailingModel(nn.Module):
        def forward(self, states: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("inference failed")

    model = FailingModel()
    with pytest.raises(RuntimeError, match="inference failed"):
        predict_actions(model, torch.tensor([0]), device=torch.device("cpu"))
    assert model.training
