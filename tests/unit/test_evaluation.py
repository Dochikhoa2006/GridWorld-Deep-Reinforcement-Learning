from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from gridworld_rl.evaluation import (
    classification_metrics,
    overlap_sliced_agreement,
    policy_agreement_matrix,
    predict_actions,
)


def test_policy_agreement_matrix_is_symmetric_and_label_free():
    result = policy_agreement_matrix(
        {
            "dqn": [0, 1, 2, 3],
            "cql": [0, 2, 2, 1],
            "double_dqn": [1, 1, 2, 3],
        },
        num_actions=4,
    )
    assert result == {
        "algorithms": ["dqn", "cql", "double_dqn"],
        "num_examples": 4,
        "agreement": [[1.0, 0.5, 0.75], [0.5, 1.0, 0.25], [0.75, 0.25, 1.0]],
        "disagreements": [[0, 2, 1], [2, 0, 3], [1, 3, 0]],
    }


def test_balanced_accuracy_averages_recall_of_supported_actions():
    result = classification_metrics(
        np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), num_actions=3
    )
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)


@pytest.mark.parametrize(
    "predictions",
    [
        {},
        {"dqn": []},
        {"dqn": [0, 1], "cql": [0]},
        {"dqn": [0.0]},
        {"dqn": [True]},
        {"dqn": [-1]},
        {"dqn": [4]},
    ],
)
def test_policy_agreement_rejects_invalid_predictions(predictions):
    with pytest.raises(ValueError):
        policy_agreement_matrix(predictions, num_actions=4)


def test_overlap_slices_partition_rows_and_weight_agreement():
    train = pd.DataFrame(
        {
            "state": [0, 1],
            "action": [0, 1],
            "reward": [1.0, 2.0],
            "next_state": [1, 2],
            "done": [False, True],
        }
    )
    solution = pd.DataFrame(
        {
            "state": [0, 0, 1, 1, 2, 0],
            "action": [0, 0, 1, 0, 1, 0],
            "reward": [1.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            "next_state": [1, 9, 2, 2, 3, 1],
            "done": [False, False, True, True, False, False],
        }
    )
    targets = solution["action"].to_numpy()
    predictions = np.array([0, 1, 0, 0, 1, 0])
    result = overlap_sliced_agreement(train, solution, targets, predictions)
    assert result == {
        "exact_training_transition": {"rows": 3, "accuracy": 2 / 3},
        "seen_state_new_transition": {"rows": 2, "accuracy": 0.5},
        "unseen_state": {"rows": 1, "accuracy": 1.0},
    }
    assert (
        sum(group["rows"] * group["accuracy"] for group in result.values())
        / len(targets)
        == 4 / 6
    )


def test_empty_overlap_group_has_null_accuracy():
    train = pd.DataFrame(
        {
            "state": [0],
            "action": [0],
            "reward": [0.0],
            "next_state": [1],
            "done": [False],
        }
    )
    result = overlap_sliced_agreement(train, train.copy(), np.array([0]), np.array([0]))
    assert result["seen_state_new_transition"] == {"rows": 0, "accuracy": None}
    assert result["unseen_state"] == {"rows": 0, "accuracy": None}
    with pytest.raises(ValueError, match="must align"):
        overlap_sliced_agreement(train, train, np.array([0]), np.array([0, 1]))


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_lengths: list[int] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        assert not self.training
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        self.batch_lengths.append(len(states))
        return torch.stack((states, -states), dim=1)


def test_numpy_inference_converts_only_one_batch_at_a_time(monkeypatch):
    states = np.arange(25, dtype=np.int64)
    states.flags.writeable = False
    original_tensor = torch.tensor
    copied_lengths = []

    def recording_tensor(data, *args, **kwargs):
        copied_lengths.append(len(data))
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", recording_tensor)
    result = predict_actions(
        RecordingModel(), states, device=torch.device("cpu"), batch_size=7
    )
    assert copied_lengths == [7, 7, 7, 4]
    np.testing.assert_array_equal(result, np.zeros(25, dtype=np.int64))


def test_tensor_inference_converts_dtype_per_batch():
    class DtypeModel(RecordingModel):
        def forward(self, states):
            assert states.dtype == torch.long
            return super().forward(states)

    model = DtypeModel()
    result = predict_actions(
        model,
        torch.arange(9, dtype=torch.float32),
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert model.batch_lengths == [4, 4, 1]
    np.testing.assert_array_equal(result, np.zeros(9, dtype=np.int64))


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


@pytest.mark.parametrize(
    "malformation", ["vector", "rows", "actions", "non_tensor", "changed_actions"]
)
def test_prediction_rejects_malformed_model_output_and_restores_mode(
    malformation: str,
) -> None:
    class MalformedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, states: torch.Tensor):
            self.calls += 1
            if malformation == "vector":
                return torch.zeros(len(states))
            if malformation == "rows":
                return torch.zeros(len(states) + 1, 2)
            if malformation == "actions":
                return torch.zeros(len(states), 1)
            if malformation == "non_tensor":
                return [[0.0, 1.0]] * len(states)
            return torch.zeros(len(states), 2 if self.calls == 1 else 3)

    model = MalformedModel()
    with pytest.raises(ValueError, match=r"consistent \[batch, actions\] Q-values"):
        predict_actions(
            model, np.array([0, 1, 2]), device=torch.device("cpu"), batch_size=2
        )
    assert model.training


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


def test_metrics_accept_tensors_that_track_gradients() -> None:
    predictions = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    result = classification_metrics(torch.tensor([0, 1, 1]), predictions)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == 0.75
    assert predictions.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_metrics_accept_cuda_tensors() -> None:
    result = classification_metrics(
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([0, 1], device="cuda"),
    )
    assert result["accuracy"] == 1.0


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
