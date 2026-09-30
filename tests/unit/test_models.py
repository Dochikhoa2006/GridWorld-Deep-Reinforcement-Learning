from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from gridworld_rl.models import QNetwork


@pytest.mark.parametrize("shape", [(7,), (2, 3), ()])
@pytest.mark.parametrize("hidden_sizes", [[8], [8, 5]])
def test_integer_state_path_matches_dense_one_hot_forward_and_gradients(
    shape, hidden_sizes
):
    torch.manual_seed(11)
    model = QNetwork(num_states=13, num_actions=4, hidden_sizes=hidden_sizes)
    reference = copy.deepcopy(model)
    states = torch.randint(0, 13, shape)
    dense_states = F.one_hot(states, num_classes=13).float()

    actual = model(states)
    expected = reference(dense_states)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, atol=1e-5, rtol=1e-5
        )


def test_integer_state_inference_does_not_build_one_hot_tensor(monkeypatch):
    model = QNetwork(num_states=100_000, num_actions=4, hidden_sizes=[8])

    def unexpected_one_hot(*args, **kwargs):
        pytest.fail("Integer inference allocated a one-hot tensor")

    monkeypatch.setattr("gridworld_rl.models.F.one_hot", unexpected_one_hot)
    assert model(torch.tensor([0, 99_999, 0])).shape == (3, 4)


def test_vector_state_path_and_checkpoint_layout_remain_compatible():
    model = QNetwork(num_states=9, num_actions=3, hidden_sizes=[6])
    states = torch.tensor([2, 7])
    dense = F.one_hot(states, num_classes=9).float()
    torch.testing.assert_close(model(states), model(dense))
    assert list(model.state_dict()) == [
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
    ]
