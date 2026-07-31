from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gridworld_rl.algorithms import (
    calculate_loss,
    conservative_q_regularizer,
    double_dqn_targets,
    dqn_targets,
    epsilon_greedy_probabilities,
    expected_sarsa_targets,
)
from gridworld_rl.evaluation import classification_metrics
from gridworld_rl.models import QNetwork


def test_dqn_target_uses_target_max_and_masks_terminal_rows() -> None:
    targets = dqn_targets(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        next_target_q=torch.tensor([[1.0, 4.0], [100.0, 200.0]]),
        gamma=0.5,
    )

    assert torch.allclose(targets, torch.tensor([3.0, 2.0]))


def test_double_dqn_selects_online_and_evaluates_target() -> None:
    targets = double_dqn_targets(
        rewards=torch.zeros(2),
        dones=torch.zeros(2),
        next_online_q=torch.tensor([[0.0, 5.0], [8.0, 1.0]]),
        next_target_q=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        gamma=1.0,
    )

    assert torch.equal(targets, torch.tensor([20.0, 30.0]))


def test_epsilon_greedy_expected_sarsa_uses_a_normalized_distribution() -> None:
    online = torch.tensor([[9.0, 1.0, 0.0, 0.0]])
    target = torch.tensor([[10.0, 2.0, 4.0, 6.0]])
    probabilities = epsilon_greedy_probabilities(online, epsilon=0.2)
    targets = expected_sarsa_targets(
        rewards=torch.tensor([1.0]),
        dones=torch.tensor([0.0]),
        next_online_q=online,
        next_target_q=target,
        gamma=0.5,
        epsilon=0.2,
    )

    assert torch.allclose(probabilities, torch.tensor([[0.85, 0.05, 0.05, 0.05]]))
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert targets.item() == pytest.approx(5.55)


def test_cql_regularizer_penalizes_all_action_values_against_dataset_action() -> None:
    q_values = torch.zeros((3, 4), requires_grad=True)
    penalty = conservative_q_regularizer(q_values, torch.tensor([0, 1, 2]))
    penalty.backward()

    assert penalty.item() == pytest.approx(math.log(4))
    assert q_values.grad is not None


@pytest.mark.parametrize(
    ("algorithm", "expected_targets"),
    [
        ("dqn", torch.tensor([11.0, 2.0])),
        ("double_dqn", torch.tensor([6.0, 2.0])),
        ("expected_sarsa", torch.tensor([6.5, 2.0])),
    ],
)
def test_calculate_loss_dispatches_to_each_td_target(
    algorithm: str,
    expected_targets: torch.Tensor,
) -> None:
    online_q = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    actions = torch.tensor([0, 1])
    loss, parts = calculate_loss(
        algorithm,
        online_q,
        actions,
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        next_online_q=torch.tensor([[5.0, 0.0], [0.0, 1000.0]]),
        next_target_q=torch.tensor([[10.0, 20.0], [500.0, 900.0]]),
        gamma=0.5,
        epsilon=0.2,
        cql_alpha=0.7,
    )
    current_values = torch.tensor([1.0, 4.0])
    expected_loss = F.smooth_l1_loss(current_values, expected_targets)

    assert loss.item() == pytest.approx(expected_loss.item())
    assert parts["td_loss"] == pytest.approx(expected_loss.item())
    assert parts["cql_loss"] == 0.0
    loss.backward()
    assert online_q.grad is not None


def test_calculate_loss_adds_weighted_cql_penalty() -> None:
    online_q = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    actions = torch.tensor([0, 1])
    alpha = 0.7
    loss, parts = calculate_loss(
        "cql",
        online_q,
        actions,
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        next_online_q=torch.zeros((2, 2)),
        next_target_q=torch.tensor([[10.0, 20.0], [500.0, 900.0]]),
        gamma=0.5,
        epsilon=0.2,
        cql_alpha=alpha,
    )

    assert parts["total_loss"] == pytest.approx(
        parts["td_loss"] + alpha * parts["cql_loss"]
    )
    assert loss.item() == pytest.approx(parts["total_loss"])


def test_double_dqn_and_expected_sarsa_do_not_bootstrap_terminal_rows() -> None:
    rewards = torch.tensor([3.0])
    dones = torch.tensor([1.0])
    online = torch.tensor([[1000.0, -1000.0]])
    target = torch.tensor([[5000.0, 9000.0]])

    assert double_dqn_targets(
        rewards,
        dones,
        online,
        target,
        gamma=0.99,
    ).item() == pytest.approx(3.0)
    assert expected_sarsa_targets(
        rewards,
        dones,
        online,
        target,
        gamma=0.99,
        epsilon=0.1,
    ).item() == pytest.approx(3.0)


def test_classification_metrics_include_confusion_and_zero_support_recall() -> None:
    result = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 2]), num_actions=4
    )

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert result["per_action_recall"] == {
        "0": 0.5,
        "1": 0.5,
        "2": 0.0,
        "3": 0.0,
    }


def test_q_network_accepts_discrete_states() -> None:
    model = QNetwork(num_states=10, num_actions=4, hidden_sizes=[8])

    assert model(torch.tensor([0, 9])).shape == (2, 4)
