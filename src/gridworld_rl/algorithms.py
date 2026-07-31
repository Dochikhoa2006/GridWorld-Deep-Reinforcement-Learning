"""Correct, independently testable offline TD targets and losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _bootstrap(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    return rewards + gamma * (1.0 - dones.float()) * values


def dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_target_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Bellman target using the maximum target-network value."""

    return _bootstrap(rewards, dones, next_target_q.max(dim=1).values, gamma)


def double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Select with the online network and evaluate with the target network."""

    selected_actions = next_online_q.argmax(dim=1, keepdim=True)
    selected_values = next_target_q.gather(1, selected_actions).squeeze(1)
    return _bootstrap(rewards, dones, selected_values, gamma)


def epsilon_greedy_probabilities(
    q_values: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Return an epsilon-greedy action distribution for each Q-value row."""

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1.")
    num_actions = q_values.shape[1]
    probabilities = torch.full_like(q_values, epsilon / num_actions)
    greedy_actions = q_values.argmax(dim=1, keepdim=True)
    probabilities.scatter_add_(
        1,
        greedy_actions,
        torch.full(
            (q_values.shape[0], 1),
            1.0 - epsilon,
            dtype=q_values.dtype,
            device=q_values.device,
        ),
    )
    return probabilities


def expected_sarsa_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> torch.Tensor:
    """Expected SARSA target under the online network's epsilon-greedy policy."""

    probabilities = epsilon_greedy_probabilities(next_online_q, epsilon)
    expected_values = (probabilities * next_target_q).sum(dim=1)
    return _bootstrap(rewards, dones, expected_values, gamma)


def conservative_q_regularizer(
    q_values: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Discrete CQL(H) penalty: logsumexp over actions minus dataset Q-value."""

    dataset_values = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return (torch.logsumexp(q_values, dim=1) - dataset_values).mean()


def calculate_loss(
    algorithm: str,
    online_q: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    gamma: float,
    epsilon: float,
    cql_alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculate Huber TD loss, optionally augmented by discrete CQL."""

    current_values = online_q.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    if algorithm in {"dqn", "cql"}:
        targets = dqn_targets(rewards, dones, next_target_q, gamma)
    elif algorithm == "double_dqn":
        targets = double_dqn_targets(
            rewards, dones, next_online_q, next_target_q, gamma
        )
    elif algorithm == "expected_sarsa":
        targets = expected_sarsa_targets(
            rewards,
            dones,
            next_online_q,
            next_target_q,
            gamma,
            epsilon,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    td_loss = F.smooth_l1_loss(current_values, targets)
    cql_loss = (
        conservative_q_regularizer(online_q, actions)
        if algorithm == "cql"
        else online_q.new_zeros(())
    )
    total_loss = td_loss + cql_alpha * cql_loss
    return total_loss, {
        "td_loss": float(td_loss.detach().cpu()),
        "cql_loss": float(cql_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
