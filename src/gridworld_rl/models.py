"""Neural-network models used by all baselines."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """MLP mapping a discrete state to one Q-value per action."""

    def __init__(
        self,
        num_states: int = 100,
        num_actions: int = 4,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        if num_states <= 1 or num_actions <= 1:
            raise ValueError("num_states and num_actions must be greater than 1.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")
        self.num_states = num_states
        self.num_actions = num_actions

        layers: list[nn.Module] = []
        input_size = num_states
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            states = F.one_hot(states.long(), num_classes=self.num_states).float()
        elif states.shape[-1] != self.num_states:
            raise ValueError(
                f"Expected integer states or vectors of size {self.num_states}; "
                f"received shape {tuple(states.shape)}."
            )
        return self.network(states.float())
