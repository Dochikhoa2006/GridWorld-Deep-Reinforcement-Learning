"""Portable loading for checkpoints produced by the trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import QNetwork


def load_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[QNetwork, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {
        "algorithm",
        "model_state_dict",
        "network",
        "num_states",
        "num_actions",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required - set(payload if isinstance(payload, dict) else ()))
        raise ValueError(f"Invalid checkpoint {checkpoint_path}; missing: {missing}")
    model = QNetwork(
        num_states=int(payload["num_states"]),
        num_actions=int(payload["num_actions"]),
        hidden_sizes=payload["network"]["hidden_sizes"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
