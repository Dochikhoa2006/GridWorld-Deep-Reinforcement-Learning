"""Portable loading for checkpoints produced by the trainer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from .config import SUPPORTED_ALGORITHMS
from .models import QNetwork
from .reproducibility import sha256_file


def load_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[QNetwork, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}") from exc
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
    version = payload.get("format_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported checkpoint format_version; expected 1.")
    if payload["algorithm"] not in SUPPORTED_ALGORITHMS:
        raise ValueError("Checkpoint contains an unsupported algorithm.")
    for field in ("num_states", "num_actions"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
            raise ValueError(f"Checkpoint {field} must be an integer greater than 1.")
    network = payload["network"]
    hidden_sizes = network.get("hidden_sizes") if isinstance(network, dict) else None
    if (
        not isinstance(hidden_sizes, (list, tuple))
        or not hidden_sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in hidden_sizes
        )
    ):
        raise ValueError(
            "Checkpoint network.hidden_sizes must contain positive integers."
        )
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) for name in state_dict
    ):
        raise ValueError("Checkpoint model_state_dict must map layer names to tensors.")
    model = QNetwork(
        num_states=payload["num_states"],
        num_actions=payload["num_actions"],
        hidden_sizes=hidden_sizes,
    )
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "Checkpoint model_state_dict layer mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for name, reference in expected.items():
        value = state_dict[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(
                f"Checkpoint model_state_dict {name} must be a dense tensor."
            )
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(
                f"Checkpoint model_state_dict {name} has shape/dtype "
                f"{tuple(value.shape)}/{value.dtype}; expected "
                f"{tuple(reference.shape)}/{reference.dtype}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Checkpoint model_state_dict {name} must contain finite tensors."
            )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate a checkpoint and return a small, stable metadata summary."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    fingerprint = sha256_file(checkpoint_path)
    model, payload = load_checkpoint(checkpoint_path)
    if sha256_file(checkpoint_path) != fingerprint:
        raise ValueError(
            "Checkpoint changed during inspection; retry with a stable file."
        )
    for field in ("seed", "global_steps"):
        value = payload.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"Checkpoint {field} must be a non-negative integer.")
    return {
        "schema_version": 1,
        "checkpoint_sha256": fingerprint,
        "format_version": payload.get("format_version", 1),
        "algorithm": payload["algorithm"],
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "hidden_sizes": list(payload["network"]["hidden_sizes"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": payload.get("seed"),
        "global_steps": payload.get("global_steps"),
    }
