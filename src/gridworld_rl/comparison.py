"""Label-free, full-state comparisons of saved greedy policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .reproducibility import resolve_device, sha256_file


def compare_checkpoints(
    checkpoints: list[str | Path], *, device: str = "cpu", batch_size: int = 1024
) -> dict[str, Any]:
    """Compare greedy actions for every discrete state in bounded batches."""

    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    paths = [Path(path) for path in checkpoints]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Checkpoint paths must be distinct.")

    selected_device = resolve_device(device)
    fingerprints = [sha256_file(path) for path in paths]
    loaded = [load_checkpoint(path) for path in paths]
    models = [model.to(selected_device) for model, _ in loaded]
    num_states, num_actions = models[0].num_states, models[0].num_actions
    if any(
        model.num_states != num_states or model.num_actions != num_actions
        for model in models[1:]
    ):
        raise ValueError("Checkpoints must have identical state and action dimensions.")

    counts = [[0] * num_actions for _ in models]
    agreements = [[0] * len(models) for _ in models]
    unanimous = 0
    with torch.inference_mode():
        for start in range(0, num_states, batch_size):
            states = torch.arange(
                start, min(start + batch_size, num_states), device=selected_device
            )
            actions = []
            for index, model in enumerate(models):
                values = model(states)
                if not torch.isfinite(values).all():
                    raise ValueError(
                        f"Checkpoint {paths[index]} produces non-finite Q-values "
                        f"at states {start}..{start + len(states) - 1}."
                    )
                predicted = values.argmax(dim=1)
                actions.append(predicted)
                batch_counts = torch.bincount(predicted, minlength=num_actions).tolist()
                counts[index] = [
                    total + amount
                    for total, amount in zip(counts[index], batch_counts, strict=True)
                ]
            unanimous += int(torch.stack(actions).eq(actions[0]).all(dim=0).sum())
            for left in range(len(models)):
                for right in range(left + 1, len(models)):
                    agreements[left][right] += int(
                        (actions[left] == actions[right]).sum()
                    )

    for path, fingerprint in zip(paths, fingerprints, strict=True):
        if sha256_file(path) != fingerprint:
            raise ValueError(
                f"Checkpoint changed during comparison: {path}; retry with stable files."
            )

    return {
        "schema_version": 1,
        "num_states": num_states,
        "num_actions": num_actions,
        "tie_breaking": "lowest action index among exact maximum Q-values",
        "checkpoints": [
            {
                "path": str(path),
                "sha256": fingerprint,
                "algorithm": metadata["algorithm"],
                "action_counts": count,
            }
            for path, fingerprint, (_, metadata), count in zip(
                paths, fingerprints, loaded, counts, strict=True
            )
        ],
        "unanimous_states": unanimous,
        "disputed_states": num_states - unanimous,
        "pairs": [
            {
                "left": left,
                "right": right,
                "agree": agreements[left][right],
                "disagree": num_states - agreements[left][right],
                "agreement_rate": agreements[left][right] / num_states,
            }
            for left in range(len(models))
            for right in range(left + 1, len(models))
        ],
    }
