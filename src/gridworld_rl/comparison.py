"""Label-free, full-state comparisons of saved greedy policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .data import load_transition_csv
from .reproducibility import resolve_device, sha256_file


def compare_checkpoints(
    checkpoints: list[str | Path],
    *,
    device: str = "cpu",
    batch_size: int = 1024,
    train_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Compare greedy actions, optionally stratified by logged training support."""

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

    train_path = Path(train_csv) if train_csv is not None else None
    train_hash = sha256_file(train_path) if train_path is not None else None
    observed = np.zeros(num_states, dtype=bool) if train_path is not None else None
    logged_pairs: set[tuple[int, int]] = set()
    if train_path is not None:
        frame = load_transition_csv(
            train_path, num_states=num_states, num_actions=num_actions
        )
        state_ids = frame["state"].to_numpy(dtype=np.int64)
        action_ids = frame["action"].to_numpy(dtype=np.int64)
        observed[state_ids] = True
        logged_pairs = set(zip(state_ids, action_ids, strict=True))

    counts = [[0] * num_actions for _ in models]
    agreements = [[0] * len(models) for _ in models]
    observed_agreements = [[0] * len(models) for _ in models]
    supported_counts = [0] * len(models)
    observed_count = int(observed.sum()) if observed is not None else 0
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
            if observed is not None:
                state_batch = np.arange(start, start + len(states))
                seen = torch.as_tensor(observed[state_batch], device=selected_device)
                for index, predicted in enumerate(actions):
                    supported_counts[index] += sum(
                        (int(state), int(action)) in logged_pairs
                        for state, action in zip(
                            state_batch[observed[state_batch]],
                            predicted[seen].cpu().tolist(),
                            strict=True,
                        )
                    )
            for left in range(len(models)):
                for right in range(left + 1, len(models)):
                    equal = actions[left] == actions[right]
                    agreements[left][right] += int(equal.sum())
                    if observed is not None:
                        observed_agreements[left][right] += int(equal[seen].sum())

    for path, fingerprint in zip(paths, fingerprints, strict=True):
        if sha256_file(path) != fingerprint:
            raise ValueError(
                f"Checkpoint changed during comparison: {path}; retry with stable files."
            )
    if train_path is not None and sha256_file(train_path) != train_hash:
        raise ValueError(
            f"Training CSV changed during comparison: {train_path}; "
            "retry with stable files."
        )

    result = {
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
    if train_path is not None:
        result["training_support"] = {
            "train_path": str(train_path),
            "train_sha256": train_hash,
            "observed_states": observed_count,
            "unobserved_states": num_states - observed_count,
            "checkpoints": [
                {
                    "supported_observed_states": count,
                    "unsupported_observed_states": observed_count - count,
                    "observed_state_support_rate": count / observed_count,
                }
                for count in supported_counts
            ],
            "pairs": [
                {
                    "left": left,
                    "right": right,
                    "observed_agree": observed_agreements[left][right],
                    "observed_agreement_rate": (
                        observed_agreements[left][right] / observed_count
                    ),
                    "unobserved_agree": (
                        agreements[left][right] - observed_agreements[left][right]
                    ),
                    "unobserved_agreement_rate": (
                        (agreements[left][right] - observed_agreements[left][right])
                        / (num_states - observed_count)
                        if observed_count < num_states
                        else None
                    ),
                }
                for left in range(len(models))
                for right in range(left + 1, len(models))
            ],
        }
    return result
