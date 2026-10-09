"""Export a greedy policy constrained to actions observed in logged data."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .data import load_transition_csv
from .reproducibility import resolve_device, sha256_file


def export_supported_policy(
    checkpoint: str | Path,
    train_csv: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Path:
    """Choose the best logged action at observed states; use greedy fallback elsewhere."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Policy output already exists: {destination}")
    checkpoint_path = Path(checkpoint)
    train_path = Path(train_csv)
    checkpoint_hash = sha256_file(checkpoint_path)
    train_hash = sha256_file(train_path)
    model, metadata = load_checkpoint(checkpoint_path)
    frame = load_transition_csv(
        train_path, num_states=model.num_states, num_actions=model.num_actions
    )
    support = {
        int(state): sorted(set(actions))
        for state, actions in frame.groupby("state")["action"]
    }
    selected_device = resolve_device(device)
    model.to(selected_device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    changed = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            metadata_doc = {
                "schema_version": 1,
                "algorithm": metadata["algorithm"],
                "checkpoint_sha256": checkpoint_hash,
                "train_sha256": train_hash,
                "num_states": model.num_states,
                "num_actions": model.num_actions,
                "selection_rule": "highest Q-value among logged actions; lowest action index on ties",
                "unobserved_fallback": "unconstrained greedy action",
            }
            temporary.write("{\n")
            for key, value in sorted(metadata_doc.items()):
                temporary.write(f"  {json.dumps(key)}: ")
                json.dump(value, temporary, allow_nan=False)
                temporary.write(",\n")
            temporary.write('  "policy": [\n')
            with torch.inference_mode():
                for start in range(0, model.num_states, batch_size):
                    states = torch.arange(
                        start,
                        min(start + batch_size, model.num_states),
                        device=selected_device,
                    )
                    values = model(states)
                    if not torch.isfinite(values).all():
                        raise ValueError("Checkpoint produces non-finite Q-values.")
                    for index, q_values in enumerate(values.cpu().tolist()):
                        state = start + index
                        logged_actions = support.get(state, [])
                        unrestricted = max(
                            range(model.num_actions), key=q_values.__getitem__
                        )
                        action = (
                            max(logged_actions, key=q_values.__getitem__)
                            if logged_actions
                            else unrestricted
                        )
                        changed += action != unrestricted
                        if state:
                            temporary.write(",\n")
                        json.dump(
                            {
                                "state": state,
                                "action": action,
                                "unconstrained_action": unrestricted,
                                "logged_actions": logged_actions,
                                "was_constrained": action != unrestricted,
                                "q_value": q_values[action],
                            },
                            temporary,
                            sort_keys=True,
                            allow_nan=False,
                        )
            temporary.write('\n  ],\n  "summary": ')
            json.dump(
                {
                    "observed_states": len(support),
                    "unobserved_states": model.num_states - len(support),
                    "changed_observed_states": changed,
                },
                temporary,
                sort_keys=True,
            )
            temporary.write("\n}\n")
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise ValueError(
                "Checkpoint changed during supported-policy export; retry with a stable file."
            )
        if sha256_file(train_path) != train_hash:
            raise ValueError(
                "Training CSV changed during supported-policy export; retry with a stable file."
            )
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
