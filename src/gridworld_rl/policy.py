"""Export complete discrete policies from saved model checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .reproducibility import resolve_device, sha256_file


def export_policy(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Path:
    """Export all states in ascending order, publishing without overwriting files."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Policy output already exists: {destination}")
    selected_device = resolve_device(device)
    # Load on CPU before moving to the requested inference device.
    fingerprint = sha256_file(checkpoint)
    model, metadata = load_checkpoint(checkpoint)
    model.to(selected_device)
    metadata = {
        "schema_version": 1,
        "algorithm": metadata["algorithm"],
        "checkpoint_sha256": fingerprint,
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "device": str(selected_device),
        "tie_breaking": "lowest action index among exact maximum Q-values",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
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
            # Write the metadata once, then stream policy rows in state order.
            temporary.write("{\n")
            for key, value in sorted(metadata.items()):
                temporary.write(f"  {json.dumps(key)}: ")
                json.dump(value, temporary, sort_keys=True, allow_nan=False)
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
                    actions = values.argmax(dim=1)
                    best = values.topk(2, dim=1).values
                    best_cpu = best.cpu().double()
                    gaps = (best_cpu[:, 0] - best_cpu[:, 1]).tolist()
                    ties = (values == best[:, :1]).sum(dim=1).cpu().tolist()
                    for index, (q_values, action, gap, count) in enumerate(
                        zip(
                            values.cpu().tolist(),
                            actions.cpu().tolist(),
                            gaps,
                            ties,
                            strict=True,
                        )
                    ):
                        if start + index:
                            temporary.write(",\n")
                        json.dump(
                            {
                                "state": start + index,
                                "action": action,
                                "q_values": q_values,
                                "action_gap": gap,
                                "num_greedy_actions": count,
                            },
                            temporary,
                            sort_keys=True,
                            allow_nan=False,
                        )
            temporary.write("\n  ]\n}\n")
        if sha256_file(checkpoint) != fingerprint:
            raise ValueError(
                "Checkpoint changed during policy export; retry with a stable file."
            )
        # An atomic no-clobber publication, including competing exporters.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
