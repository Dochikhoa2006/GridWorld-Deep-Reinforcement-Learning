"""Compare two validated offline transition logs without training models."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from .data import (
    NUM_ACTIONS,
    NUM_STATES,
    REQUIRED_COLUMNS,
    load_transition_csv,
    transition_diagnostics,
)
from .reproducibility import sha256_file


def _jensen_shannon_bits(left: Counter[Any], right: Counter[Any]) -> float:
    """Jensen-Shannon divergence in bits, bounded by zero and one."""

    left_total = sum(left.values())
    right_total = sum(right.values())
    divergence = 0.0
    for key in left.keys() | right.keys():
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return max(0.0, min(1.0, divergence))


def _overlap(left: set[Any], right: set[Any]) -> dict[str, int | float]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "shared": intersection,
        "left_only": len(left) - intersection,
        "right_only": len(right) - intersection,
        "jaccard": intersection / union,
    }


def compare_datasets(
    left_csv: str | Path,
    right_csv: str | Path,
    *,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, Any]:
    """Quantify coverage and frequency shift between two transition CSVs."""

    if (
        type(num_states) is not int
        or num_states <= 1
        or type(num_actions) is not int
        or num_actions <= 1
    ):
        raise ValueError("num_states and num_actions must be integers greater than 1.")
    paths = {"left": Path(left_csv), "right": Path(right_csv)}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    left = load_transition_csv(
        paths["left"], num_states=num_states, num_actions=num_actions
    )
    right = load_transition_csv(
        paths["right"], num_states=num_states, num_actions=num_actions
    )
    left_states = set(left["state"])
    right_states = set(right["state"])
    left_pairs = Counter(
        left.loc[:, ["state", "action"]].itertuples(index=False, name=None)
    )
    right_pairs = Counter(
        right.loc[:, ["state", "action"]].itertuples(index=False, name=None)
    )
    left_transitions = set(
        left.loc[:, list(REQUIRED_COLUMNS)].itertuples(index=False, name=None)
    )
    right_transitions = set(
        right.loc[:, list(REQUIRED_COLUMNS)].itertuples(index=False, name=None)
    )
    left_actions = Counter(left["action"])
    right_actions = Counter(right["action"])
    result = {
        "schema_version": 1,
        "num_states": num_states,
        "num_actions": num_actions,
        "inputs": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "left": {
            "rows": len(left),
            "diagnostics": transition_diagnostics(
                left, num_states=num_states, num_actions=num_actions
            ),
        },
        "right": {
            "rows": len(right),
            "diagnostics": transition_diagnostics(
                right, num_states=num_states, num_actions=num_actions
            ),
        },
        "overlap": {
            "states": _overlap(left_states, right_states),
            "state_action_pairs": _overlap(set(left_pairs), set(right_pairs)),
            "full_transitions": _overlap(left_transitions, right_transitions),
            "right_rows_at_new_states": int((~right["state"].isin(left_states)).sum()),
            "right_rows_at_new_state_action_pairs": sum(
                count for pair, count in right_pairs.items() if pair not in left_pairs
            ),
        },
        "distribution_shift": {
            "action_js_divergence_bits": _jensen_shannon_bits(
                left_actions, right_actions
            ),
            "state_action_js_divergence_bits": _jensen_shannon_bits(
                left_pairs, right_pairs
            ),
        },
    }
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(
                f"{name.capitalize()} CSV changed during dataset comparison: "
                f"{path}; retry with stable files."
            )
    return result
