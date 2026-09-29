"""Learning-rate schedules indexed by zero-based optimizer update."""

from __future__ import annotations

import math

from .config import TrainingConfig


def learning_rate_for_step(
    config: TrainingConfig, step: int, total_steps: int
) -> float:
    """Return the rate used for an update, with optional linear warmup.

    Without warmup, cosine decay includes both endpoints (unless only one
    update exists). After warmup reaches the base rate, each remaining update
    advances decay toward the configured floor, reaching it on the last update.
    """

    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("step must identify an optimizer update within total_steps.")
    if config.warmup_steps >= total_steps:
        raise ValueError(
            "training.warmup_steps must be smaller than total optimizer updates."
        )
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    if config.warmup_steps:
        progress = (step - config.warmup_steps + 1) / (
            total_steps - config.warmup_steps
        )
    else:
        progress = step / max(total_steps - 1, 1)
    multiplier = (
        config.min_learning_rate_ratio
        + (1 - config.min_learning_rate_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )
    return config.learning_rate * multiplier
