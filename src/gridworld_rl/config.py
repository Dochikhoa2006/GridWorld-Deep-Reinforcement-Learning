"""Typed experiment configuration with JSON serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_ALGORITHMS = ("dqn", "double_dqn", "expected_sarsa", "cql")


@dataclass
class DatasetConfig:
    train: str = "Gridworld-10_Dataset/train.csv"
    eval_challenge: str = "Gridworld-10_Dataset/eval_challenge.csv"
    eval_solution: str = "Gridworld-10_Dataset/eval_solution.csv"
    num_states: int = 100
    num_actions: int = 4


@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])


@dataclass
class TrainingConfig:
    algorithms: list[str] = field(default_factory=lambda: list(SUPPORTED_ALGORITHMS))
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 128
    gamma: float = 0.99
    epsilon: float = 0.1
    cql_alpha: float = 1.0
    target_update_interval: int = 250
    gradient_clip_norm: float = 10.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


@dataclass
class OutputConfig:
    directory: str = "artifacts"
    run_name: str = "latest"
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        allowed = {"dataset", "network", "training", "output"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration section(s): {sorted(unknown)}")
        try:
            config = cls(
                dataset=DatasetConfig(**values.get("dataset", {})),
                network=NetworkConfig(**values.get("network", {})),
                training=TrainingConfig(**values.get("training", {})),
                output=OutputConfig(**values.get("output", {})),
            )
        except TypeError as exc:
            raise ValueError(f"Invalid configuration field: {exc}") from exc
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in configuration file {config_path}: {exc}"
            ) from exc
        if not isinstance(values, dict):
            raise ValueError("The configuration root must be a JSON object.")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def validate(self) -> None:
        t = self.training
        if any(
            not isinstance(path, str) or not path
            for path in (
                self.dataset.train,
                self.dataset.eval_challenge,
                self.dataset.eval_solution,
            )
        ):
            raise ValueError("dataset paths must be non-empty strings.")
        if (
            isinstance(self.dataset.num_states, bool)
            or not isinstance(self.dataset.num_states, int)
            or self.dataset.num_states <= 1
        ):
            raise ValueError("dataset.num_states must be greater than 1.")
        if (
            isinstance(self.dataset.num_actions, bool)
            or not isinstance(self.dataset.num_actions, int)
            or self.dataset.num_actions <= 1
        ):
            raise ValueError("dataset.num_actions must be greater than 1.")
        if not self.network.hidden_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.network.hidden_sizes
        ):
            raise ValueError("network.hidden_sizes must contain positive integers.")
        if not isinstance(t.algorithms, list) or not all(
            isinstance(algorithm, str) for algorithm in t.algorithms
        ):
            raise ValueError("training.algorithms must be a list of names.")
        unknown = set(t.algorithms) - set(SUPPORTED_ALGORITHMS)
        if unknown:
            raise ValueError(
                f"Unsupported algorithm(s): {sorted(unknown)}. "
                f"Choose from {list(SUPPORTED_ALGORITHMS)}."
            )
        if not t.algorithms:
            raise ValueError("training.algorithms cannot be empty.")
        if len(set(t.algorithms)) != len(t.algorithms):
            raise ValueError("training.algorithms cannot contain duplicates.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                t.epochs,
                t.batch_size,
                t.target_update_interval,
            )
        ):
            raise ValueError(
                "training.epochs, training.batch_size, and "
                "training.target_update_interval must be positive integers."
            )
        if (
            isinstance(t.learning_rate, bool)
            or not isinstance(t.learning_rate, (int, float))
            or not math.isfinite(t.learning_rate)
            or t.learning_rate <= 0
        ):
            raise ValueError("training.learning_rate must be positive and finite.")
        if (
            isinstance(t.gamma, bool)
            or not isinstance(t.gamma, (int, float))
            or not math.isfinite(t.gamma)
            or not 0 <= t.gamma <= 1
        ):
            raise ValueError("training.gamma must be between 0 and 1.")
        if (
            isinstance(t.epsilon, bool)
            or not isinstance(t.epsilon, (int, float))
            or not math.isfinite(t.epsilon)
            or not 0 <= t.epsilon <= 1
        ):
            raise ValueError("training.epsilon must be between 0 and 1.")
        if (
            isinstance(t.cql_alpha, bool)
            or not isinstance(t.cql_alpha, (int, float))
            or not math.isfinite(t.cql_alpha)
            or t.cql_alpha < 0
        ):
            raise ValueError("training.cql_alpha must be non-negative and finite.")
        if (
            isinstance(t.gradient_clip_norm, bool)
            or not isinstance(t.gradient_clip_norm, (int, float))
            or not math.isfinite(t.gradient_clip_norm)
            or t.gradient_clip_norm <= 0
        ):
            raise ValueError("training.gradient_clip_norm must be positive and finite.")
        if (
            isinstance(t.num_workers, bool)
            or not isinstance(t.num_workers, int)
            or t.num_workers < 0
        ):
            raise ValueError("training.num_workers cannot be negative.")
        if isinstance(t.seed, bool) or not isinstance(t.seed, int) or t.seed < 0:
            raise ValueError("training.seed must be a non-negative integer.")
        if not isinstance(t.device, str) or t.device not in {
            "auto",
            "cpu",
            "cuda",
            "mps",
        }:
            raise ValueError("training.device must be auto, cpu, cuda, or mps.")
        if any(
            not isinstance(path, str) or not path
            for path in (self.output.directory, self.output.run_name)
        ):
            raise ValueError(
                "output.directory and output.run_name must be non-empty strings."
            )
        run_name = Path(self.output.run_name)
        if (
            run_name.is_absolute()
            or len(run_name.parts) != 1
            or self.output.run_name in {".", ".."}
        ):
            raise ValueError("output.run_name must be a single directory name.")
        if not isinstance(self.output.overwrite, bool):
            raise ValueError("output.overwrite must be a boolean.")
