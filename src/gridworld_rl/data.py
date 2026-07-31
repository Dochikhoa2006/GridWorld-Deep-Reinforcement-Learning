"""Validated, reproducible data loading for the offline Gridworld dataset.

The training and solution files contain complete transitions.  Evaluation
challenge files use ``action == -1`` to represent the action that a model must
predict; that sentinel is accepted only by the evaluation-pair loader.
"""

from __future__ import annotations

import os
import random
from numbers import Integral, Real
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

NUM_STATES: Final = 100
NUM_ACTIONS: Final = 4
UNKNOWN_ACTION: Final = -1
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "state",
    "action",
    "reward",
    "next_state",
    "done",
)


class DatasetValidationError(ValueError):
    """Raised when a transition dataset does not satisfy the data contract."""


def _row_preview(mask: np.ndarray, *, limit: int = 5) -> str:
    """Return a compact list of zero-based row positions selected by ``mask``."""

    rows = np.flatnonzero(mask)[:limit].tolist()
    suffix = " ..." if int(mask.sum()) > limit else ""
    return f"{rows}{suffix}"


def _validate_schema(frame: pd.DataFrame, *, source: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{source} must be a pandas DataFrame, got {type(frame).__name__}"
        )

    duplicated = [
        column for column in REQUIRED_COLUMNS if list(frame.columns).count(column) > 1
    ]
    if duplicated:
        names = ", ".join(duplicated)
        raise DatasetValidationError(
            f"{source} has duplicate required column(s): {names}"
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        names = ", ".join(missing)
        raise DatasetValidationError(
            f"{source} is missing required column(s): {names}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}"
        )

    if frame.empty:
        raise DatasetValidationError(f"{source} contains no transition rows")

    null_counts = frame.loc[:, REQUIRED_COLUMNS].isna().sum()
    columns_with_nulls = {
        column: int(count) for column, count in null_counts.items() if count
    }
    if columns_with_nulls:
        detail = ", ".join(
            f"{column} ({count})" for column, count in columns_with_nulls.items()
        )
        raise DatasetValidationError(
            f"{source} contains null values in required column(s): {detail}"
        )


def _normalise_integer_column(
    series: pd.Series,
    *,
    column: str,
    source: str,
    minimum: int,
    maximum: int,
    extra_allowed: tuple[int, ...] = (),
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)

    non_numeric = ~np.isfinite(values)
    if non_numeric.any():
        raise DatasetValidationError(
            f"{source} column '{column}' must contain finite numeric values; "
            f"invalid row(s): {_row_preview(non_numeric)}"
        )

    non_integral = values != np.floor(values)
    if non_integral.any():
        raise DatasetValidationError(
            f"{source} column '{column}' must contain integer values; "
            f"invalid row(s): {_row_preview(non_integral)}"
        )

    in_range = (values >= minimum) & (values <= maximum)
    for allowed in extra_allowed:
        in_range |= values == allowed
    if not in_range.all():
        allowed_text = f"{minimum}..{maximum}"
        if extra_allowed:
            sentinels = ", ".join(str(value) for value in extra_allowed)
            allowed_text += f" or sentinel(s) {sentinels}"
        invalid = ~in_range
        raise DatasetValidationError(
            f"{source} column '{column}' must be in {allowed_text}; "
            f"invalid row(s): {_row_preview(invalid)}"
        )

    return pd.Series(values.astype(np.int64), index=series.index, name=column)


def _normalise_reward(series: pd.Series, *, source: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise DatasetValidationError(
            f"{source} column 'reward' must contain finite numeric values; "
            f"invalid row(s): {_row_preview(invalid)}"
        )
    outside_float32 = np.abs(values) > np.finfo(np.float32).max
    if outside_float32.any():
        raise DatasetValidationError(
            f"{source} column 'reward' must fit the finite float32 range; "
            f"invalid row(s): {_row_preview(outside_float32)}"
        )
    return pd.Series(values, index=series.index, name="reward")


def _coerce_boolean(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, Real):
        numeric = float(value)
        return (
            bool(int(numeric))
            if np.isfinite(numeric) and numeric in (0.0, 1.0)
            else None
        )
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1"}:
            return True
        if normalised in {"false", "0"}:
            return False
    return None


def _normalise_done(series: pd.Series, *, source: str) -> pd.Series:
    converted: list[bool] = []
    invalid = np.zeros(len(series), dtype=bool)
    for position, value in enumerate(series.to_numpy(dtype=object)):
        boolean = _coerce_boolean(value)
        if boolean is None:
            invalid[position] = True
            converted.append(False)
        else:
            converted.append(boolean)

    if invalid.any():
        raise DatasetValidationError(
            f"{source} column 'done' must contain booleans or binary 0/1 values; "
            f"invalid row(s): {_row_preview(invalid)}"
        )
    return pd.Series(converted, index=series.index, name="done", dtype=bool)


def validate_transition_frame(
    frame: pd.DataFrame,
    *,
    source: str = "dataset",
    allow_unknown_action: bool = False,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> pd.DataFrame:
    """Validate and return a normalised copy of a transition DataFrame.

    State and action identifiers must fit the configured discrete spaces,
    rewards must be finite, and terminal flags must be boolean (or an
    unambiguous binary representation). Numeric strings and integral
    floating-point values are normalised rather than rejected.

    ``allow_unknown_action`` exists for evaluation challenge files and permits
    the single ``-1`` sentinel in addition to valid actions.
    """

    if num_states <= 1 or num_actions <= 1:
        raise ValueError("num_states and num_actions must be greater than 1")
    _validate_schema(frame, source=source)
    validated = frame.copy(deep=True).reset_index(drop=True)
    validated["state"] = _normalise_integer_column(
        validated["state"],
        column="state",
        source=source,
        minimum=0,
        maximum=num_states - 1,
    )
    validated["action"] = _normalise_integer_column(
        validated["action"],
        column="action",
        source=source,
        minimum=0,
        maximum=num_actions - 1,
        extra_allowed=(UNKNOWN_ACTION,) if allow_unknown_action else (),
    )
    validated["reward"] = _normalise_reward(validated["reward"], source=source)
    validated["next_state"] = _normalise_integer_column(
        validated["next_state"],
        column="next_state",
        source=source,
        minimum=0,
        maximum=num_states - 1,
    )
    validated["done"] = _normalise_done(validated["done"], source=source)
    return validated


def _read_csv(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, Path]:
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Download or copy the Gridworld CSV files before running this command."
        )

    try:
        return pd.read_csv(dataset_path), dataset_path
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(
            f"Could not parse dataset CSV '{dataset_path}': {exc}"
        ) from exc


def load_transition_csv(
    path: str | os.PathLike[str],
    *,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> pd.DataFrame:
    """Load and strictly validate a training or solution transition CSV."""

    frame, dataset_path = _read_csv(path)
    return validate_transition_frame(
        frame,
        source=str(dataset_path),
        num_states=num_states,
        num_actions=num_actions,
    )


def validate_evaluation_alignment(
    challenge: pd.DataFrame,
    solution: pd.DataFrame,
    *,
    challenge_source: str = "evaluation challenge",
    solution_source: str = "evaluation solution",
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate an evaluation pair and ensure transition rows are aligned.

    The action and reward columns are labels/metadata and therefore are not
    compared.  ``state``, ``next_state``, and ``done`` must match at every row.
    """

    checked_challenge = validate_transition_frame(
        challenge,
        source=challenge_source,
        allow_unknown_action=True,
        num_states=num_states,
        num_actions=num_actions,
    )
    checked_solution = validate_transition_frame(
        solution,
        source=solution_source,
        num_states=num_states,
        num_actions=num_actions,
    )

    if len(checked_challenge) != len(checked_solution):
        raise DatasetValidationError(
            "Evaluation challenge/solution row-count mismatch: "
            f"{len(checked_challenge)} != {len(checked_solution)}"
        )

    for column in ("state", "next_state", "done"):
        left = checked_challenge[column].to_numpy()
        right = checked_solution[column].to_numpy()
        mismatch = left != right
        if mismatch.any():
            raise DatasetValidationError(
                "Evaluation challenge/solution alignment mismatch in "
                f"column '{column}' at row(s): {_row_preview(mismatch)}"
            )

    return checked_challenge, checked_solution


def load_evaluation_data(
    challenge_path: str | os.PathLike[str],
    solution_path: str | os.PathLike[str],
    *,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and align evaluation challenge and solution CSVs."""

    challenge, challenge_file = _read_csv(challenge_path)
    solution, solution_file = _read_csv(solution_path)
    return validate_evaluation_alignment(
        challenge,
        solution,
        challenge_source=str(challenge_file),
        solution_source=str(solution_file),
        num_states=num_states,
        num_actions=num_actions,
    )


def transition_diagnostics(
    frame: pd.DataFrame,
    *,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, object]:
    """Summarize logged state-action coverage without exposing transition rows."""

    checked = validate_transition_frame(
        frame,
        source="diagnostic dataset",
        num_states=num_states,
        num_actions=num_actions,
    )
    observed_pairs = checked.loc[:, ["state", "action"]].drop_duplicates()
    action_counts = checked["action"].value_counts()
    rewards = checked["reward"].to_numpy(dtype=np.float64)
    return {
        "observed_states": int(checked["state"].nunique()),
        "possible_states": num_states,
        "observed_state_action_pairs": len(observed_pairs),
        "possible_state_action_pairs": num_states * num_actions,
        "state_action_coverage": float(
            len(observed_pairs) / (num_states * num_actions)
        ),
        "action_counts": {
            str(action): int(action_counts.get(action, 0))
            for action in range(num_actions)
        },
        "terminal_fraction": float(checked["done"].mean()),
        "reward": {
            "minimum": float(rewards.min()),
            "maximum": float(rewards.max()),
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=0)),
        },
    }


def evaluation_split_diagnostics(
    train: pd.DataFrame,
    solution: pd.DataFrame,
    *,
    num_states: int = NUM_STATES,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, object]:
    """Quantify evaluation overlap, label ambiguity, and trivial references."""

    checked_train = validate_transition_frame(
        train,
        source="diagnostic training dataset",
        num_states=num_states,
        num_actions=num_actions,
    )
    checked_solution = validate_transition_frame(
        solution,
        source="diagnostic evaluation solution",
        num_states=num_states,
        num_actions=num_actions,
    )
    transition_columns = list(REQUIRED_COLUMNS)
    train_transitions = set(
        checked_train.loc[:, transition_columns].itertuples(
            index=False,
            name=None,
        )
    )
    solution_rows = list(
        checked_solution.loc[:, transition_columns].itertuples(
            index=False,
            name=None,
        )
    )
    overlap_rows = sum(row in train_transitions for row in solution_rows)
    unique_evaluation_transitions = len(set(solution_rows))

    evaluation_action_counts = (
        checked_solution["action"]
        .value_counts()
        .reindex(
            range(num_actions),
            fill_value=0,
        )
    )
    train_action_counts = (
        checked_train.groupby(["state", "action"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["state", "count", "action"],
            ascending=[True, False, True],
        )
    )
    train_state_mode = (
        train_action_counts.drop_duplicates("state")
        .set_index("state")["action"]
        .to_dict()
    )
    fallback_action = int(checked_train["action"].value_counts().idxmax())
    training_majority_accuracy = float(
        (checked_solution["action"] == fallback_action).mean()
    )
    train_mode_predictions = checked_solution["state"].map(train_state_mode)
    train_mode_predictions = train_mode_predictions.fillna(fallback_action).astype(
        np.int64
    )
    train_state_mode_accuracy = float(
        (train_mode_predictions == checked_solution["action"]).mean()
    )

    evaluation_state_action_counts = checked_solution.groupby(
        ["state", "action"]
    ).size()
    evaluation_state_mode_ceiling = float(
        evaluation_state_action_counts.groupby(level="state").max().sum()
        / len(checked_solution)
    )
    conflicting_states = int(
        (checked_solution.groupby("state")["action"].nunique() > 1).sum()
    )
    unique_evaluation_states = int(checked_solution["state"].nunique())
    training_states = set(checked_train["state"].unique())
    evaluation_states = set(checked_solution["state"].unique())

    return {
        "rows": len(checked_solution),
        "unique_full_transitions": unique_evaluation_transitions,
        "duplicate_rows": len(checked_solution) - unique_evaluation_transitions,
        "exact_training_overlap_rows": overlap_rows,
        "exact_training_overlap_fraction": float(overlap_rows / len(checked_solution)),
        "unique_states": unique_evaluation_states,
        "states_seen_in_training": len(evaluation_states & training_states),
        "states_with_conflicting_actions": conflicting_states,
        "conflicting_state_fraction": float(
            conflicting_states / unique_evaluation_states
        ),
        "action_counts": {
            str(action): int(evaluation_action_counts.loc[action])
            for action in range(num_actions)
        },
        "training_majority": {
            "action": fallback_action,
            "accuracy": training_majority_accuracy,
        },
        "training_state_mode_accuracy": train_state_mode_accuracy,
        "evaluation_state_mode_ceiling": evaluation_state_mode_ceiling,
    }


class TransitionDataset(Dataset[dict[str, Tensor]]):
    """In-memory PyTorch dataset backed by validated transition tensors."""

    def __init__(
        self,
        data: pd.DataFrame | str | os.PathLike[str],
        *,
        source: str = "dataset",
        num_states: int = NUM_STATES,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            frame = validate_transition_frame(
                data,
                source=source,
                num_states=num_states,
                num_actions=num_actions,
            )
        else:
            frame = load_transition_csv(
                data,
                num_states=num_states,
                num_actions=num_actions,
            )

        self.frame = frame
        self.states = torch.tensor(frame["state"].to_numpy(), dtype=torch.long)
        self.actions = torch.tensor(frame["action"].to_numpy(), dtype=torch.long)
        self.rewards = torch.tensor(frame["reward"].to_numpy(), dtype=torch.float32)
        self.next_states = torch.tensor(
            frame["next_state"].to_numpy(), dtype=torch.long
        )
        self.dones = torch.tensor(frame["done"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "state": self.states[index],
            "action": self.actions[index],
            "reward": self.rewards[index],
            "next_state": self.next_states[index],
            "done": self.dones[index],
        }


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset[dict[str, Tensor]],
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Tensor]]:
    """Build a DataLoader with deterministic sampling and worker RNG seeds."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral):
        raise TypeError("num_workers must be an integer")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers else None,
    )
