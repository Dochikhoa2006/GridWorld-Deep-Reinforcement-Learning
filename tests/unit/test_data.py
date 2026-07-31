from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gridworld_rl.data import (
    DatasetValidationError,
    TransitionDataset,
    evaluation_split_diagnostics,
    load_evaluation_data,
    load_transition_csv,
    make_dataloader,
    transition_diagnostics,
    validate_evaluation_alignment,
    validate_transition_frame,
)


def valid_frame(rows: int = 4) -> pd.DataFrame:
    data = {
        "state": [0, 11, 42, 98],
        "action": [0, 1, 2, 3],
        "reward": [0.4, -1.3, 20.0, 0.0],
        "next_state": [1, 21, 43, 99],
        "done": [False, "false", 0, 1],
    }
    return pd.DataFrame({key: values[:rows] for key, values in data.items()})


def test_transition_dataset_returns_typed_tensor_mapping() -> None:
    dataset = TransitionDataset(valid_frame())

    assert len(dataset) == 4
    sample = dataset[1]
    assert set(sample) == {"state", "action", "reward", "next_state", "done"}
    assert sample["state"].dtype == torch.long
    assert sample["action"].dtype == torch.long
    assert sample["reward"].dtype == torch.float32
    assert sample["next_state"].dtype == torch.long
    assert sample["done"].dtype == torch.float32
    assert sample["done"].item() == 0.0


def test_load_transition_csv_has_clear_missing_file_error(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError, match=r"Dataset file not found: .*missing.csv"
    ):
        load_transition_csv(missing)


def test_missing_required_columns_are_reported_together() -> None:
    frame = valid_frame().drop(columns=["reward", "done"])

    with pytest.raises(
        DatasetValidationError,
        match=r"missing required column\(s\): reward, done",
    ):
        validate_transition_frame(frame)


def test_null_in_required_column_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[2, "reward"] = np.nan

    with pytest.raises(DatasetValidationError, match=r"null values.*reward \(1\)"):
        validate_transition_frame(frame)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("state", "not-a-number", "finite numeric"),
        ("state", 1.5, "integer values"),
        ("action", "left", "finite numeric"),
        ("action", 2.25, "integer values"),
    ],
)
def test_state_and_action_must_be_numeric_integers(
    column: str, value: object, message: str
) -> None:
    frame = valid_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value

    with pytest.raises(DatasetValidationError, match=message):
        validate_transition_frame(frame)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("state", -1),
        ("state", 100),
        ("next_state", -1),
        ("next_state", 100),
        ("action", -1),
        ("action", 4),
    ],
)
def test_state_and_action_ranges_are_enforced(column: str, value: int) -> None:
    frame = valid_frame()
    frame.loc[0, column] = value

    with pytest.raises(DatasetValidationError, match=rf"column '{column}'.*must be in"):
        validate_transition_frame(frame)


@pytest.mark.parametrize("reward", [np.inf, -np.inf, "not-a-reward"])
def test_reward_must_be_finite_numeric(reward: object) -> None:
    frame = valid_frame()
    frame["reward"] = frame["reward"].astype(object)
    frame.loc[0, "reward"] = reward

    with pytest.raises(DatasetValidationError, match=r"reward.*finite numeric"):
        validate_transition_frame(frame)


def test_reward_must_remain_finite_after_float32_tensor_conversion() -> None:
    frame = valid_frame()
    frame.loc[0, "reward"] = 1e300

    with pytest.raises(DatasetValidationError, match="finite float32 range"):
        validate_transition_frame(frame)


@pytest.mark.parametrize("done", [2, -1, 0.5, "yes", "finished"])
def test_done_rejects_ambiguous_or_non_boolean_values(done: object) -> None:
    frame = valid_frame()
    frame.loc[0, "done"] = done

    with pytest.raises(DatasetValidationError, match=r"done.*booleans or binary"):
        validate_transition_frame(frame)


def test_dataloader_shuffle_is_reproducible_for_a_seed() -> None:
    frame = pd.concat([valid_frame()] * 4, ignore_index=True)
    frame["state"] = np.arange(len(frame))
    frame["next_state"] = np.arange(1, len(frame) + 1)
    dataset = TransitionDataset(frame)

    first = torch.cat(
        [batch["state"] for batch in make_dataloader(dataset, batch_size=3, seed=17)]
    )
    second = torch.cat(
        [batch["state"] for batch in make_dataloader(dataset, batch_size=3, seed=17)]
    )
    different = torch.cat(
        [batch["state"] for batch in make_dataloader(dataset, batch_size=3, seed=18)]
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_evaluation_loader_accepts_unknown_action_and_normalises_pair(tmp_path) -> None:
    solution = valid_frame()
    challenge = solution.copy()
    challenge["action"] = -1
    challenge["reward"] = -999
    challenge_path = tmp_path / "challenge.csv"
    solution_path = tmp_path / "solution.csv"
    challenge.to_csv(challenge_path, index=False)
    solution.to_csv(solution_path, index=False)

    checked_challenge, checked_solution = load_evaluation_data(
        challenge_path, solution_path
    )

    assert checked_challenge["action"].tolist() == [-1, -1, -1, -1]
    assert checked_solution["action"].tolist() == [0, 1, 2, 3]
    assert checked_challenge["done"].dtype == bool


def test_evaluation_pair_rejects_row_count_mismatch() -> None:
    with pytest.raises(DatasetValidationError, match="row-count mismatch"):
        validate_evaluation_alignment(valid_frame(3), valid_frame(4))


@pytest.mark.parametrize("column", ["state", "next_state", "done"])
def test_evaluation_pair_rejects_transition_misalignment(column: str) -> None:
    challenge = valid_frame()
    solution = valid_frame()
    if column == "done":
        challenge.loc[0, column] = True
    else:
        challenge.loc[0, column] = challenge.loc[0, column] + 1

    with pytest.raises(
        DatasetValidationError,
        match=rf"alignment mismatch in column '{column}'",
    ):
        validate_evaluation_alignment(challenge, solution)


def test_validation_uses_configured_discrete_space_sizes() -> None:
    frame = pd.DataFrame(
        {
            "state": [0, 4],
            "action": [0, 1],
            "reward": [0.0, 1.0],
            "next_state": [1, 5],
            "done": [False, True],
        }
    )

    checked = validate_transition_frame(frame, num_states=6, num_actions=2)
    assert checked["next_state"].max() == 5
    with pytest.raises(DatasetValidationError, match=r"state'.*0..4"):
        validate_transition_frame(frame, num_states=5, num_actions=2)


def test_transition_diagnostics_report_coverage_and_action_counts() -> None:
    diagnostics = transition_diagnostics(
        valid_frame(),
        num_states=100,
        num_actions=4,
    )

    assert diagnostics["observed_states"] == 4
    assert diagnostics["observed_state_action_pairs"] == 4
    assert diagnostics["possible_state_action_pairs"] == 400
    assert diagnostics["state_action_coverage"] == pytest.approx(0.01)
    assert diagnostics["action_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert diagnostics["terminal_fraction"] == pytest.approx(0.25)


def test_evaluation_diagnostics_expose_overlap_ambiguity_and_references() -> None:
    train = pd.DataFrame(
        {
            "state": [0, 0, 1],
            "action": [0, 1, 1],
            "reward": [0.0, 0.1, 1.0],
            "next_state": [1, 1, 2],
            "done": [False, False, True],
        }
    )
    solution = pd.DataFrame(
        {
            "state": [0, 0, 0, 1],
            "action": [0, 0, 1, 0],
            "reward": [0.0, 0.0, 0.1, -1.0],
            "next_state": [1, 1, 1, 2],
            "done": [False, False, False, True],
        }
    )

    diagnostics = evaluation_split_diagnostics(
        train,
        solution,
        num_states=3,
        num_actions=2,
    )

    assert diagnostics["unique_full_transitions"] == 3
    assert diagnostics["duplicate_rows"] == 1
    assert diagnostics["exact_training_overlap_rows"] == 3
    assert diagnostics["exact_training_overlap_fraction"] == pytest.approx(0.75)
    assert diagnostics["states_with_conflicting_actions"] == 1
    assert diagnostics["training_majority"] == {
        "action": 1,
        "accuracy": 0.25,
    }
    assert diagnostics["training_state_mode_accuracy"] == pytest.approx(0.5)
    assert diagnostics["evaluation_state_mode_ceiling"] == pytest.approx(0.75)
