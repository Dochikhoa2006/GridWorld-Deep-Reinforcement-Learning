from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.dataset_comparison import compare_datasets
from gridworld_rl.reproducibility import sha256_file


def _logs(tmp_path):
    header = "state,action,reward,next_state,done\n"
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text(header + "0,0,1,1,0\n0,0,1,1,0\n1,1,0,2,1\n")
    right.write_text(header + "0,0,1,1,0\n2,1,0,2,1\n2,1,0,2,1\n")
    return left, right


def test_dataset_comparison_measures_coverage_and_weighted_shift(tmp_path, capsys):
    left, right = _logs(tmp_path)
    result = compare_datasets(left, right, num_states=3, num_actions=2)
    assert result["inputs"]["left"]["sha256"] == sha256_file(left)
    assert result["inputs"]["right"]["sha256"] == sha256_file(right)
    assert result["overlap"]["states"] == {
        "left_unique": 2,
        "right_unique": 2,
        "shared": 1,
        "left_only": 1,
        "right_only": 1,
        "jaccard": pytest.approx(1 / 3),
    }
    assert result["overlap"]["state_action_pairs"]["shared"] == 1
    assert result["overlap"]["full_transitions"]["shared"] == 1
    assert result["overlap"]["right_rows_at_new_states"] == 2
    assert result["overlap"]["right_rows_at_new_state_action_pairs"] == 2
    shift = result["distribution_shift"]
    assert 0 < shift["action_js_divergence_bits"] < 1
    assert shift["state_action_js_divergence_bits"] > shift["action_js_divergence_bits"]
    assert (
        main(
            [
                "compare-datasets",
                "--left",
                str(left),
                "--right",
                str(right),
                "--num-states",
                "3",
                "--num-actions",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_dataset_comparison_identical_logs_have_zero_shift(tmp_path):
    left, _ = _logs(tmp_path)
    result = compare_datasets(left, left, num_states=3, num_actions=2)
    assert result["overlap"]["states"]["jaccard"] == 1
    assert result["overlap"]["full_transitions"]["jaccard"] == 1
    assert result["distribution_shift"] == {
        "action_js_divergence_bits": 0,
        "state_action_js_divergence_bits": 0,
    }


def test_dataset_comparison_rejects_changed_csv(tmp_path, monkeypatch):
    left, right = _logs(tmp_path)
    original = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if str(path) == str(right):
            calls += 1
            if calls == 2:
                return "changed"
        return original(path)

    monkeypatch.setattr("gridworld_rl.dataset_comparison.sha256_file", changed_hash)
    with pytest.raises(ValueError, match="Right CSV changed"):
        compare_datasets(left, right, num_states=3, num_actions=2)


@pytest.mark.parametrize(("num_states", "num_actions"), [(1, 2), (3, True), (3.5, 2)])
def test_dataset_comparison_rejects_invalid_dimensions(num_states, num_actions):
    with pytest.raises(ValueError, match="integers greater than 1"):
        compare_datasets(
            "left.csv", "right.csv", num_states=num_states, num_actions=num_actions
        )
