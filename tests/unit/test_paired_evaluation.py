from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.models import QNetwork
from gridworld_rl.paired_evaluation import compare_evaluations
from gridworld_rl.reproducibility import sha256_file


def _checkpoint(tmp_path, name, actions):
    model = QNetwork(4, 2, [4])
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(4))
        model.network[0].bias.zero_()
        model.network[2].weight.zero_()
        model.network[2].bias.zero_()
        for state, action in enumerate(actions):
            model.network[2].weight[action, state] = 1
    path = tmp_path / name
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "num_states": 4,
            "num_actions": 2,
            "network": {"hidden_sizes": [4]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _data(tmp_path):
    header = "state,action,reward,next_state,done\n"
    challenge = tmp_path / "challenge.csv"
    solution = tmp_path / "solution.csv"
    train = tmp_path / "train.csv"
    challenge.write_text(header + "0,-1,0,1,0\n1,-1,0,2,0\n2,-1,0,3,1\n3,-1,0,0,1\n")
    solution.write_text(header + "0,0,0,1,0\n1,0,0,2,0\n2,1,0,3,1\n3,1,0,0,1\n")
    train.write_text(header + "0,0,0,1,0\n1,1,0,2,0\n")
    return challenge, solution, train


def test_paired_evaluation_counts_shared_rows_and_slices(tmp_path, capsys):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 0, 0])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 0])
    third = _checkpoint(tmp_path, "c.pt", [1, 0, 0, 1])
    challenge, solution, train = _data(tmp_path)
    result = compare_evaluations(
        [first, second, third], challenge, solution, train_csv=train, batch_size=2
    )
    assert result["evaluation_rows"] == 4
    assert [row["metrics"]["accuracy"] for row in result["checkpoints"]] == [
        0.5,
        0.5,
        0.5,
    ]
    assert result["pairs"][0] == {
        "left": 0,
        "right": 1,
        "both_correct": 1,
        "left_only_correct": 1,
        "right_only_correct": 1,
        "both_wrong": 1,
        "action_agreement": 2,
        "action_agreement_rate": 0.5,
    }
    assert len(result["pairs"]) == 3
    assert result["inputs"]["train"]["sha256"] == sha256_file(train)
    assert result["evaluation_split_diagnostics"]["exact_training_overlap_rows"] == 1
    assert result["checkpoints"][0]["overlap_sliced_agreement"][
        "exact_training_transition"
    ] == {"rows": 1, "accuracy": 1.0}
    assert (
        main(
            [
                "compare-evaluations",
                "--checkpoints",
                str(first),
                str(second),
                str(third),
                "--challenge",
                str(challenge),
                "--solution",
                str(solution),
                "--train",
                str(train),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_paired_evaluation_without_train_and_duplicate_rejection(tmp_path):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 0, 0])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 0])
    challenge, solution, _ = _data(tmp_path)
    result = compare_evaluations([first, second], challenge, solution)
    assert "evaluation_split_diagnostics" not in result
    assert "overlap_sliced_agreement" not in result["checkpoints"][0]
    with pytest.raises(ValueError, match="distinct"):
        compare_evaluations([first, first], challenge, solution)
    with pytest.raises(ValueError, match="At least two"):
        compare_evaluations([first], challenge, solution)


def test_paired_evaluation_rejects_changed_input(tmp_path, monkeypatch):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 0, 0])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 0])
    challenge, solution, _ = _data(tmp_path)
    original = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if str(path) == str(solution):
            calls += 1
            if calls == 2:
                return "changed"
        return original(path)

    monkeypatch.setattr("gridworld_rl.paired_evaluation.sha256_file", changed_hash)
    with pytest.raises(ValueError, match="Solution changed"):
        compare_evaluations([first, second], challenge, solution)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_paired_evaluation_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        compare_evaluations(
            ["a.pt", "b.pt"], "challenge.csv", "solution.csv", batch_size=batch_size
        )
