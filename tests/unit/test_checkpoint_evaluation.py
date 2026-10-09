from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.checkpoint_evaluation import evaluate_checkpoint
from gridworld_rl.cli import main
from gridworld_rl.models import QNetwork
from gridworld_rl.reproducibility import sha256_file


def _inputs(tmp_path):
    model = QNetwork(4, 2, [4])
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(4))
        model.network[0].bias.zero_()
        model.network[2].weight.zero_()
        model.network[2].bias.zero_()
        for state, action in enumerate([0, 1, 0, 1]):
            model.network[2].weight[action, state] = 1
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "num_states": 4,
            "num_actions": 2,
            "network": {"hidden_sizes": [4]},
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    header = "state,action,reward,next_state,done\n"
    challenge = tmp_path / "challenge.csv"
    solution = tmp_path / "solution.csv"
    train = tmp_path / "train.csv"
    challenge.write_text(header + "0,-1,0,1,0\n1,-1,0,2,0\n2,-1,0,3,1\n")
    solution.write_text(header + "0,0,0,1,0\n1,0,0,2,0\n2,0,0,3,1\n")
    train.write_text(header + "0,0,0,1,0\n1,1,0,0,0\n")
    return checkpoint, challenge, solution, train


def test_evaluate_checkpoint_scores_and_slices_without_training(tmp_path, capsys):
    checkpoint, challenge, solution, train = _inputs(tmp_path)
    result = evaluate_checkpoint(
        checkpoint, challenge, solution, train_csv=train, batch_size=2
    )
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["confusion_matrix"] == [[2, 1], [0, 0]]
    assert result["inputs"]["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert result["overlap_sliced_agreement"] == {
        "exact_training_transition": {"rows": 1, "accuracy": 1.0},
        "seen_state_new_transition": {"rows": 1, "accuracy": 0.0},
        "unseen_state": {"rows": 1, "accuracy": 1.0},
    }
    assert result["evaluation_split_diagnostics"]["exact_training_overlap_rows"] == 1
    assert (
        main(
            [
                "evaluate-checkpoint",
                "--checkpoint",
                str(checkpoint),
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


def test_evaluate_checkpoint_works_without_training_csv(tmp_path):
    checkpoint, challenge, solution, _ = _inputs(tmp_path)
    result = evaluate_checkpoint(checkpoint, challenge, solution)
    assert result["metrics"]["num_examples"] == 3
    assert "overlap_sliced_agreement" not in result
    assert "train" not in result["inputs"]


def test_evaluate_checkpoint_rejects_unaligned_pair(tmp_path):
    checkpoint, challenge, solution, _ = _inputs(tmp_path)
    solution.write_text(solution.read_text().replace("2,0,0,3,1", "2,0,0,0,1"))
    with pytest.raises(ValueError, match="alignment mismatch"):
        evaluate_checkpoint(checkpoint, challenge, solution)


def test_evaluate_checkpoint_rejects_changed_input(tmp_path, monkeypatch):
    checkpoint, challenge, solution, _ = _inputs(tmp_path)
    real_hash = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if str(path) == str(solution):
            calls += 1
            if calls == 2:
                return "changed"
        return real_hash(path)

    monkeypatch.setattr("gridworld_rl.checkpoint_evaluation.sha256_file", changed_hash)
    with pytest.raises(ValueError, match="Solution changed"):
        evaluate_checkpoint(checkpoint, challenge, solution)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_evaluate_checkpoint_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_checkpoint(
            "missing.pt", "missing.csv", "missing.csv", batch_size=batch_size
        )
