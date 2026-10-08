from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.models import QNetwork
from gridworld_rl.reproducibility import sha256_file
from gridworld_rl.support import audit_policy_support


def _inputs(tmp_path):
    model = QNetwork(5, 2, [5])
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(5))
        model.network[0].bias.zero_()
        model.network[2].weight.zero_()
        model.network[2].bias.zero_()
        for state, action in enumerate([0, 1, 0, 1, 0]):
            model.network[2].weight[action, state] = 1
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "cql",
            "num_states": 5,
            "num_actions": 2,
            "network": {"hidden_sizes": [5]},
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    train = tmp_path / "train.csv"
    train.write_text(
        "state,action,reward,next_state,done\n"
        "0,0,0,1,0\n0,1,0,1,0\n"
        "1,0,0,2,0\n1,0,0,2,0\n1,0,0,2,0\n"
        "3,1,0,4,1\n"
    )
    return checkpoint, train


def test_support_audit_distinguishes_state_and_row_weighting(tmp_path, capsys):
    checkpoint, train = _inputs(tmp_path)
    result = audit_policy_support(checkpoint, train, batch_size=2)
    assert result["algorithm"] == "cql"
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["train_sha256"] == sha256_file(train)
    assert result["training_rows"] == 6
    assert result["observed_states"] == 3
    assert result["unobserved_states"] == 2
    assert result["supported_observed_states"] == 2
    assert result["unsupported_observed_states"] == [1]
    assert result["observed_state_support_rate"] == pytest.approx(2 / 3)
    assert result["logged_row_support_rate"] == 0.5
    assert result["chosen_action_counts"] == [1, 2]
    assert result["unsupported_chosen_action_counts"] == [0, 1]
    assert (
        main(
            [
                "audit-policy-support",
                "--checkpoint",
                str(checkpoint),
                "--train",
                str(train),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_support_audit_rejects_incompatible_csv(tmp_path):
    checkpoint, train = _inputs(tmp_path)
    train.write_text(train.read_text() + "5,0,0,0,0\n")
    with pytest.raises(ValueError, match=r"must be in 0\.\.4"):
        audit_policy_support(checkpoint, train)


def test_support_audit_rejects_changed_input(tmp_path, monkeypatch):
    checkpoint, train = _inputs(tmp_path)
    original = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        calls += 1
        return "changed" if calls == 4 else original(path)

    monkeypatch.setattr("gridworld_rl.support.sha256_file", changed_hash)
    with pytest.raises(ValueError, match="Training CSV changed"):
        audit_policy_support(checkpoint, train)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_support_audit_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        audit_policy_support("missing.pt", "missing.csv", batch_size=batch_size)
