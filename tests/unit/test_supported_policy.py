from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.models import QNetwork
from gridworld_rl.reproducibility import sha256_file
from gridworld_rl.supported_policy import export_supported_policy


def _inputs(tmp_path):
    model = QNetwork(5, 3, [5])
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(5))
        model.network[0].bias.zero_()
        model.network[2].weight.copy_(
            torch.tensor(
                [
                    [3.0, 0.0, -1.0, 2.0, 1.0],
                    [1.0, 5.0, -1.0, 2.0, 0.0],
                    [0.0, 1.0, -2.0, 0.0, 4.0],
                ]
            )
        )
        model.network[2].bias.zero_()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "cql",
            "num_states": 5,
            "num_actions": 3,
            "network": {"hidden_sizes": [5]},
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    train = tmp_path / "train.csv"
    train.write_text(
        "state,action,reward,next_state,done\n"
        "0,1,0,1,0\n0,2,0,1,0\n1,0,0,2,0\n"
        "2,1,0,3,0\n2,2,0,3,0\n3,0,0,4,1\n3,1,0,4,1\n"
    )
    return checkpoint, train


@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_supported_policy_constrains_only_observed_states(tmp_path, batch_size):
    checkpoint, train = _inputs(tmp_path)
    output = tmp_path / "exports/policy.json"
    assert (
        export_supported_policy(checkpoint, train, output, batch_size=batch_size)
        == output
    )
    payload = json.loads(output.read_text())
    assert payload["checkpoint_sha256"] == sha256_file(checkpoint)
    assert payload["train_sha256"] == sha256_file(train)
    assert [row["state"] for row in payload["policy"]] == list(range(5))
    assert [row["action"] for row in payload["policy"]] == [1, 0, 1, 0, 2]
    assert [row["unconstrained_action"] for row in payload["policy"]] == [
        0,
        1,
        0,
        0,
        2,
    ]
    assert payload["policy"][2]["logged_actions"] == [1, 2]
    assert payload["policy"][3]["was_constrained"] is False
    assert payload["policy"][4]["logged_actions"] == []
    assert payload["summary"] == {
        "observed_states": 4,
        "unobserved_states": 1,
        "changed_observed_states": 3,
    }
    assert list(output.parent.iterdir()) == [output]


def test_supported_policy_cli_and_no_clobber(tmp_path, capsys):
    checkpoint, train = _inputs(tmp_path)
    output = tmp_path / "policy.json"
    args = [
        "export-supported-policy",
        "--checkpoint",
        str(checkpoint),
        "--train",
        str(train),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert "Supported policy exported" in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert output.read_bytes() == before


def test_supported_policy_rejects_changed_training_csv(tmp_path, monkeypatch):
    checkpoint, train = _inputs(tmp_path)
    real_hash = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if str(path) == str(train):
            calls += 1
            if calls == 2:
                return "changed"
        return real_hash(path)

    monkeypatch.setattr("gridworld_rl.supported_policy.sha256_file", changed_hash)
    output = tmp_path / "policy.json"
    with pytest.raises(ValueError, match="Training CSV changed"):
        export_supported_policy(checkpoint, train, output)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.pt", "train.csv"]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_supported_policy_rejects_bad_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        export_supported_policy(
            "model.pt", "train.csv", tmp_path / "out.json", batch_size=batch_size
        )
