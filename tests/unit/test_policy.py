from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.checkpoints import load_checkpoint
from gridworld_rl.cli import main
from gridworld_rl.models import QNetwork
from gridworld_rl.policy import export_policy
from gridworld_rl.reproducibility import sha256_file


def _checkpoint(tmp_path):
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
    payload = {
        "format_version": 1,
        "algorithm": "dqn",
        "num_states": 5,
        "num_actions": 3,
        "network": {"hidden_sizes": [5]},
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path, payload


@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_exports_complete_policy_with_gaps_and_exact_ties(tmp_path, batch_size):
    checkpoint, _ = _checkpoint(tmp_path)
    output = tmp_path / "exports/policy.json"
    assert export_policy(checkpoint, output, batch_size=batch_size) == output
    payload = json.loads(output.read_text())
    assert payload["checkpoint_sha256"] == sha256_file(checkpoint)
    assert payload["num_states"] == 5 and payload["num_actions"] == 3
    rows = payload["policy"]
    assert [r["state"] for r in rows] == list(range(5))
    assert [r["action"] for r in rows] == [0, 1, 0, 0, 2]
    assert [r["action_gap"] for r in rows] == [2, 4, 0, 0, 3]
    assert [r["num_greedy_actions"] for r in rows] == [1, 1, 2, 2, 1]
    assert rows[2]["q_values"] == [-1, -1, -2]
    assert list(output.parent.iterdir()) == [output]


def test_cli_export_and_existing_output_protection(tmp_path, capsys):
    checkpoint, _ = _checkpoint(tmp_path)
    output = tmp_path / "policy.json"
    args = ["export-policy", "--checkpoint", str(checkpoint), "--output", str(output)]
    assert main(args) == 0
    assert "Policy exported" in capsys.readouterr().out
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        export_policy(
            tmp_path / "absent.pt", tmp_path / "out.json", batch_size=batch_size
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("format_version", 2, "format_version"),
        ("format_version", True, "format_version"),
        ("algorithm", "unknown", "algorithm"),
        ("num_states", 5.5, "num_states"),
        ("num_actions", True, "num_actions"),
        ("network", None, "hidden_sizes"),
        ("network", {"hidden_sizes": [False]}, "hidden_sizes"),
        ("model_state_dict", {"x": "bad"}, "finite tensors"),
        ("model_state_dict", {"x": torch.tensor(float("nan"))}, "finite tensors"),
    ],
)
def test_rejects_malformed_checkpoint_metadata(tmp_path, field, value, message):
    path, payload = _checkpoint(tmp_path)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        export_policy(path, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_legacy_checkpoint_without_version_loads(tmp_path):
    path, payload = _checkpoint(tmp_path)
    del payload["format_version"]
    torch.save(payload, path)
    model, _ = load_checkpoint(path)
    assert model.num_states == 5


def test_corrupt_checkpoint_has_clear_error(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        load_checkpoint(path)


def test_publication_failure_cleans_temporary_file(tmp_path, monkeypatch):
    path, _ = _checkpoint(tmp_path)
    output = tmp_path / "policy.json"

    def competing_writer(source, destination):
        destination.write_text("another writer")
        raise FileExistsError("race")

    monkeypatch.setattr("gridworld_rl.policy.os.link", competing_writer)
    with pytest.raises(FileExistsError):
        export_policy(path, output)
    assert output.read_text() == "another writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt", "policy.json"]


def test_changed_checkpoint_aborts_export(tmp_path, monkeypatch):
    path, _ = _checkpoint(tmp_path)
    hashes = iter(["before", "after"])
    monkeypatch.setattr("gridworld_rl.policy.sha256_file", lambda _: next(hashes))
    with pytest.raises(ValueError, match="changed during"):
        export_policy(path, tmp_path / "policy.json")
    assert not (tmp_path / "policy.json").exists()


def test_export_streams_bounded_batches_and_cleans_up_late_failure(
    tmp_path, monkeypatch
):
    path, _ = _checkpoint(tmp_path)
    original_forward = QNetwork.forward
    batches = []

    def recording_forward(model, states):
        batches.append(states.tolist())
        result = original_forward(model, states)
        if states[0].item() == 4:
            result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(QNetwork, "forward", recording_forward)
    output = tmp_path / "policy.json"
    with pytest.raises(ValueError, match="non-finite Q-values"):
        export_policy(path, output, batch_size=2)
    assert batches == [[0, 1], [2, 3], [4]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt"]
