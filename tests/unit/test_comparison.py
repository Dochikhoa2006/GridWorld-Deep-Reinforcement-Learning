from __future__ import annotations

import json

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.comparison import compare_checkpoints
from gridworld_rl.models import QNetwork
from gridworld_rl.reproducibility import sha256_file


def _checkpoint(tmp_path, name, actions, *, num_states=4):
    model = QNetwork(num_states, 2, [num_states])
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(num_states))
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
            "num_states": num_states,
            "num_actions": 2,
            "network": {"hidden_sizes": [num_states]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_three_way_comparison_counts_full_state_space(tmp_path, capsys):
    paths = [
        _checkpoint(tmp_path, "a.pt", [0, 0, 1, 1]),
        _checkpoint(tmp_path, "b.pt", [0, 1, 1, 1]),
        _checkpoint(tmp_path, "c.pt", [1, 1, 1, 1]),
    ]
    result = compare_checkpoints(paths, batch_size=2)
    assert result["unanimous_states"] == 2
    assert result["disputed_states"] == 2
    assert [row["action_counts"] for row in result["checkpoints"]] == [
        [2, 2],
        [1, 3],
        [0, 4],
    ]
    assert [(pair["agree"], pair["disagree"]) for pair in result["pairs"]] == [
        (3, 1),
        (2, 2),
        (3, 1),
    ]
    assert result["pairs"][0]["agreement_rate"] == 0.75
    assert [row["sha256"] for row in result["checkpoints"]] == [
        sha256_file(path) for path in paths
    ]
    assert main(["compare-checkpoints", "--checkpoints", *(str(p) for p in paths)]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_comparison_rejects_duplicate_and_incompatible_checkpoints(tmp_path):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 1, 1])
    with pytest.raises(ValueError, match="distinct"):
        compare_checkpoints([first, first])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 1, 0], num_states=5)
    with pytest.raises(ValueError, match="identical state and action dimensions"):
        compare_checkpoints([first, second])
    with pytest.raises(ValueError, match="At least two"):
        compare_checkpoints([first])


def test_comparison_rejects_changed_checkpoint(tmp_path, monkeypatch):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 1, 1])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 1])
    real_hash = sha256_file
    calls = 0

    def changed_hash(path):
        nonlocal calls
        calls += 1
        return "changed" if calls == 3 else real_hash(path)

    monkeypatch.setattr("gridworld_rl.comparison.sha256_file", changed_hash)
    with pytest.raises(ValueError, match="changed during comparison"):
        compare_checkpoints([first, second])


def test_comparison_rejects_nonfinite_inference(tmp_path):
    first = _checkpoint(tmp_path, "a.pt", [0, 0, 1, 1])
    second = _checkpoint(tmp_path, "b.pt", [0, 1, 1, 1])
    payload = torch.load(second, weights_only=True)
    payload["model_state_dict"]["network.0.weight"].fill_(1e38)
    payload["model_state_dict"]["network.2.weight"].fill_(1e38)
    torch.save(payload, second)
    with pytest.raises(ValueError, match="non-finite Q-values"):
        compare_checkpoints([first, second], batch_size=2)
