from __future__ import annotations

import csv
import json

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.ensemble_audit import audit_ensemble_predictions
from gridworld_rl.ensemble_inference import predict_ensemble_csv
from gridworld_rl.models import QNetwork
from gridworld_rl.reproducibility import sha256_file


def _checkpoint(tmp_path, name, action):
    model = QNetwork(5, 3, [4])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[2].bias[action] = 1
    path = tmp_path / name
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "num_states": 5,
            "num_actions": 3,
            "network": {"hidden_sizes": [4]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _inputs(tmp_path):
    checkpoints = [
        _checkpoint(tmp_path, "first.pt", 2),
        _checkpoint(tmp_path, "second.pt", 1),
        _checkpoint(tmp_path, "third.pt", 2),
    ]
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n4\n0\n")
    train = tmp_path / "train.csv"
    train.write_text("state,action,reward,next_state,done\n0,0,0,1,0\n")
    saved = tmp_path / "predictions.csv"
    predict_ensemble_csv(
        checkpoints,
        source,
        saved,
        train_csv=train,
        weights=[1, 4, 1],
        abstain_below=0.75,
        batch_size=2,
    )
    return checkpoints, source, train, saved


def test_audit_replays_supported_weighted_abstaining_predictions(tmp_path, capsys):
    checkpoints, source, train, saved = _inputs(tmp_path)
    result = audit_ensemble_predictions(
        checkpoints,
        source,
        saved,
        train_csv=train,
        weights=[1, 4, 1],
        abstain_below=0.75,
        batch_size=1,
    )
    assert result["matched"] is True
    assert result["verified_rows"] == 3
    assert result["inputs"]["predictions"]["sha256"] == sha256_file(saved)
    assert result["inputs"]["train"]["sha256"] == sha256_file(train)
    assert [item["sha256"] for item in result["checkpoints"]] == [
        sha256_file(path) for path in checkpoints
    ]
    assert (
        main(
            [
                "audit-ensemble-predictions",
                "--checkpoints",
                *map(str, checkpoints),
                "--input",
                str(source),
                "--predictions",
                str(saved),
                "--train",
                str(train),
                "--weights",
                "1",
                "4",
                "1",
                "--abstain-below",
                "0.75",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_audit_identifies_changed_prediction_column(tmp_path):
    checkpoints, source, train, saved = _inputs(tmp_path)
    with saved.open(newline="") as file:
        rows = list(csv.reader(file))
    column = rows[0].index("member_actions")
    rows[1][column] = "1;1;1"
    with saved.open("w", newline="") as file:
        csv.writer(file).writerows(rows)
    with pytest.raises(ValueError, match="row 0, column member_actions"):
        audit_ensemble_predictions(
            checkpoints,
            source,
            saved,
            train_csv=train,
            weights=[1, 4, 1],
            abstain_below=0.75,
        )


def test_audit_rejects_wrong_settings_and_truncated_rows(tmp_path):
    checkpoints, source, train, saved = _inputs(tmp_path)
    with pytest.raises(ValueError, match="header does not match replay"):
        audit_ensemble_predictions(checkpoints, source, saved, train_csv=train)
    with saved.open(newline="") as file:
        rows = list(csv.reader(file))
    with saved.open("w", newline="") as file:
        csv.writer(file).writerows(rows[:-1])
    with pytest.raises(ValueError, match="row count differs"):
        audit_ensemble_predictions(
            checkpoints,
            source,
            saved,
            train_csv=train,
            weights=[1, 4, 1],
            abstain_below=0.75,
        )


def test_audit_detects_changed_prediction_file(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_audit as audit

    checkpoints, source, train, saved = _inputs(tmp_path)
    original = audit.sha256_file
    calls = 0

    def changed(path):
        nonlocal calls
        if path == saved:
            calls += 1
            if calls == 2:
                return "changed"
        return original(path)

    monkeypatch.setattr(audit, "sha256_file", changed)
    with pytest.raises(ValueError, match="Predictions changed"):
        audit_ensemble_predictions(
            checkpoints,
            source,
            saved,
            train_csv=train,
            weights=[1, 4, 1],
            abstain_below=0.75,
        )
