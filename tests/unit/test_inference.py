from __future__ import annotations

import csv

import pandas as pd
import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.inference import predict_csv
from gridworld_rl.models import QNetwork


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
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "num_states": 5,
            "num_actions": 3,
            "network": {"hidden_sizes": [5]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_prediction_preserves_input_order_and_duplicates(tmp_path, batch_size):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n4\n0\n2\n4\n1\n3\n")
    output = tmp_path / "result/predictions.csv"
    assert predict_csv(checkpoint, source, output, batch_size=batch_size) == output
    with output.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames == [
            "state",
            "action",
            "action_gap",
            "q_0",
            "q_1",
            "q_2",
        ]
    assert [int(row["state"]) for row in rows] == [4, 0, 2, 4, 1, 3]
    assert [int(row["action"]) for row in rows] == [2, 0, 0, 2, 1, 0]
    assert [float(row["action_gap"]) for row in rows] == [3, 2, 0, 3, 4, 0]
    assert [float(rows[2][f"q_{i}"]) for i in range(3)] == [-1, -1, -2]
    assert list(output.parent.iterdir()) == [output]


def test_cli_predict_and_no_clobber(tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n1\n")
    output = tmp_path / "predictions.csv"
    argv = [
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(source),
        "--output",
        str(output),
        "--batch-size",
        "1",
    ]
    assert main(argv) == 0
    assert "Predictions exported" in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "body,error",
    [
        ("state\n", "at least one row"),
        ("state,action\n0,1\n", "exactly one column"),
        ("wrong\n0\n", "exactly one column"),
        ("state,state\n0,1\n", "exactly one column"),
        ("state\nfoo\n", "non-integer"),
        ("state\n1.5\n", "non-integer"),
        ("state\nNaN\n", "non-integer"),
        ("state\ninf\n", "non-integer"),
        ("state\n-1\n", "out-of-range"),
        ("state\n5\n", "out-of-range"),
    ],
)
def test_invalid_csv_does_not_create_output(tmp_path, body, error):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text(body)
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match=error):
        predict_csv(checkpoint, source, output)
    assert not output.exists()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        predict_csv(
            "missing.pt", "missing.csv", tmp_path / "output.csv", batch_size=batch_size
        )


def test_nonfinite_model_output_removes_temporary_file(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model_state_dict"]["network.0.weight"].fill_(1e38)
    payload["model_state_dict"]["network.2.weight"].fill_(1e38)
    torch.save(payload, checkpoint)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="non-finite Q-values"):
        predict_csv(checkpoint, source, output)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.pt",
        "states.csv",
    ]


def test_changed_checkpoint_aborts_output(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    hashes = iter(["before", "after"])
    monkeypatch.setattr("gridworld_rl.inference.sha256_file", lambda _: next(hashes))
    with pytest.raises(ValueError, match="changed during prediction"):
        predict_csv(checkpoint, source, tmp_path / "predictions.csv")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.pt",
        "states.csv",
    ]


def test_publication_race_preserves_competing_output(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    output = tmp_path / "predictions.csv"

    def competing_writer(_source, destination):
        destination.write_text("another writer")
        raise FileExistsError("race")

    monkeypatch.setattr("gridworld_rl.inference.os.link", competing_writer)
    with pytest.raises(FileExistsError):
        predict_csv(checkpoint, source, output)
    assert output.read_text() == "another writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.pt",
        "predictions.csv",
        "states.csv",
    ]


def test_accepts_integer_valued_numeric_state(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n1.0\n")
    output = tmp_path / "predictions.csv"
    predict_csv(checkpoint, source, output)
    assert pd.read_csv(output)["state"].tolist() == [1]
