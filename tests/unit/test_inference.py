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


@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_compact_prediction_keeps_actions_and_gaps(tmp_path, batch_size):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n4\n0\n2\n4\n1\n3\n")
    output = tmp_path / "compact.csv"
    predict_csv(checkpoint, source, output, batch_size=batch_size, compact=True)
    with output.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames == ["state", "action", "action_gap"]
    assert [int(row["state"]) for row in rows] == [4, 0, 2, 4, 1, 3]
    assert [int(row["action"]) for row in rows] == [2, 0, 0, 2, 1, 0]
    assert [float(row["action_gap"]) for row in rows] == [3, 2, 0, 3, 4, 0]


def test_cli_compact_prediction(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n1\n")
    output = tmp_path / "compact.csv"
    assert (
        main(
            [
                "predict",
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(source),
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )
    assert output.read_text().splitlines() == [
        "state,action,action_gap",
        "1,1,4.0",
    ]


def _training_csv(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text(
        "state,action,reward,next_state,done\n"
        "0,1,0,1,0\n0,2,0,1,0\n1,0,0,2,0\n"
        "2,1,0,3,0\n2,2,0,3,0\n3,0,0,4,1\n3,1,0,4,1\n"
    )
    return train


@pytest.mark.parametrize("compact", [False, True])
def test_prediction_can_restrict_to_logged_actions(tmp_path, capsys, compact):
    checkpoint = _checkpoint(tmp_path)
    train = _training_csv(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n4\n0\n1\n2\n3\n4\n")
    output = tmp_path / "predictions.csv"
    args = [
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(source),
        "--output",
        str(output),
        "--train",
        str(train),
        "--batch-size",
        "2",
    ]
    if compact:
        args.append("--compact")
    assert main(args) == 0
    capsys.readouterr()
    with output.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames == [
            "state",
            "action",
            "supported_action_gap",
            "unconstrained_action",
            "unconstrained_action_gap",
            "was_constrained",
            "logged_actions",
            *([] if compact else ["q_0", "q_1", "q_2"]),
        ]
    assert [int(row["state"]) for row in rows] == [4, 0, 1, 2, 3, 4]
    assert [int(row["action"]) for row in rows] == [2, 1, 0, 1, 0, 2]
    assert [int(row["unconstrained_action"]) for row in rows] == [2, 0, 1, 0, 0, 2]
    assert [row["was_constrained"] for row in rows] == [
        "False",
        "True",
        "True",
        "True",
        "False",
        "False",
    ]
    assert [row["supported_action_gap"] for row in rows] == [
        "3.0",
        "1.0",
        "",
        "1.0",
        "0.0",
        "3.0",
    ]
    assert [row["logged_actions"] for row in rows] == ["", "1;2", "0", "1;2", "0;1", ""]
    if not compact:
        assert [float(rows[1][f"q_{i}"]) for i in range(3)] == [3, 1, 0]


def test_changed_training_csv_aborts_supported_prediction(tmp_path, monkeypatch):
    import gridworld_rl.inference as inference

    checkpoint = _checkpoint(tmp_path)
    train = _training_csv(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    original = inference._write_batch

    def write_then_change(*args, **kwargs):
        original(*args, **kwargs)
        train.write_text(train.read_text() + "4,2,0,4,1\n")

    monkeypatch.setattr(inference, "_write_batch", write_then_change)
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="Training CSV changed"):
        predict_csv(checkpoint, source, output, train_csv=train, batch_size=1)
    assert not output.exists()


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
        ("state\n1.0000000000000001\n", "non-integer"),
        ("state\n4.9999999999999999\n", "non-integer"),
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
    from gridworld_rl.reproducibility import sha256_file

    def changed_checkpoint_hash(path):
        return next(hashes) if path == checkpoint else sha256_file(path)

    monkeypatch.setattr("gridworld_rl.inference.sha256_file", changed_checkpoint_hash)
    with pytest.raises(ValueError, match="changed during prediction"):
        predict_csv(checkpoint, source, tmp_path / "predictions.csv")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.pt",
        "states.csv",
    ]


def test_changed_state_csv_aborts_output(tmp_path, monkeypatch):
    import gridworld_rl.inference as inference

    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    original = inference._write_batch

    def write_then_change(*args, **kwargs):
        original(*args, **kwargs)
        source.write_text("state\n1\n")

    monkeypatch.setattr(inference, "_write_batch", write_then_change)
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="State CSV changed during prediction"):
        predict_csv(checkpoint, source, output, batch_size=1)
    assert not output.exists()
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
    source.write_text("state\n1.0000000000000000\n")
    output = tmp_path / "predictions.csv"
    predict_csv(checkpoint, source, output)
    assert pd.read_csv(output)["state"].tolist() == [1]


def test_streams_many_rows_in_bounded_batches(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n" + "".join(f"{index % 5}\n" for index in range(257)))
    original_forward = QNetwork.forward
    lengths = []

    def recording_forward(model, states):
        lengths.append(len(states))
        return original_forward(model, states)

    monkeypatch.setattr(QNetwork, "forward", recording_forward)
    output = tmp_path / "predictions.csv"
    predict_csv(checkpoint, source, output, batch_size=16)
    assert lengths == [16] * 16 + [1]
    assert len(pd.read_csv(output)) == 257


def test_late_invalid_row_does_not_publish_partial_predictions(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    source = tmp_path / "states.csv"
    source.write_text("state\n" + "0\n" * 32 + "bad\n")
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="row 32"):
        predict_csv(checkpoint, source, output, batch_size=8)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.pt",
        "states.csv",
    ]
