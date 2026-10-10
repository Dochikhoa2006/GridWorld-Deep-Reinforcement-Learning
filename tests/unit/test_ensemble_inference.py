from __future__ import annotations

import csv

import pytest
import torch

from gridworld_rl.cli import main
from gridworld_rl.ensemble_inference import predict_ensemble_csv
from gridworld_rl.models import QNetwork


def _checkpoint(tmp_path, name, favored, *, num_states=5):
    model = QNetwork(num_states, 3, [4])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[2].bias[favored] = 1
    path = tmp_path / name
    torch.save(
        {
            "format_version": 1,
            "algorithm": "dqn",
            "num_states": num_states,
            "num_actions": 3,
            "network": {"hidden_sizes": [4]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_cli_ensemble_majority_and_tie_preserve_input_order(tmp_path, capsys):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    source = tmp_path / "states.csv"
    source.write_text("state\n4\n0\n4\n")
    output = tmp_path / "result.csv"
    assert (
        main(
            [
                "predict-ensemble",
                "--checkpoints",
                str(first),
                str(second),
                str(third),
                "--input",
                str(source),
                "--output",
                str(output),
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    assert "Ensemble predictions exported" in capsys.readouterr().out
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["state"] for row in rows] == ["4", "0", "4"]
    assert [row["action"] for row in rows] == ["2"] * 3
    assert rows[0]["member_actions"] == "2;1;2"
    assert [rows[0][f"votes_{i}"] for i in range(3)] == ["0", "1", "2"]
    assert rows[0]["agreement_fraction"] == str(2 / 3)
    assert rows[0]["unanimous"] == "False"
    tied = tmp_path / "tied.csv"
    predict_ensemble_csv([first, second], source, tied, batch_size=1)
    with tied.open(newline="") as file:
        assert [row["action"] for row in csv.DictReader(file)] == ["1"] * 3


def test_supported_ensemble_filters_each_vote_and_reports_fallback(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    train = tmp_path / "train.csv"
    train.write_text(
        "state,action,reward,next_state,done\n"
        "0,0,0,1,0\n0,1,0,1,0\n0,1,0,1,0\n0,1,0,1,0\n"
        "1,0,0,2,0\n"
    )
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n1\n4\n0\n")
    output = tmp_path / "out.csv"
    assert (
        main(
            [
                "predict-ensemble",
                "--checkpoints",
                str(first),
                str(second),
                str(third),
                "--input",
                str(source),
                "--output",
                str(output),
                "--train",
                str(train),
                "--min-action-count",
                "2",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["action"] for row in rows] == ["1", "0", "2", "1"]
    assert [row["unconstrained_action"] for row in rows] == ["2"] * 4
    assert [row["member_actions"] for row in rows] == [
        "1;1;1",
        "0;0;0",
        "2;1;2",
        "1;1;1",
    ]
    assert rows[0]["unconstrained_member_actions"] == "2;1;2"
    assert [row["support_fallback"] for row in rows] == [
        "False",
        "True",
        "False",
        "False",
    ]
    assert [row["eligible_actions"] for row in rows] == ["1", "0", "", "1"]
    assert rows[0]["logged_action_counts"] == "0:1;1:3"
    assert rows[2]["was_constrained"] == "False"


def test_weighted_votes_can_outweigh_member_majority(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    output = tmp_path / "out.csv"
    assert (
        main(
            [
                "predict-ensemble",
                "--checkpoints",
                str(first),
                str(second),
                str(third),
                "--weights",
                "1",
                "4",
                "1",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    with output.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["action"] == "1"
    assert row["votes"] == "1"
    assert row["vote_weight"] == "4.0"
    assert row["agreement_fraction"] == str(1 / 3)
    assert row["agreement_weight_fraction"] == str(4 / 6)
    assert [row[f"weight_{i}"] for i in range(3)] == ["0.0", "4.0", "2.0"]
    assert row["unanimous"] == "False"


def test_weighted_supported_ensemble_uses_weights_for_raw_and_supported_winners(
    tmp_path,
):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    train = tmp_path / "train.csv"
    train.write_text("state,action,reward,next_state,done\n0,0,0,1,0\n")
    output = tmp_path / "out.csv"
    predict_ensemble_csv(
        [first, second, third], source, output, train_csv=train, weights=[1, 4, 1]
    )
    with output.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["action"] == "0"
    assert row["unconstrained_action"] == "1"
    assert row["was_constrained"] == "True"
    assert row["vote_weight"] == "6.0"
    assert row["agreement_weight_fraction"] == "1.0"


def test_ensemble_abstains_below_member_agreement_threshold(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    output = tmp_path / "out.csv"
    assert (
        main(
            [
                "predict-ensemble",
                "--checkpoints",
                str(first),
                str(second),
                str(third),
                "--input",
                str(source),
                "--output",
                str(output),
                "--abstain-below",
                "0.75",
            ]
        )
        == 0
    )
    with output.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["action"] == ""
    assert row["suggested_action"] == "2"
    assert row["abstained"] == "True"
    assert row["votes"] == "2"
    accepted = tmp_path / "accepted.csv"
    predict_ensemble_csv([first, second, third], source, accepted, abstain_below=2 / 3)
    with accepted.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["action"] == "2"
    assert row["abstained"] == "False"


def test_weighted_supported_ensemble_abstains_on_weight_share(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 2)
    second = _checkpoint(tmp_path, "second.pt", 1)
    third = _checkpoint(tmp_path, "third.pt", 2)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n1\n")
    train = tmp_path / "train.csv"
    train.write_text("state,action,reward,next_state,done\n0,0,0,1,0\n")
    output = tmp_path / "out.csv"
    predict_ensemble_csv(
        [first, second, third],
        source,
        output,
        train_csv=train,
        weights=[1, 4, 1],
        abstain_below=0.75,
    )
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["action"] for row in rows] == ["0", ""]
    assert [row["suggested_action"] for row in rows] == ["0", "1"]
    assert [row["abstained"] for row in rows] == ["False", "True"]
    assert rows[1]["agreement_weight_fraction"] == str(4 / 6)
    assert rows[1]["agreement_fraction"] == str(1 / 3)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf"), True])
def test_invalid_abstention_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match="abstain_below"):
        predict_ensemble_csv(
            ["first.pt", "second.pt"],
            "states.csv",
            tmp_path / "out.csv",
            abstain_below=threshold,
        )


@pytest.mark.parametrize(
    "weights",
    [
        [1],
        [1, 0],
        [1, -1],
        [1, float("nan")],
        [1, float("inf")],
        [1e308, 1e308],
        [True, 1],
    ],
)
def test_invalid_vote_weights_are_rejected(tmp_path, weights):
    with pytest.raises(ValueError, match="weights"):
        predict_ensemble_csv(
            ["first.pt", "second.pt"],
            "states.csv",
            tmp_path / "out.csv",
            weights=weights,
        )


def test_supported_ensemble_detects_changed_training_csv(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_inference as ensemble

    first = _checkpoint(tmp_path, "first.pt", 0)
    second = _checkpoint(tmp_path, "second.pt", 1)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    train = tmp_path / "train.csv"
    train.write_text("state,action,reward,next_state,done\n0,0,0,1,0\n")
    original = ensemble._write_ensemble_batch

    def write_then_change(*args):
        original(*args)
        train.write_text(train.read_text() + "0,1,0,1,0\n")

    monkeypatch.setattr(ensemble, "_write_ensemble_batch", write_then_change)
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Training CSV changed"):
        predict_ensemble_csv([first, second], source, output, train_csv=train)
    assert not output.exists()


def test_ensemble_rejects_incompatible_or_duplicate_checkpoints(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 0)
    other = _checkpoint(tmp_path, "other.pt", 1, num_states=6)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="distinct"):
        predict_ensemble_csv([first, first], source, output)
    with pytest.raises(ValueError, match="identical state and action dimensions"):
        predict_ensemble_csv([first, other], source, output)
    assert not output.exists()


def test_ensemble_changed_input_aborts_publication(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_inference as ensemble

    first = _checkpoint(tmp_path, "first.pt", 0)
    second = _checkpoint(tmp_path, "second.pt", 1)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    original = ensemble._write_ensemble_batch

    def write_then_change(*args):
        original(*args)
        source.write_text("state\n1\n")

    monkeypatch.setattr(ensemble, "_write_ensemble_batch", write_then_change)
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="State CSV changed"):
        predict_ensemble_csv([first, second], source, output)
    assert not output.exists()
    assert list(tmp_path.glob(".out.csv.*.tmp")) == []


def test_ensemble_changed_checkpoint_aborts_publication(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_inference as ensemble

    first = _checkpoint(tmp_path, "first.pt", 0)
    second = _checkpoint(tmp_path, "second.pt", 1)
    source = tmp_path / "states.csv"
    source.write_text("state\n0\n")
    original = ensemble._write_ensemble_batch

    def write_then_change(*args):
        original(*args)
        second.write_bytes(second.read_bytes() + b"changed")

    monkeypatch.setattr(ensemble, "_write_ensemble_batch", write_then_change)
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Checkpoint changed"):
        predict_ensemble_csv([first, second], source, output)
    assert not output.exists()


def test_ensemble_rejects_invalid_state_csv_without_output(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 0)
    second = _checkpoint(tmp_path, "second.pt", 1)
    source = tmp_path / "states.csv"
    source.write_text("state\n5\n")
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="out-of-range"):
        predict_ensemble_csv([first, second], source, output)
    assert not output.exists()
