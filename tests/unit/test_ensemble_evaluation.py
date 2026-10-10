from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.ensemble_evaluation import evaluate_ensemble_csv
from gridworld_rl.reproducibility import sha256_file


def _files(tmp_path):
    transition_header = "state,action,reward,next_state,done\n"
    challenge = tmp_path / "challenge.csv"
    solution = tmp_path / "solution.csv"
    challenge.write_text(transition_header + "0,-1,0,1,0\n1,-1,0,2,0\n2,-1,0,3,1\n")
    solution.write_text(transition_header + "0,0,0,1,0\n1,0,0,2,0\n2,0,0,3,1\n")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "state,action,votes,agreement_fraction,unanimous,member_actions,suggested_action,abstained,votes_0,votes_1\n"
        "0,0,3,1.0,True,0;0;0,0,False,3,0\n"
        "1,,2,0.6666666666666666,False,1;1;0,1,True,1,2\n"
        "2,0,2,0.6666666666666666,False,0;0;1,0,False,2,1\n"
    )
    return predictions, challenge, solution


def test_evaluate_ensemble_reports_coverage_and_selective_accuracy(tmp_path, capsys):
    predictions, challenge, solution = _files(tmp_path)
    result = evaluate_ensemble_csv(predictions, challenge, solution)
    assert result["accepted_rows"] == 2
    assert result["abstained_rows"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["selective_accuracy"] == 1.0
    assert result["suggested_accuracy"] == pytest.approx(2 / 3)
    assert result["accepted_metrics"]["confusion_matrix"] == [[2, 0], [0, 0]]
    assert result["mean_agreement_accepted"] == pytest.approx(5 / 6)
    assert result["inputs"]["predictions"]["sha256"] == sha256_file(predictions)
    assert (
        main(
            [
                "evaluate-ensemble",
                "--predictions",
                str(predictions),
                "--challenge",
                str(challenge),
                "--solution",
                str(solution),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_evaluate_ensemble_handles_all_abstained(tmp_path):
    predictions, challenge, solution = _files(tmp_path)
    body = predictions.read_text().replace(
        "0,0,3,1.0,True,0;0;0,0,False,3,0",
        "0,,2,0.6666666666666666,False,0;0;1,0,True,2,1",
    )
    body = body.replace(
        "2,0,2,0.6666666666666666,False,0;0;1,0,False",
        "2,,2,0.6666666666666666,False,0;0;1,0,True",
    )
    predictions.write_text(body)
    result = evaluate_ensemble_csv(predictions, challenge, solution)
    assert result["coverage"] == 0
    assert result["selective_accuracy"] is None
    assert result["accepted_metrics"] is None
    assert result["mean_agreement_accepted"] is None


def test_evaluate_weighted_ensemble_uses_weighted_winner(tmp_path):
    predictions, challenge, solution = _files(tmp_path)
    predictions.write_text(
        "state,action,votes,agreement_fraction,unanimous,member_actions,vote_weight,agreement_weight_fraction,"
        "suggested_action,abstained,votes_0,votes_1,weight_0,weight_1\n"
        "0,1,1,0.3333333333333333,False,0;1;0,4.0,0.6666666666666666,1,False,2,1,2.0,4.0\n"
        "1,1,1,0.3333333333333333,False,0;1;0,4.0,0.6666666666666666,1,False,2,1,2.0,4.0\n"
        "2,1,1,0.3333333333333333,False,0;1;0,4.0,0.6666666666666666,1,False,2,1,2.0,4.0\n"
    )
    result = evaluate_ensemble_csv(predictions, challenge, solution)
    assert result["coverage"] == 1
    assert result["selective_accuracy"] == 0


@pytest.mark.parametrize(
    "old,new,error",
    [
        ("2,0,2,", "4,0,2,", "state mismatch"),
        ("1,,2,", "1,0,2,", "Abstention mismatch"),
        ("0;0;1,0,False,2,1", "0;0;1,0,False,1,2", "vote counts mismatch"),
        ("1;1;0,1,True,1,2", "1;1;0,0,True,1,2", "winner mismatch"),
    ],
)
def test_evaluate_ensemble_rejects_corrupt_rows(tmp_path, old, new, error):
    predictions, challenge, solution = _files(tmp_path)
    predictions.write_text(predictions.read_text().replace(old, new))
    with pytest.raises(ValueError, match=error):
        evaluate_ensemble_csv(predictions, challenge, solution)


def test_evaluate_ensemble_rejects_row_count_mismatch(tmp_path):
    predictions, challenge, solution = _files(tmp_path)
    predictions.write_text("\n".join(predictions.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="fewer rows"):
        evaluate_ensemble_csv(predictions, challenge, solution)


def test_evaluate_ensemble_detects_changed_input(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_evaluation as evaluation

    predictions, challenge, solution = _files(tmp_path)
    original = evaluation.sha256_file
    calls = 0

    def changed(path):
        nonlocal calls
        if path == solution:
            calls += 1
            if calls == 2:
                return "changed"
        return original(path)

    monkeypatch.setattr(evaluation, "sha256_file", changed)
    with pytest.raises(ValueError, match="Solution changed"):
        evaluate_ensemble_csv(predictions, challenge, solution)
