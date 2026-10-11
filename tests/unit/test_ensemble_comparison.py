from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.ensemble_comparison import compare_ensemble_predictions
from gridworld_rl.reproducibility import sha256_file


def _files(tmp_path):
    header = "state,action,reward,next_state,done\n"
    challenge = tmp_path / "challenge.csv"
    solution = tmp_path / "solution.csv"
    challenge.write_text(header + "0,-1,0,1,0\n1,-1,0,2,0\n2,-1,0,3,0\n3,-1,0,4,1\n")
    solution.write_text(header + "0,0,0,1,0\n1,1,0,2,0\n2,0,0,3,0\n3,1,0,4,1\n")
    prediction_header = "state,action,votes,agreement_fraction,unanimous,member_actions,suggested_action,abstained,votes_0,votes_1\n"
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text(
        prediction_header
        + "0,0,2,1.0,True,0;0,0,False,2,0\n"
        + "1,0,2,1.0,True,0;0,0,False,2,0\n"
        + "2,0,2,1.0,True,0;0,0,False,2,0\n"
        + "3,,1,0.5,False,0;1,0,True,1,1\n"
    )
    right.write_text(
        prediction_header
        + "0,1,2,1.0,True,1;1,1,False,0,2\n"
        + "1,,1,0.5,False,0;1,0,True,1,1\n"
        + "2,0,2,1.0,True,0;0,0,False,2,0\n"
        + "3,1,2,1.0,True,1;1,1,False,0,2\n"
    )
    return left, right, challenge, solution


def test_compare_ensembles_reports_shared_accuracy_and_coverage(tmp_path, capsys):
    left, right, challenge, solution = _files(tmp_path)
    result = compare_ensemble_predictions(left, right, challenge, solution)
    assert result["num_rows"] == 4
    assert result["both_accepted"] == 2
    assert result["left_only_accepted"] == 1
    assert result["right_only_accepted"] == 1
    assert result["neither_accepted"] == 0
    assert result["coverage_difference"] == 0
    assert result["left"]["selective_accuracy"] == pytest.approx(2 / 3)
    assert result["right"]["selective_accuracy"] == pytest.approx(2 / 3)
    assert result["shared_left_accuracy"] == 1.0
    assert result["shared_right_accuracy"] == 0.5
    assert result["shared_accuracy_difference"] == 0.5
    assert result["shared_action_disagreements"] == 1
    assert result["suggested_action_disagreements"] == 2
    assert result["inputs"]["left"]["sha256"] == sha256_file(left)
    assert (
        main(
            [
                "compare-ensemble-predictions",
                "--left",
                str(left),
                "--right",
                str(right),
                "--challenge",
                str(challenge),
                "--solution",
                str(solution),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_compare_ensembles_bootstrap_is_deterministic(tmp_path):
    left, right, challenge, solution = _files(tmp_path)
    result = compare_ensemble_predictions(
        left,
        right,
        challenge,
        solution,
        bootstrap_replicates=100,
        bootstrap_seed=9,
        confidence_level=0.9,
    )
    bootstrap = result["bootstrap"]
    assert bootstrap["unique_states"] == 4
    assert bootstrap["coverage_difference_interval"]["estimate"] == 0
    assert bootstrap["shared_accuracy_difference_interval"]["estimate"] == 0.5
    assert bootstrap["shared_accuracy_replicates"] <= 100
    assert (
        bootstrap
        == compare_ensemble_predictions(
            left,
            right,
            challenge,
            solution,
            bootstrap_replicates=100,
            bootstrap_seed=9,
            confidence_level=0.9,
        )["bootstrap"]
    )


def test_compare_ensembles_bootstrap_clusters_duplicate_states(tmp_path):
    left, right, challenge, solution = _files(tmp_path)
    challenge.write_text(challenge.read_text().replace("3,-1,0,4,1", "0,-1,0,4,1"))
    solution.write_text(solution.read_text().replace("3,1,0,4,1", "0,1,0,4,1"))
    left.write_text(left.read_text().replace("3,,1,0.5", "0,,1,0.5"))
    right.write_text(right.read_text().replace("3,1,2,1.0", "0,1,2,1.0"))
    result = compare_ensemble_predictions(
        left, right, challenge, solution, bootstrap_replicates=20
    )
    assert result["bootstrap"]["unique_states"] == 3


def test_compare_ensembles_handles_no_shared_accepted_rows(tmp_path):
    left, right, challenge, solution = _files(tmp_path)
    left.write_text(
        left.read_text()
        .replace(
            "2,0,2,1.0,True,0;0,0,False,2,0",
            "2,,1,0.5,False,0;1,0,True,1,1",
        )
        .replace(
            "0,0,2,1.0,True,0;0,0,False,2,0",
            "0,,1,0.5,False,0;1,0,True,1,1",
        )
    )
    right.write_text(
        right.read_text().replace(
            "2,0,2,1.0,True,0;0,0,False,2,0",
            "2,,1,0.5,False,0;1,0,True,1,1",
        )
    )
    result = compare_ensemble_predictions(
        left, right, challenge, solution, bootstrap_replicates=20
    )
    assert result["both_accepted"] == 0
    assert result["shared_accuracy_difference"] is None
    assert result["bootstrap"]["shared_accuracy_replicates"] == 0
    assert result["bootstrap"]["shared_accuracy_difference_interval"] is None


def test_compare_ensembles_rejects_same_file(tmp_path):
    left, _, challenge, solution = _files(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        compare_ensemble_predictions(left, left, challenge, solution)
