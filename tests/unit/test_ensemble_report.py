from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.ensemble_report import (
    audit_ensemble_evaluation_report,
    save_ensemble_evaluation,
)
from gridworld_rl.reproducibility import sha256_file


def _files(tmp_path):
    header = "state,action,reward,next_state,done\n"
    challenge = tmp_path / "challenge.csv"
    solution = tmp_path / "solution.csv"
    train = tmp_path / "train.csv"
    predictions = tmp_path / "predictions.csv"
    challenge.write_text(header + "0,-1,0,1,0\n1,-1,0,2,1\n")
    solution.write_text(header + "0,0,0,1,0\n1,1,0,2,1\n")
    train.write_text(header + "0,0,0,1,0\n")
    predictions.write_text(
        "state,action,votes,agreement_fraction,unanimous,member_actions,suggested_action,abstained,votes_0,votes_1\n"
        "0,0,2,1.0,True,0;0,0,False,2,0\n"
        "1,,1,0.5,False,0;1,0,True,1,1\n"
    )
    return predictions, challenge, solution, train


def test_cli_saves_and_audits_ensemble_evaluation(tmp_path, capsys):
    predictions, challenge, solution, train = _files(tmp_path)
    output = tmp_path / "reports/evaluation.json"
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
                "--train",
                str(train),
                "--target-coverage",
                "0.5",
                "--bootstrap-replicates",
                "20",
                "--bootstrap-seed",
                "7",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Ensemble evaluation saved" in capsys.readouterr().out
    saved = json.loads(output.read_text())
    assert saved["num_rows"] == 2
    assert saved["inputs"]["predictions"]["path"] == str(predictions.resolve())
    assert saved["bootstrap"]["replicates"] == 20
    assert saved["recommended_threshold"]["target_coverage"] == 0.5
    audit = audit_ensemble_evaluation_report(output)
    assert audit["valid"] is True
    assert audit["verified_rows"] == 2
    assert audit["report"]["sha256"] == sha256_file(output)
    assert main(["audit-ensemble-report", "--report", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == audit


def test_saved_ensemble_evaluation_is_no_clobber(tmp_path):
    predictions, challenge, solution, _ = _files(tmp_path)
    output = tmp_path / "evaluation.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        save_ensemble_evaluation(output, predictions, challenge, solution)
    assert output.read_text() == "keep"


def test_report_audit_detects_changed_metrics_and_inputs(tmp_path):
    predictions, challenge, solution, _ = _files(tmp_path)
    output = tmp_path / "evaluation.json"
    save_ensemble_evaluation(output, predictions, challenge, solution)
    saved = json.loads(output.read_text())
    saved["coverage"] = 0.75
    output.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"report\.coverage"):
        audit_ensemble_evaluation_report(output)
    saved["coverage"] = 0.5
    output.write_text(json.dumps(saved))
    challenge.write_text(challenge.read_text().replace("0,-1,0,1,0", "0,-1,1,1,0"))
    with pytest.raises(ValueError, match=r"report\.inputs\.challenge\.sha256"):
        audit_ensemble_evaluation_report(output)


def test_report_publication_aborts_when_input_changes(tmp_path, monkeypatch):
    import gridworld_rl.ensemble_report as report

    predictions, challenge, solution, _ = _files(tmp_path)
    original = report.sha256_file

    def changed(path):
        if str(path) == str(predictions.resolve()):
            return "changed"
        return original(path)

    monkeypatch.setattr(report, "sha256_file", changed)
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="Predictions changed"):
        save_ensemble_evaluation(output, predictions, challenge, solution)
    assert not output.exists()
    assert list(tmp_path.glob(".evaluation.json.*.tmp")) == []
