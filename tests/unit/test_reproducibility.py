from __future__ import annotations

import gridworld_rl.reproducibility as reproducibility
from gridworld_rl.reproducibility import source_revision


def test_source_revision_does_not_fall_back_to_an_unrelated_caller_repo(
    tmp_path,
) -> None:
    assert source_revision(tmp_path) == {
        "git_commit": None,
        "git_dirty": None,
    }


def test_source_revision_rejects_a_wheel_inside_an_unrelated_repo(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / ".git").mkdir()
    installed_module = (
        tmp_path / ".venv/lib/python3.11/site-packages/gridworld_rl/reproducibility.py"
    )
    installed_module.parent.mkdir(parents=True)
    installed_module.write_text("# simulated installed wheel")
    monkeypatch.setattr(
        reproducibility,
        "__file__",
        str(installed_module),
    )

    assert source_revision() == {
        "git_commit": None,
        "git_dirty": None,
    }
