from __future__ import annotations

import json

import pytest

from gridworld_rl.cli import main
from gridworld_rl.integrity import verify_artifacts
from gridworld_rl.reproducibility import sha256_file


def _artifact_directory(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/model.pt").write_bytes(b"model weights")
    (tmp_path / "metrics.json").write_text('{"accuracy": 0.5}')
    hashes = {
        name: sha256_file(tmp_path / name)
        for name in ("checkpoints/model.pt", "metrics.json")
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    return tmp_path


def test_valid_artifacts_are_verified_without_changes(tmp_path, capsys):
    root = _artifact_directory(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = verify_artifacts(root)
    assert result.valid
    assert result.checked_files == 2
    assert main(["verify", "--run-dir", str(root)]) == 0
    assert "2 files checked" in capsys.readouterr().out
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_all_integrity_failures_are_reported(tmp_path, capsys):
    root = _artifact_directory(tmp_path)
    (root / "metrics.json").write_text("tampered")
    (root / "checkpoints/model.pt").unlink()
    (root / "extra.txt").write_text("unexpected")
    result = verify_artifacts(root)
    assert not result.valid
    assert result.checked_files == 1
    assert result.missing == ("checkpoints/model.pt",)
    assert result.modified == ("metrics.json",)
    assert result.unexpected == ("extra.txt",)
    assert main(["verify", "--run-dir", str(root)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "missing: checkpoints/model.pt",
        "modified: metrics.json",
        "unexpected: extra.txt",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "../outside",
        "/absolute",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "C:/file",
        "a\\b",
        "a\0b",
        "manifest.json",
    ],
)
def test_rejects_unsafe_paths_before_hashing(tmp_path, name, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {name: "0" * 64}}))

    def unexpected_read(path):
        pytest.fail(f"Attempted to hash unsafe manifest: {path}")

    monkeypatch.setattr("gridworld_rl.integrity.sha256_file", unexpected_read)
    with pytest.raises(ValueError, match="Unsafe"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize("payload", [[], {}, {"sha256": []}, {"sha256": {}}])
def test_rejects_invalid_manifest_structure(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-empty sha256 object"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize("digest", [None, 123, "g" * 64, "0" * 63])
def test_rejects_invalid_digest(tmp_path, digest):
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"file": digest}}))
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize(
    "content,message",
    [
        (b"{", "Invalid artifact manifest"),
        (b"\xff", "Invalid artifact manifest"),
        (b'{"sha256": {}, "sha256": {}}', "Duplicate manifest key"),
        (b'{"sha256": {"a": "x", "a": "y"}}', "Duplicate manifest key"),
    ],
)
def test_rejects_unreadable_or_ambiguous_manifest(tmp_path, content, message):
    (tmp_path / "manifest.json").write_bytes(content)
    with pytest.raises(ValueError, match=message):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize("kind", ["file", "directory", "manifest", "dangling"])
def test_rejects_symbolic_links(tmp_path, kind):
    root = _artifact_directory(tmp_path)
    if kind == "manifest":
        (root / "manifest.json").rename(root / "saved.json")
        (root / "manifest.json").symlink_to(root / "saved.json")
    else:
        target = {
            "file": root / "metrics.json",
            "directory": root / "checkpoints",
            "dangling": root / "absent",
        }[kind]
        (root / "link").symlink_to(target)
    with pytest.raises(ValueError, match=r"[Ss]ymbolic link"):
        verify_artifacts(root)


def test_nested_manifests_are_checked_and_uppercase_digests_accepted(tmp_path):
    nested = tmp_path / "runs/seed-1"
    nested.mkdir(parents=True)
    _artifact_directory(nested)
    hashes = {
        p.relative_to(tmp_path).as_posix(): sha256_file(p).upper()
        for p in nested.rglob("*")
        if p.is_file()
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    assert verify_artifacts(tmp_path).checked_files == 3
    assert verify_artifacts(tmp_path).valid
    (nested / "manifest.json").write_text("{}")
    assert verify_artifacts(tmp_path).modified == ("runs/seed-1/manifest.json",)


def test_missing_manifest_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["verify", "--run-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "Artifact manifest not found" in capsys.readouterr().err


def test_directory_does_not_substitute_for_expected_file(tmp_path):
    root = _artifact_directory(tmp_path)
    (root / "metrics.json").unlink()
    (root / "metrics.json").mkdir()
    assert verify_artifacts(root).missing == ("metrics.json",)
