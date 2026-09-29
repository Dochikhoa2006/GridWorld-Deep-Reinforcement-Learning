"""Read-only verification of saved experiment and benchmark artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .reproducibility import sha256_file


@dataclass(frozen=True)
class VerificationResult:
    checked_files: int
    missing: tuple[str, ...]
    modified: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (self.missing or self.modified or self.unexpected)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def verify_artifacts(directory: str | Path) -> VerificationResult:
    """Check every listed hash and detect extra files without modifying artifacts.

    Manifest paths must be canonical relative POSIX paths. Symlinks anywhere
    inside the artifact directory are rejected rather than followed.
    """

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Artifact manifest must not be a symbolic link.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid artifact manifest: {manifest_path}: {exc}") from exc
    hashes = payload.get("sha256") if isinstance(payload, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Artifact manifest must contain a non-empty sha256 object.")

    for name, digest in hashes.items():
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or str(path) != name
            or ".." in path.parts
            or name in {".", "manifest.json"}
            or any(character in name for character in ("\\", ":", "\0"))
        ):
            raise ValueError(f"Unsafe artifact manifest path: {name!r}")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            raise ValueError(f"Invalid SHA-256 digest for artifact: {name}")

    actual: set[str] = set()

    def fail_walk(error: OSError) -> None:
        raise error

    for parent, directories, files in os.walk(
        root, followlinks=False, onerror=fail_walk
    ):
        for name in [*directories, *files]:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(
                    f"Symbolic links are not supported in artifacts: {path.relative_to(root)}"
                )
        for name in files:
            path = Path(parent) / name
            if not path.is_file():
                raise ValueError(f"Artifact is not a regular file: {path}")
            actual.add(path.relative_to(root).as_posix())
    actual.discard("manifest.json")
    expected = set(hashes)
    checked = sorted(actual & expected)
    modified = tuple(
        name for name in checked if sha256_file(root / name) != hashes[name].lower()
    )
    return VerificationResult(
        checked_files=len(checked),
        missing=tuple(sorted(expected - actual)),
        modified=modified,
        unexpected=tuple(sorted(actual - expected)),
    )
