"""Portable artifacts for quantitative experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy


QUANTITATIVE_RESULT_FORMAT = "somehand.quantitative_result.v1"
QUANTITATIVE_ANNOTATION_FORMAT = "somehand.quantitative_annotations.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment_metadata(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    git_commit = _git_output(root, "rev-parse", "HEAD")
    git_status = _git_output(root, "status", "--short")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mujoco": mujoco.__version__,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines() if git_status else [],
    }


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def save_quantitative_result(
    output_base: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    base = Path(output_base)
    if base.suffix in {".npz", ".json"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    metadata_path = base.with_suffix(".json")

    np.savez_compressed(npz_path, **arrays)
    payload = dict(metadata)
    payload["format"] = QUANTITATIVE_RESULT_FORMAT
    payload["arrays_path"] = npz_path.name
    payload["arrays_sha256"] = sha256_file(npz_path)
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    return npz_path, metadata_path


def load_quantitative_result(
    metadata_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata_file = Path(metadata_path)
    with metadata_file.open(encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    if metadata.get("format") != QUANTITATIVE_RESULT_FORMAT:
        raise ValueError(f"Unsupported quantitative result format: {metadata.get('format')!r}")

    arrays_path = metadata_file.parent / str(metadata["arrays_path"])
    expected_hash = str(metadata["arrays_sha256"])
    actual_hash = sha256_file(arrays_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Quantitative result hash mismatch for {arrays_path}: expected {expected_hash}, got {actual_hash}"
        )
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    return arrays, metadata


def save_annotations(path: str | Path, payload: dict[str, object]) -> Path:
    annotation_path = Path(path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = dict(payload)
    serialized["format"] = QUANTITATIVE_ANNOTATION_FORMAT
    with annotation_path.open("w", encoding="utf-8") as file_obj:
        json.dump(serialized, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    return annotation_path


def load_annotations(path: str | Path) -> dict[str, object]:
    annotation_path = Path(path)
    with annotation_path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if payload.get("format") != QUANTITATIVE_ANNOTATION_FORMAT:
        raise ValueError(f"Unsupported quantitative annotation format: {payload.get('format')!r}")
    return payload
