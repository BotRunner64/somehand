"""Project-local external asset manifest and helper utilities."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "configs" / "retargeting").is_dir():
        return candidate
    return None


def _user_data_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.environ.get("APPDATA", Path.home())))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "somehand"


SOURCE_ROOT = _source_root()


def _data_root() -> Path:
    override = os.environ.get("SOMEHAND_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return SOURCE_ROOT or _user_data_root()


DATA_ROOT = _data_root()
# Kept as a compatibility alias for repository-oriented callers.
PROJECT_ROOT = SOURCE_ROOT or DATA_ROOT

DEFAULT_MODELSCOPE_REPO_ID = "BingqianWu/somehand-assets"
DEFAULT_HUGGINGFACE_REPO_ID = "12e21/somehand-assets"


@dataclass(frozen=True)
class AssetEntry:
    remote_path: str
    local_path: str
    mode: str = "copy"


ASSET_GROUPS: dict[str, list[AssetEntry]] = {
    "mjcf": [
        AssetEntry(
            remote_path="archives/mjcf_assets.tar.gz",
            local_path="assets/mjcf",
            mode="extract",
        ),
    ],
    "mediapipe": [
        AssetEntry(
            remote_path="models/hand_landmarker.task",
            local_path="assets/models/hand_landmarker.task",
        ),
    ],
    "examples": [
        AssetEntry(
            remote_path="archives/reference_assets.tar.gz",
            local_path="assets",
            mode="extract",
        ),
        AssetEntry(
            remote_path="archives/sample_recordings.tar.gz",
            local_path="recordings",
            mode="extract",
        ),
    ],
}


def iter_asset_entries(groups: Iterable[str] | None = None) -> list[tuple[str, AssetEntry]]:
    selected_groups = list(groups) if groups is not None else list(ASSET_GROUPS)
    entries: list[tuple[str, AssetEntry]] = []
    for group in selected_groups:
        entries.extend((group, entry) for entry in ASSET_GROUPS[group])
    return entries


def resolve_asset_path(local_path: str | Path) -> Path:
    return DATA_ROOT / Path(local_path)


def infer_asset_group(path: str | Path) -> str | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (DATA_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for group, entries in ASSET_GROUPS.items():
        for entry in entries:
            local_root = resolve_asset_path(entry.local_path).resolve()
            if candidate == local_root:
                return group
            try:
                candidate.relative_to(local_root)
                return group
            except ValueError:
                continue
    return None


def build_download_command(*, group: str | None = None, source: str = "modelscope") -> str:
    command = ["somehand", "assets", "download"]
    if group is not None:
        command.extend(["--only", group])
    if source != "modelscope":
        command.extend(["--source", source])
    return " ".join(command)


def build_missing_asset_message(
    path: str | Path,
    *,
    group: str | None = None,
    label: str = "Asset",
) -> str:
    asset_group = group or infer_asset_group(path)
    command = build_download_command(group=asset_group)
    return f"{label} not found: {path}. Download it with `{command}`."
