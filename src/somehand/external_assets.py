"""Project-local external asset manifest and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELSCOPE_REPO_ID_ENV = "SOMEHAND_MODELSCOPE_REPO_ID"
HUGGINGFACE_REPO_ID_ENV = "SOMEHAND_HF_REPO_ID"

DEFAULT_MODELSCOPE_REPO_ID = os.getenv(MODELSCOPE_REPO_ID_ENV, "")
DEFAULT_HUGGINGFACE_REPO_ID = os.getenv(HUGGINGFACE_REPO_ID_ENV, "")


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
}


def iter_asset_entries(groups: Iterable[str] | None = None) -> list[tuple[str, AssetEntry]]:
    selected_groups = list(groups) if groups is not None else list(ASSET_GROUPS)
    entries: list[tuple[str, AssetEntry]] = []
    for group in selected_groups:
        entries.extend((group, entry) for entry in ASSET_GROUPS[group])
    return entries


def resolve_asset_path(local_path: str) -> Path:
    return PROJECT_ROOT / Path(local_path)


def infer_asset_group(path: str | Path) -> str | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
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
    command = ["python", "scripts/setup/download_assets.py"]
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
    return (
        f"{label} not found: {path}. "
        f"Download it with `{command}` after setting "
        f"`{MODELSCOPE_REPO_ID_ENV}` or passing `--repo-id`."
    )
