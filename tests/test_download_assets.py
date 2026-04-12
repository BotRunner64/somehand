from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from scripts.setup.download_assets import _place_assets, _resolve_entry_source, _safe_extract_tar
from somehand.external_assets import AssetEntry, build_missing_asset_message


def test_safe_extract_tar_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "b.txt").write_text("world\n", encoding="utf-8")

    archive = tmp_path / "bundle.tar.gz"

    import tarfile

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=".")

    dst = tmp_path / "dst"
    _safe_extract_tar(archive, dst)

    assert (dst / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (dst / "nested" / "b.txt").read_text(encoding="utf-8") == "world\n"


def test_resolve_entry_source_uses_remote_layout(tmp_path: Path) -> None:
    archive = tmp_path / "archives" / "mjcf_assets.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")

    entry = AssetEntry(
        remote_path="archives/mjcf_assets.tar.gz",
        local_path="assets/mjcf",
        mode="extract",
    )

    assert _resolve_entry_source(tmp_path, entry) == archive


def test_place_assets_fails_when_requested_entry_missing(tmp_path: Path) -> None:
    entry = AssetEntry(
        remote_path="archives/mjcf_assets.tar.gz",
        local_path="assets/mjcf",
        mode="extract",
    )

    with pytest.raises(FileNotFoundError, match="missing requested asset entries"):
        _place_assets([entry], tmp_path)


def test_missing_asset_message_points_to_group_download() -> None:
    message = build_missing_asset_message(
        "assets/mjcf/linkerhand_l20_right/model.xml",
        label="MJCF file",
    )

    assert "MJCF file not found" in message
    assert "--only mjcf" in message
    assert "SOMEHAND_MODELSCOPE_REPO_ID" in message
