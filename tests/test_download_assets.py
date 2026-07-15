from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from somehand.asset_download import (
    _place_assets,
    _resolve_entry_source,
    _safe_extract_tar,
    build_download_parser,
)
from somehand.external_assets import AssetEntry, build_missing_asset_message


def test_safe_extract_tar_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "b.txt").write_text("world\n", encoding="utf-8")

    archive = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=".")

    dst = tmp_path / "dst"
    _safe_extract_tar(archive, dst)

    assert (dst / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (dst / "nested" / "b.txt").read_text(encoding="utf-8") == "world\n"


def test_safe_extract_tar_strips_single_top_level_directory(tmp_path: Path) -> None:
    src = tmp_path / "src"
    wrapped = src / "mjcf"
    wrapped.mkdir(parents=True)
    (wrapped / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (wrapped / "meshes").mkdir()
    (wrapped / "meshes" / "part.stl").write_text("mesh\n", encoding="utf-8")

    archive = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(wrapped, arcname="mjcf")

    dst = tmp_path / "assets" / "mjcf"
    _safe_extract_tar(archive, dst)

    assert (dst / "model.xml").read_text(encoding="utf-8") == "<mujoco/>\n"
    assert (dst / "meshes" / "part.stl").read_text(encoding="utf-8") == "mesh\n"
    assert not (dst / "mjcf").exists()


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
        _place_assets([entry], tmp_path, data_root=tmp_path / "data-root")


def test_place_assets_uses_explicit_data_root(tmp_path: Path) -> None:
    repo_cache = tmp_path / "cache"
    source = repo_cache / "models" / "hand_landmarker.task"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model")
    entry = AssetEntry(
        remote_path="models/hand_landmarker.task",
        local_path="assets/models/hand_landmarker.task",
    )

    data_root = tmp_path / "somehand-home"
    _place_assets([entry], repo_cache, data_root=data_root)

    assert (data_root / "assets" / "models" / "hand_landmarker.task").read_bytes() == b"model"


def test_download_parser_accepts_wheel_safe_data_root(tmp_path: Path) -> None:
    args = build_download_parser().parse_args(
        ["--only", "mjcf", "mediapipe", "--data-root", str(tmp_path)]
    )

    assert args.only == ["mjcf", "mediapipe"]
    assert args.data_root == str(tmp_path)


def test_missing_asset_message_points_to_group_download() -> None:
    message = build_missing_asset_message(
        "assets/mjcf/linkerhand_l20_right/model.xml",
        label="MJCF file",
    )

    assert "MJCF file not found" in message
    assert "--only mjcf" in message
    assert "Download it with `somehand assets download --only mjcf`." in message
