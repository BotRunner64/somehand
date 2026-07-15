"""Download and place external somehand runtime assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from somehand.external_assets import (
    ASSET_GROUPS,
    DATA_ROOT,
    DEFAULT_HUGGINGFACE_REPO_ID,
    DEFAULT_MODELSCOPE_REPO_ID,
    AssetEntry,
    iter_asset_entries,
)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _safe_extract_tar(archive_path: Path, dst: Path) -> None:
    tmp_dst = dst.parent / f".{dst.name}.extracting"
    _remove_path(tmp_dst)
    tmp_dst.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tar.extractall(tmp_dst)

    extracted_children = list(tmp_dst.iterdir())
    extracted_root = tmp_dst
    if (
        len(extracted_children) == 1
        and extracted_children[0].is_dir()
        and not extracted_children[0].is_symlink()
    ):
        extracted_root = extracted_children[0]

    if dst.is_dir() and extracted_root.is_dir():
        for child in extracted_root.iterdir():
            target = dst / child.name
            _remove_path(target)
            child.replace(target)
        _remove_path(tmp_dst)
    else:
        _remove_path(dst)
        extracted_root.replace(dst)
        if extracted_root != tmp_dst:
            _remove_path(tmp_dst)


def _resolve_entry_source(repo_cache: Path, entry: AssetEntry) -> Path:
    return repo_cache / entry.remote_path


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        _remove_path(dst)
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)


def _place_assets(entries: list[AssetEntry], repo_cache: Path, *, data_root: Path = DATA_ROOT) -> None:
    print("\nPlacing files...")
    missing_entries: list[str] = []
    for entry in entries:
        src = _resolve_entry_source(repo_cache, entry)
        if not src.exists():
            missing_entries.append(entry.remote_path)
            continue

        dst = data_root / entry.local_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if entry.mode == "extract" and src.is_file():
            _safe_extract_tar(src, dst)
            print(f"  {entry.remote_path} -> {dst} (extracted)")
        else:
            _copy_path(src, dst)
            print(f"  {entry.remote_path} -> {dst}")

    if missing_entries:
        missing_list = ", ".join(missing_entries)
        raise FileNotFoundError(f"Downloaded repo is missing requested asset entries: {missing_list}")

    print(f"\nDone! Assets are available under {data_root}")


def _download_modelscope(
    repo_id: str,
    entries: list[AssetEntry],
    cache_dir: Path,
    *,
    data_root: Path,
) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("modelscope not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        from modelscope import snapshot_download

    allow_patterns = [f"{entry.remote_path}*" for entry in entries]
    repo_cache = cache_dir / "model" / repo_id.split("/")[-1]

    print(f"\nDownloading {repo_id} from ModelScope to {repo_cache} ...")
    print(f"Fetching: {[entry.remote_path for entry in entries]}")
    snapshot_download(
        repo_id,
        repo_type="model",
        local_dir=str(repo_cache),
        allow_patterns=allow_patterns,
        allow_file_pattern=allow_patterns,
    )
    _place_assets(entries, repo_cache, data_root=data_root)


def _download_huggingface(
    repo_id: str,
    entries: list[AssetEntry],
    cache_dir: Path,
    *,
    data_root: Path,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    allow_patterns = [f"{entry.remote_path}*" for entry in entries]
    repo_cache = cache_dir / "model" / repo_id.split("/")[-1]

    print(f"\nDownloading {repo_id} from HuggingFace to {repo_cache} ...")
    print(f"Fetching: {[entry.remote_path for entry in entries]}")
    snapshot_download(
        repo_id,
        repo_type="model",
        local_dir=str(repo_cache),
        allow_patterns=allow_patterns,
    )
    _place_assets(entries, repo_cache, data_root=data_root)


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        choices=list(ASSET_GROUPS.keys()),
        nargs="+",
        help="Only download specific asset groups (default: all)",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download source backend (default: modelscope)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Remote asset repo id override (default: built-in somehand asset repo)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local cache directory for downloads",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Asset destination root; defaults to SOMEHAND_HOME or the platform user-data directory",
    )


def download_from_args(args: argparse.Namespace) -> None:
    groups = args.only or list(ASSET_GROUPS.keys())
    entries = [entry for _, entry in iter_asset_entries(groups)]
    data_root = Path(args.data_root).expanduser() if args.data_root else DATA_ROOT

    if args.source == "huggingface":
        repo_id = args.repo_id or DEFAULT_HUGGINGFACE_REPO_ID
        cache_dir = (
            Path(args.cache_dir).expanduser()
            if args.cache_dir
            else data_root / "data" / "huggingface_cache"
        )
        _download_huggingface(repo_id, entries, cache_dir, data_root=data_root)
        return

    repo_id = args.repo_id or DEFAULT_MODELSCOPE_REPO_ID
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else data_root / "data" / "modelscope_cache"
    )
    _download_modelscope(repo_id, entries, cache_dir, data_root=data_root)


def build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download somehand assets")
    add_download_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_download_parser().parse_args(argv)
    download_from_args(args)


__all__ = ["add_download_arguments", "build_download_parser", "download_from_args", "main"]


if __name__ == "__main__":
    main()
