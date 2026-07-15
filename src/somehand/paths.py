"""Project-local default paths used by CLI entrypoints."""

import os
from pathlib import Path

from .external_assets import (
    DATA_ROOT,
    PACKAGE_ROOT,
    PROJECT_ROOT as PROJECT_ROOT,
    SOURCE_ROOT,
    resolve_asset_path,
)


def _config_root() -> Path:
    override = os.environ.get("SOMEHAND_CONFIG_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if SOURCE_ROOT is not None:
        return SOURCE_ROOT / "configs" / "retargeting"
    return PACKAGE_ROOT / "_resources" / "retargeting"


CONFIG_ROOT = _config_root()
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "right" / "linkerhand_l20_right.yaml"
DEFAULT_BIHAND_CONFIG_PATH = CONFIG_ROOT / "bihand" / "linkerhand_l20_bihand.yaml"
DEFAULT_HC_MOCAP_REFERENCE_BVH = "__builtin_hc_mocap__"
DEFAULT_HAND_LANDMARKER_MODEL = resolve_asset_path("assets/models/hand_landmarker.task")
DEFAULT_LINKERHAND_SDK_PATH = DATA_ROOT / "third_party" / "linkerhand-python-sdk"


def resolve_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate

    relative = candidate
    if candidate.parts[:2] == ("configs", "retargeting"):
        relative = Path(*candidate.parts[2:])
    bundled_candidate = CONFIG_ROOT / relative
    return bundled_candidate if bundled_candidate.exists() else candidate
