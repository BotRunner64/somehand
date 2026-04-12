import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.external_assets import resolve_asset_path
from somehand.paths import DEFAULT_LINKERHAND_SDK_PATH, PROJECT_ROOT


def test_default_linkerhand_sdk_path_points_inside_repo():
    assert DEFAULT_LINKERHAND_SDK_PATH == PROJECT_ROOT / "third_party" / "linkerhand-python-sdk"


def test_resolve_asset_path_is_project_relative():
    assert resolve_asset_path("assets/models/hand_landmarker.task") == PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
