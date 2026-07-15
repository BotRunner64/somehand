import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.external_assets import DATA_ROOT, resolve_asset_path
from somehand.paths import (
    CONFIG_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LINKERHAND_SDK_PATH,
    PROJECT_ROOT,
    resolve_config_path,
)


def test_default_linkerhand_sdk_path_points_inside_repo():
    assert DEFAULT_LINKERHAND_SDK_PATH == PROJECT_ROOT / "third_party" / "linkerhand-python-sdk"


def test_resolve_asset_path_is_project_relative():
    assert resolve_asset_path("assets/models/hand_landmarker.task") == DATA_ROOT / "assets" / "models" / "hand_landmarker.task"


def test_default_config_uses_repository_source_of_truth_in_checkout():
    assert CONFIG_ROOT == PROJECT_ROOT / "configs" / "retargeting"
    assert DEFAULT_CONFIG_PATH == CONFIG_ROOT / "right" / "linkerhand_l20_right.yaml"


def test_resolve_config_path_accepts_portable_bundled_relative_name():
    assert resolve_config_path("right/omnihand_right.yaml") == CONFIG_ROOT / "right" / "omnihand_right.yaml"
    assert resolve_config_path("configs/retargeting/right/omnihand_right.yaml").name == "omnihand_right.yaml"
