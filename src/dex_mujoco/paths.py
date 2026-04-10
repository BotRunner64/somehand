"""Project-local default paths used by CLI entrypoints."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "retargeting" / "right" / "linkerhand_l20_right.yaml"
DEFAULT_BIHAND_CONFIG_PATH = PROJECT_ROOT / "configs" / "retargeting" / "bihand" / "linkerhand_l20_bihand.yaml"
DEFAULT_HC_MOCAP_REFERENCE_BVH = PROJECT_ROOT / "assets" / "ref_with_toe.bvh"
DEFAULT_HAND_LANDMARKER_MODEL = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
DEFAULT_LINKERHAND_SDK_PATH = PROJECT_ROOT.parent / "linkerhand-python-sdk"
