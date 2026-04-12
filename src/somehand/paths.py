"""Project-local default paths used by CLI entrypoints."""

from .external_assets import PROJECT_ROOT, resolve_asset_path

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "retargeting" / "right" / "linkerhand_l20_right.yaml"
DEFAULT_BIHAND_CONFIG_PATH = PROJECT_ROOT / "configs" / "retargeting" / "bihand" / "linkerhand_l20_bihand.yaml"
DEFAULT_HC_MOCAP_REFERENCE_BVH = "__builtin_hc_mocap__"
DEFAULT_HAND_LANDMARKER_MODEL = resolve_asset_path("assets/models/hand_landmarker.task")
DEFAULT_LINKERHAND_SDK_PATH = PROJECT_ROOT / "third_party" / "linkerhand-python-sdk"
