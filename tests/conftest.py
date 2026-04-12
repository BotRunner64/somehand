from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJCF_SENTINEL = PROJECT_ROOT / "assets" / "mjcf" / "linkerhand_l20_right" / "model.xml"
MJCF_REQUIRED_MODULES = {
    "test_acceptance.py",
    "test_bihand.py",
    "test_config_model.py",
    "test_controller.py",
}


def pytest_collection_modifyitems(config, items) -> None:
    if MJCF_SENTINEL.exists():
        return

    reason = "somehand MJCF assets not downloaded; run `python scripts/setup/download_assets.py --only mjcf`"
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.path.name in MJCF_REQUIRED_MODULES:
            item.add_marker(marker)
