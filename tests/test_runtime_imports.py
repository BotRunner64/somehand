import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


@pytest.mark.parametrize(
    "module_name",
    [
        "somehand.runtime.sink_outputs",
        "somehand.runtime.sink_rendering",
        "somehand.runtime.manus_source",
        "somehand.runtime.source_adapters",
        "somehand.runtime.source_recording",
        "somehand.runtime.source_sampling",
        "somehand.runtime.source_transforms",
    ],
)
def test_runtime_modules_import_cleanly_in_fresh_interpreter(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                f"sys.path.insert(0, {str(SRC)!r}); "
                f"importlib.import_module({module_name!r})"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_removed_signal_filter_module_is_not_importable():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                f"sys.path.insert(0, {str(SRC)!r}); "
                "importlib.import_module('somehand.runtime.source_signal')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


def test_infrastructure_sinks_keeps_rendering_compat_exports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(SRC)!r}); "
                "from somehand.infrastructure.sinks import "
                "_create_offscreen_renderer, _fit_video_size, _transform_points; "
                "assert callable(_create_offscreen_renderer); "
                "assert callable(_fit_video_size); "
                "assert callable(_transform_points)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
