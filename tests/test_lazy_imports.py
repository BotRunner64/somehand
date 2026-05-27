import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent
_SRC = _REPO_ROOT / "src"


def _assert_import_does_not_import_cv2(module_name: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    script = f"""
import builtins
import importlib

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cv2" or name.startswith("cv2."):
        raise AssertionError("{module_name} should not import cv2")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
importlib.import_module("{module_name}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
    )


def test_pico_input_import_does_not_import_cv2():
    _assert_import_does_not_import_cv2("somehand.pico_input")


def test_pico_source_adapter_import_does_not_import_cv2():
    _assert_import_does_not_import_cv2("somehand.runtime.source_adapters")
