"""Setuptools build customizations for bundled retargeting configs."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class BuildPyWithConfigs(_build_py):
    """Copy the repository config source of truth into built wheels."""

    def run(self) -> None:
        super().run()
        source = Path(__file__).resolve().parent / "configs" / "retargeting"
        target = Path(self.build_lib) / "somehand" / "_resources" / "retargeting"
        shutil.copytree(source, target, dirs_exist_ok=True)


setup(cmdclass={"build_py": BuildPyWithConfigs})
