"""Stable public API for embedding somehand as a retargeting library."""

from somehand.application import BiHandRetargetingEngine, RetargetingEngine
from somehand.domain import (
    BiHandFrame,
    BiHandRetargetingConfig,
    BiHandRetargetingResult,
    HandFrame,
    RetargetingConfig,
    RetargetingStepResult,
)
from somehand.infrastructure.config_loader import load_bihand_config, load_retargeting_config
from somehand.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, resolve_config_path

__all__ = [
    "BiHandFrame",
    "BiHandRetargetingConfig",
    "BiHandRetargetingEngine",
    "BiHandRetargetingResult",
    "DEFAULT_BIHAND_CONFIG_PATH",
    "DEFAULT_CONFIG_PATH",
    "HandFrame",
    "RetargetingConfig",
    "RetargetingEngine",
    "RetargetingStepResult",
    "load_bihand_config",
    "load_retargeting_config",
    "resolve_config_path",
]
