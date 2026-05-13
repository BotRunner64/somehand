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

__all__ = [
    "BiHandFrame",
    "BiHandRetargetingConfig",
    "BiHandRetargetingEngine",
    "BiHandRetargetingResult",
    "HandFrame",
    "RetargetingConfig",
    "RetargetingEngine",
    "RetargetingStepResult",
    "load_bihand_config",
    "load_retargeting_config",
]
