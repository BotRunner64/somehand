"""Domain-layer models and pure transformations."""

from .config import (
    AngleConstraint,
    HandConfig,
    PinchConfig,
    PositionConfig,
    PositionConstraint,
    PreprocessConfig,
    RetargetingConfig,
    SolverConfig,
    VectorLossConfig,
)
from .hand_side import HAND_SIDES, HandSide, display_hand_side, normalize_hand_side
from .models import HandFrame, HandFrameSink, HandTrackingSource, OutputSink, PreviewWindow, RetargetingStepResult, SessionSummary, SourceFrame
from .preprocessing import compute_target_directions, preprocess_landmarks

__all__ = [
    "AngleConstraint",
    "display_hand_side",
    "HandConfig",
    "HandFrame",
    "HandFrameSink",
    "HAND_SIDES",
    "HandSide",
    "HandTrackingSource",
    "normalize_hand_side",
    "OutputSink",
    "PinchConfig",
    "PositionConfig",
    "PositionConstraint",
    "PreviewWindow",
    "PreprocessConfig",
    "RetargetingConfig",
    "RetargetingStepResult",
    "SessionSummary",
    "SolverConfig",
    "SourceFrame",
    "compute_target_directions",
    "preprocess_landmarks",
    "VectorLossConfig",
]
