"""Shared domain data structures and lightweight protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .hand_side import display_hand_side, normalize_hand_side


@dataclass(slots=True)
class HandFrame:
    """Normalized hand-tracking frame used by the application layer."""

    landmarks_3d: np.ndarray
    landmarks_2d: np.ndarray | None
    hand_side: str

    def __post_init__(self) -> None:
        self.hand_side = normalize_hand_side(self.hand_side)

    @property
    def handedness(self) -> str:
        return display_hand_side(self.hand_side)


@dataclass(slots=True)
class SourceFrame:
    """A frame read from a source, optionally containing a hand detection."""

    detection: HandFrame | None
    preview_frame: np.ndarray | None = None


@dataclass(slots=True)
class RetargetingStepResult:
    """Output of a single retargeting step."""

    qpos: np.ndarray
    target_directions: np.ndarray | None
    processed_landmarks: np.ndarray
    hand_side: str

    def __post_init__(self) -> None:
        self.hand_side = normalize_hand_side(self.hand_side)

    @property
    def handedness(self) -> str:
        return display_hand_side(self.hand_side)


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """High-level counters produced by a retargeting session."""

    num_frames: int
    num_detected: int
    source_desc: str
    input_type: str


class OutputSink(Protocol):
    @property
    def is_running(self) -> bool: ...

    def on_result(self, result: RetargetingStepResult) -> None: ...

    def close(self) -> None: ...


class HandFrameSink(Protocol):
    @property
    def is_running(self) -> bool: ...

    def on_frame(self, frame: HandFrame) -> None: ...

    def close(self) -> None: ...


class PreviewWindow(Protocol):
    def show(self, source: object, frame: SourceFrame) -> bool: ...

    def close(self) -> None: ...


class HandTrackingSource(Protocol):
    source_desc: str

    @property
    def fps(self) -> int: ...

    def is_available(self) -> bool: ...

    def get_frame(self) -> SourceFrame: ...

    def reset(self) -> bool: ...

    def close(self) -> None: ...

    def stats_snapshot(self) -> Mapping[str, object]: ...
