"""Unified source adapters for MediaPipe, hc_mocap, and PICO inputs."""

from __future__ import annotations

import cv2
import numpy as np
import time
from threading import Lock

from somehand.domain import (
    BiHandFrame,
    BiHandSourceFrame,
    HandFrame,
    SourceFrame,
    normalize_hand_side,
)
from somehand.hand_detector import HandDetection, HandDetector
from somehand.hc_mocap_input import (
    _DirectHCMocapUDPProvider,
    create_hc_mocap_udp_provider,
    hc_mocap_frame_to_landmarks,
)
from somehand.pico_input import create_pico_provider

from .artifacts import load_bihand_recording_artifact, load_hand_recording_artifact

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _to_hand_frame(detection: HandDetection) -> HandFrame:
    return HandFrame(
        landmarks_3d=detection.landmarks_3d,
        landmarks_2d=detection.landmarks_2d,
        hand_side=detection.hand_side,
    )


def _annotate_preview(frame: np.ndarray, detection: HandFrame) -> np.ndarray:
    annotated = frame.copy()
    if detection.landmarks_2d is None:
        return annotated
    for x, y in detection.landmarks_2d:
        cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)
    for start_idx, end_idx in _HAND_CONNECTIONS:
        p1 = tuple(detection.landmarks_2d[start_idx].astype(int))
        p2 = tuple(detection.landmarks_2d[end_idx].astype(int))
        cv2.line(annotated, p1, p2, (0, 200, 0), 1)
    return annotated


def _copy_hand_frame(frame: HandFrame) -> HandFrame:
    return HandFrame(
        landmarks_3d=np.array(frame.landmarks_3d, copy=True),
        landmarks_2d=None if frame.landmarks_2d is None else np.array(frame.landmarks_2d, copy=True),
        hand_side=frame.hand_side,
    )


def _to_bihand_frame(*, left: HandDetection | None = None, right: HandDetection | None = None) -> BiHandFrame:
    return BiHandFrame(
        left=None if left is None else _to_hand_frame(left),
        right=None if right is None else _to_hand_frame(right),
    )


def _copy_bihand_frame(frame: BiHandFrame) -> BiHandFrame:
    return BiHandFrame(
        left=None if frame.left is None else _copy_hand_frame(frame.left),
        right=None if frame.right is None else _copy_hand_frame(frame.right),
    )


def _annotate_single_hand(frame: np.ndarray, detection: HandFrame, *, color: tuple[int, int, int]) -> np.ndarray:
    annotated = frame.copy()
    if detection.landmarks_2d is None:
        return annotated
    for x, y in detection.landmarks_2d:
        cv2.circle(annotated, (int(x), int(y)), 3, color, -1)
    for start_idx, end_idx in _HAND_CONNECTIONS:
        p1 = tuple(detection.landmarks_2d[start_idx].astype(int))
        p2 = tuple(detection.landmarks_2d[end_idx].astype(int))
        cv2.line(annotated, p1, p2, color, 1)
    return annotated


def _annotate_bihand_preview(frame: np.ndarray, detection: BiHandFrame) -> np.ndarray:
    annotated = frame.copy()
    if detection.left is not None:
        annotated = _annotate_single_hand(annotated, detection.left, color=(255, 140, 0))
        if detection.left.landmarks_2d is not None:
            wrist = detection.left.landmarks_2d[0].astype(int)
            cv2.putText(annotated, "Left", tuple(wrist), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
    if detection.right is not None:
        annotated = _annotate_single_hand(annotated, detection.right, color=(0, 220, 0))
        if detection.right.landmarks_2d is not None:
            wrist = detection.right.landmarks_2d[0].astype(int)
            cv2.putText(annotated, "Right", tuple(wrist), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return annotated


class MediaPipeInputSource:
    def __init__(
        self,
        source: int | str,
        *,
        hand_side: str | None,
        swap_handedness: bool,
        source_desc: str,
    ):
        self.source_desc = source_desc
        self.hand_side = None if hand_side is None else normalize_hand_side(hand_side)
        self._frames = HandDetector.create_source(source)
        self._detector = HandDetector(
            target_hand=self.hand_side,
            swap_handedness=swap_handedness,
        )
        self._available = True

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._available

    def get_frame(self) -> SourceFrame:
        if not self._available:
            raise StopIteration
        try:
            preview_frame = next(self._frames)
        except StopIteration as exc:
            self._available = False
            raise StopIteration from exc

        detection = self._detector.detect(preview_frame)
        return SourceFrame(
            detection=None if detection is None else _to_hand_frame(detection),
            preview_frame=preview_frame,
        )

    def annotate_preview(self, frame: np.ndarray, detection: HandFrame) -> np.ndarray:
        return _annotate_preview(frame, detection)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._detector.close()

    def stats_snapshot(self) -> dict[str, object]:
        return {}


class BiHandMediaPipeInputSource:
    def __init__(
        self,
        source: int | str,
        *,
        swap_handedness: bool,
        source_desc: str,
    ):
        self.source_desc = source_desc
        self._frames = HandDetector.create_source(source)
        self._detector = HandDetector(
            num_hands=2,
            target_hand=None,
            swap_handedness=swap_handedness,
        )
        self._available = True
        self._latest_frame: BiHandFrame | None = None
        self._frame_index = 0

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._available

    def get_frame(self) -> BiHandSourceFrame:
        if not self._available:
            raise StopIteration
        try:
            preview_frame = next(self._frames)
        except StopIteration as exc:
            self._available = False
            raise StopIteration from exc

        detections = self._detector.detect_all(preview_frame)
        left_detection = next((item for item in detections if item.hand_side == "left"), None)
        right_detection = next((item for item in detections if item.hand_side == "right"), None)
        detection = _to_bihand_frame(left=left_detection, right=right_detection)
        self._frame_index += 1
        self._latest_frame = detection if detection.has_detection else None
        return BiHandSourceFrame(
            detection=detection if detection.has_detection else None,
            preview_frame=preview_frame,
        )

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        if self._latest_frame is None:
            return None
        return self._frame_index, _copy_bihand_frame(self._latest_frame)

    def annotate_preview(self, frame: np.ndarray, detection: BiHandFrame) -> np.ndarray:
        return _annotate_bihand_preview(frame, detection)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._detector.close()

    def stats_snapshot(self) -> dict[str, object]:
        return {}


class RecordingHandTrackingSource:
    def __init__(self, wrapped_source: object, *, recording_enabled: bool = True):
        self._wrapped_source = wrapped_source
        self.recorded_frames: list[HandFrame] = []
        self._recording_lock = Lock()
        self._recording_enabled = recording_enabled

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def source_desc(self) -> str:
        return str(getattr(self._wrapped_source, "source_desc"))

    @property
    def fps(self) -> int:
        return int(getattr(self._wrapped_source, "fps"))

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> SourceFrame:
        frame = self._wrapped_source.get_frame()
        if frame.detection is not None and self.is_recording:
            self.recorded_frames.append(_copy_hand_frame(frame.detection))
        return frame

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_hand_frame_snapshot", None)
        if not callable(snapshot_fn):
            return None
        return snapshot_fn()

    def reset(self) -> bool:
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        return dict(self._wrapped_source.stats_snapshot())

    @property
    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording_enabled

    def start_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = True

    def stop_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = False


class RecordingBiHandTrackingSource:
    def __init__(self, wrapped_source: object, *, recording_enabled: bool = True):
        self._wrapped_source = wrapped_source
        self.recorded_frames: list[BiHandFrame] = []
        self._recording_lock = Lock()
        self._recording_enabled = recording_enabled

    def __getattr__(self, name: str):
        return getattr(self._wrapped_source, name)

    @property
    def source_desc(self) -> str:
        return str(getattr(self._wrapped_source, "source_desc"))

    @property
    def fps(self) -> int:
        return int(getattr(self._wrapped_source, "fps"))

    def is_available(self) -> bool:
        return bool(self._wrapped_source.is_available())

    def get_frame(self) -> BiHandSourceFrame:
        frame = self._wrapped_source.get_frame()
        if frame.detection is not None and frame.detection.has_detection and self.is_recording:
            self.recorded_frames.append(_copy_bihand_frame(frame.detection))
        return frame

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        snapshot_fn = getattr(self._wrapped_source, "latest_bihand_frame_snapshot", None)
        if not callable(snapshot_fn):
            return None
        return snapshot_fn()

    def reset(self) -> bool:
        return bool(self._wrapped_source.reset())

    def close(self) -> None:
        self._wrapped_source.close()

    def stats_snapshot(self) -> dict[str, object]:
        return dict(self._wrapped_source.stats_snapshot())

    @property
    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording_enabled

    def start_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = True

    def stop_recording(self) -> None:
        with self._recording_lock:
            self._recording_enabled = False


class HCMocapInputSource:
    def __init__(self, provider: object, *, source_desc: str):
        self.source_desc = source_desc
        self._provider = provider

    @property
    def fps(self) -> int:
        return int(getattr(self._provider, "fps", 30))

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    def get_frame(self) -> SourceFrame:
        detection = self._provider.get_detection()
        return SourceFrame(
            detection=_to_hand_frame(detection),
        )

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        snapshot_fn = getattr(self._provider, "latest_detection_snapshot", None)
        if not callable(snapshot_fn):
            return None

        snapshot = snapshot_fn()
        if snapshot is None:
            return None

        frame_index, detection = snapshot
        return frame_index, _to_hand_frame(detection)

    def reset(self) -> bool:
        reset_fn = getattr(getattr(self._provider, "_provider", None), "reset", None)
        if not callable(reset_fn):
            return False
        reset_fn()
        return True

    def close(self) -> None:
        self._provider.close()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._provider, "stats_snapshot", None)
        if callable(stats_fn):
            return dict(stats_fn())
        return {}


class RecordedHandDataSource:
    def __init__(self, recording_path: str):
        recording = load_hand_recording_artifact(recording_path)
        self._frames: list[HandFrame] = list(recording["frames"])
        self._fps = int(recording["fps"])
        self._index = 0
        self.recording_path = recording_path
        self.source_desc = recording_path
        self.recording_metadata = {
            "input_source": recording["input_source"],
            "input_type": recording["input_type"],
            "hand_side": recording.get("hand_side"),
            "num_frames": recording["num_frames"],
            "num_detected": recording["num_detected"],
        }

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> SourceFrame:
        if not self.is_available():
            raise StopIteration
        frame = self._frames[self._index]
        self._index += 1
        return SourceFrame(detection=_copy_hand_frame(frame))

    def reset(self) -> bool:
        if not self._frames:
            return False
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self) -> dict[str, object]:
        return {}


class BiHandPicoInputSource:
    def __init__(self, *, timeout: float):
        self.source_desc = "pico://both"
        self._timeout = timeout
        self._left_provider = create_pico_provider(hand_side="left", timeout=timeout)
        self._right_provider = create_pico_provider(hand_side="right", timeout=timeout)
        self._last_frame_index = 0

    @property
    def fps(self) -> int:
        return min(self._left_provider.fps, self._right_provider.fps)

    def is_available(self) -> bool:
        return self._left_provider.is_available() or self._right_provider.is_available()

    def get_frame(self) -> BiHandSourceFrame:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            left_snapshot = self._left_provider.latest_detection_snapshot()
            right_snapshot = self._right_provider.latest_detection_snapshot()
            latest_index = max(
                left_snapshot[0] if left_snapshot is not None else 0,
                right_snapshot[0] if right_snapshot is not None else 0,
            )
            if latest_index > self._last_frame_index:
                self._last_frame_index = latest_index
                detection = _to_bihand_frame(
                    left=None if left_snapshot is None else left_snapshot[1],
                    right=None if right_snapshot is None else right_snapshot[1],
                )
                return BiHandSourceFrame(detection=detection if detection.has_detection else None)
            time.sleep(0.002)

        raise TimeoutError(f"No new PICO bi-hand frame within {self._timeout}s")

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        left_snapshot = self._left_provider.latest_detection_snapshot()
        right_snapshot = self._right_provider.latest_detection_snapshot()
        latest_index = max(
            left_snapshot[0] if left_snapshot is not None else 0,
            right_snapshot[0] if right_snapshot is not None else 0,
        )
        if latest_index <= 0:
            return None
        detection = _to_bihand_frame(
            left=None if left_snapshot is None else left_snapshot[1],
            right=None if right_snapshot is None else right_snapshot[1],
        )
        return latest_index, detection

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._left_provider.close()
        self._right_provider.close()

    def stats_snapshot(self) -> dict[str, object]:
        left_stats = self._left_provider.stats_snapshot()
        right_stats = self._right_provider.stats_snapshot()
        return {
            "left": left_stats,
            "right": right_stats,
            "active_frames": int(left_stats.get("active_frames", 0)) + int(right_stats.get("active_frames", 0)),
        }


class BiHCMocapInputSource:
    def __init__(
        self,
        *,
        reference_bvh: str | None,
        host: str,
        port: int,
        timeout: float,
    ):
        self._provider = _DirectHCMocapUDPProvider(
            reference_bvh=reference_bvh,
            host=host,
            port=port,
            timeout=timeout,
        )
        self.source_desc = f"udp://{host or '0.0.0.0'}:{port}"

    @property
    def fps(self) -> int:
        return self._provider.fps

    def is_available(self) -> bool:
        return self._provider.is_available()

    def get_frame(self) -> BiHandSourceFrame:
        frame = self._provider.get_frame()
        return BiHandSourceFrame(detection=self._frame_to_detection(frame))

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        snapshot = self._provider.latest_frame_snapshot()
        if snapshot is None:
            return None
        frame_index, frame = snapshot
        return frame_index, self._frame_to_detection(frame)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._provider.close()

    def stats_snapshot(self) -> dict[str, object]:
        return dict(self._provider.stats_snapshot())

    @staticmethod
    def _frame_to_detection(frame: dict[str, tuple[np.ndarray, np.ndarray]]) -> BiHandFrame:
        left = HandDetection(
            landmarks_3d=hc_mocap_frame_to_landmarks(frame, "left"),
            landmarks_2d=np.zeros((21, 2), dtype=np.float64),
            hand_side="left",
        )
        right = HandDetection(
            landmarks_3d=hc_mocap_frame_to_landmarks(frame, "right"),
            landmarks_2d=np.zeros((21, 2), dtype=np.float64),
            hand_side="right",
        )
        return _to_bihand_frame(left=left, right=right)


class RecordedBiHandDataSource:
    def __init__(self, recording_path: str):
        recording = load_bihand_recording_artifact(recording_path)
        self._frames: list[BiHandFrame] = list(recording["frames"])
        self._fps = int(recording["fps"])
        self._index = 0
        self.recording_path = recording_path
        self.source_desc = recording_path
        self.recording_metadata = {
            "input_source": recording["input_source"],
            "input_type": recording["input_type"],
            "num_frames": recording["num_frames"],
            "num_detected": recording["num_detected"],
        }

    @property
    def fps(self) -> int:
        return self._fps

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> BiHandSourceFrame:
        if not self.is_available():
            raise StopIteration
        frame = self._frames[self._index]
        self._index += 1
        return BiHandSourceFrame(detection=_copy_bihand_frame(frame))

    def reset(self) -> bool:
        if not self._frames:
            return False
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self) -> dict[str, object]:
        return {}

def create_hc_mocap_udp_source(
    *,
    reference_bvh: str | None,
    hand_side: str,
    host: str,
    port: int,
    timeout: float,
) -> HCMocapInputSource:
    provider = create_hc_mocap_udp_provider(
        reference_bvh=reference_bvh,
        hand_side=hand_side,
        host=host,
        port=port,
        timeout=timeout,
    )
    return HCMocapInputSource(
        provider,
        source_desc=f"udp://{host or '0.0.0.0'}:{port}",
    )


def create_pico_source(*, hand_side: str, timeout: float) -> HCMocapInputSource:
    normalized_side = normalize_hand_side(hand_side)
    provider = create_pico_provider(hand_side=normalized_side, timeout=timeout)
    return HCMocapInputSource(
        provider,
        source_desc=f"pico://{normalized_side}",
    )


def create_recording_source(*, recording_path: str) -> RecordedHandDataSource:
    return RecordedHandDataSource(recording_path)


def create_bihand_pico_source(*, timeout: float) -> BiHandPicoInputSource:
    return BiHandPicoInputSource(timeout=timeout)


def create_bihand_hc_mocap_udp_source(
    *,
    reference_bvh: str | None,
    host: str,
    port: int,
    timeout: float,
) -> BiHCMocapInputSource:
    return BiHCMocapInputSource(
        reference_bvh=reference_bvh,
        host=host,
        port=port,
        timeout=timeout,
    )


def create_bihand_recording_source(*, recording_path: str) -> RecordedBiHandDataSource:
    return RecordedBiHandDataSource(recording_path)
