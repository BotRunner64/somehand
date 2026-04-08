"""Unified source adapters for MediaPipe, hc_mocap, and PICO inputs."""

from __future__ import annotations

import cv2
import numpy as np
from threading import Lock

from dex_mujoco.domain import HandFrame, SourceFrame
from dex_mujoco.hand_detector import HandDetection, HandDetector
from dex_mujoco.hc_mocap_input import create_hc_mocap_bvh_provider, create_hc_mocap_udp_provider
from dex_mujoco.pico_input import create_pico_provider

from .artifacts import load_hand_recording_artifact

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _to_hand_frame(detection: HandDetection, *, local_frame_override: bool = False) -> HandFrame:
    metadata: dict[str, object] = {}
    if local_frame_override and detection.landmarks_3d_local is not None:
        metadata["preprocess_frame_override"] = "camera_aligned"
    return HandFrame(
        landmarks_3d=detection.landmarks_3d,
        landmarks_2d=detection.landmarks_2d,
        handedness=detection.handedness,
        landmarks_3d_local=detection.landmarks_3d_local,
        metadata=metadata,
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
        handedness=frame.handedness,
        landmarks_3d_local=None if frame.landmarks_3d_local is None else np.array(frame.landmarks_3d_local, copy=True),
        metadata=dict(frame.metadata),
    )


class MediaPipeInputSource:
    def __init__(
        self,
        source: int | str,
        *,
        target_hand: str | None,
        swap_handedness: bool,
        source_desc: str,
    ):
        self.source_desc = source_desc
        self._frames = HandDetector.create_source(source)
        self._detector = HandDetector(
            target_hand=target_hand,
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


class HCMocapInputSource:
    def __init__(self, provider: object, *, source_desc: str, use_local_frame: bool = True):
        self.source_desc = source_desc
        self._provider = provider
        self._use_local_frame = use_local_frame

    @property
    def fps(self) -> int:
        return int(getattr(self._provider, "fps", 30))

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    def get_frame(self) -> SourceFrame:
        detection = self._provider.get_detection()
        return SourceFrame(
            detection=_to_hand_frame(detection, local_frame_override=self._use_local_frame),
        )

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        snapshot_fn = getattr(self._provider, "latest_detection_snapshot", None)
        if not callable(snapshot_fn):
            return None

        snapshot = snapshot_fn()
        if snapshot is None:
            return None

        frame_index, detection = snapshot
        return frame_index, _to_hand_frame(detection, local_frame_override=self._use_local_frame)

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
            "handedness": recording.get("handedness"),
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


def create_hc_mocap_bvh_source(
    *,
    bvh_path: str,
    handedness: str,
    teleopit_root: str | None,
) -> HCMocapInputSource:
    provider = create_hc_mocap_bvh_provider(
        bvh_path=bvh_path,
        handedness=handedness,
        teleopit_root=teleopit_root,
    )
    return HCMocapInputSource(provider, source_desc=bvh_path)


def create_hc_mocap_udp_source(
    *,
    reference_bvh: str,
    handedness: str,
    host: str,
    port: int,
    timeout: float,
) -> HCMocapInputSource:
    provider = create_hc_mocap_udp_provider(
        reference_bvh=reference_bvh,
        handedness=handedness,
        host=host,
        port=port,
        timeout=timeout,
    )
    return HCMocapInputSource(
        provider,
        source_desc=f"udp://{host or '0.0.0.0'}:{port}",
    )


def create_pico_source(*, handedness: str, timeout: float) -> HCMocapInputSource:
    provider = create_pico_provider(handedness=handedness, timeout=timeout)
    return HCMocapInputSource(
        provider,
        source_desc=f"pico://{handedness.lower()}",
    )


def create_recording_source(*, recording_path: str) -> RecordedHandDataSource:
    return RecordedHandDataSource(recording_path)
