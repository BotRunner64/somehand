"""Live source adapters for MediaPipe, hc_mocap, and PICO."""

from __future__ import annotations

import time

import numpy as np

from somehand.core import BiHandFrame, BiHandSourceFrame, HandFrame, SourceFrame, normalize_hand_side
from somehand.hand_detector import HandDetection, HandDetector
from somehand.hc_mocap_input import _DirectHCMocapUDPProvider, create_hc_mocap_udp_provider, hc_mocap_frame_to_landmarks
from somehand.pico_input import create_pico_provider

from .source_transforms import annotate_bihand_preview, annotate_preview, copy_bihand_frame, to_bihand_frame, to_hand_frame


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
            detection=None if detection is None else to_hand_frame(detection),
            preview_frame=preview_frame,
        )

    def annotate_preview(self, frame: np.ndarray, detection: HandFrame) -> np.ndarray:
        return annotate_preview(frame, detection)

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

        detections = self._detector.detect_all(preview_frame) or []
        left_detection = next((item for item in detections if item.hand_side == "left"), None)
        right_detection = next((item for item in detections if item.hand_side == "right"), None)
        detection = to_bihand_frame(left=left_detection, right=right_detection)
        self._frame_index += 1
        self._latest_frame = detection if detection.has_detection else None
        return BiHandSourceFrame(
            detection=detection if detection.has_detection else None,
            preview_frame=preview_frame,
        )

    def latest_bihand_frame_snapshot(self) -> tuple[int, BiHandFrame] | None:
        if self._latest_frame is None:
            return None
        return self._frame_index, copy_bihand_frame(self._latest_frame)

    def annotate_preview(self, frame: np.ndarray, detection: BiHandFrame) -> np.ndarray:
        return annotate_bihand_preview(frame, detection)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self._detector.close()

    def stats_snapshot(self) -> dict[str, object]:
        return {}


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
            detection=to_hand_frame(detection),
        )

    def latest_hand_frame_snapshot(self) -> tuple[int, HandFrame] | None:
        snapshot_fn = getattr(self._provider, "latest_detection_snapshot", None)
        if not callable(snapshot_fn):
            return None

        snapshot = snapshot_fn()
        if snapshot is None:
            return None

        frame_index, detection = snapshot
        return frame_index, to_hand_frame(detection)

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
                detection = to_bihand_frame(
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
        detection = to_bihand_frame(
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
        return to_bihand_frame(left=left, right=right)


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
