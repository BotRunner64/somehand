"""Adapter for PICO Bridge PC receiver hand tracking data."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .domain.hand_side import normalize_hand_side
from .hand_detector import HandDetection

PICO_HAND_JOINT_NAMES: list[str] = [
    "Palm", "Wrist",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]

# PICO Bridge 26-joint indices -> MediaPipe 21-landmark indices.
# Skips Palm(0), *Metacarpal joints (6, 11, 16, 21).
_PICO_BRIDGE_TO_MEDIAPIPE: list[int] = [
    1,   # MP[0]  wrist       <- PICO[1]
    2,   # MP[1]  thumb_cmc   <- PICO[2]
    3,   # MP[2]  thumb_mcp   <- PICO[3]
    4,   # MP[3]  thumb_ip    <- PICO[4]
    5,   # MP[4]  thumb_tip   <- PICO[5]
    7,   # MP[5]  index_mcp   <- PICO[7]
    8,   # MP[6]  index_pip   <- PICO[8]
    9,   # MP[7]  index_dip   <- PICO[9]
    10,  # MP[8]  index_tip   <- PICO[10]
    12,  # MP[9]  middle_mcp  <- PICO[12]
    13,  # MP[10] middle_pip  <- PICO[13]
    14,  # MP[11] middle_dip  <- PICO[14]
    15,  # MP[12] middle_tip  <- PICO[15]
    17,  # MP[13] ring_mcp    <- PICO[17]
    18,  # MP[14] ring_pip    <- PICO[18]
    19,  # MP[15] ring_dip    <- PICO[19]
    20,  # MP[16] ring_tip    <- PICO[20]
    22,  # MP[17] little_mcp  <- PICO[22]
    23,  # MP[18] little_pip  <- PICO[23]
    24,  # MP[19] little_dip  <- PICO[24]
    25,  # MP[20] little_tip  <- PICO[25]
]

# PICO Bridge hands are emitted in PICO native tracking coordinates.
# Convert positions to the right-handed coordinate system consumed by the
# existing retargeting pipeline.
_PICO_NATIVE_TO_RH = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _transform_positions(positions: np.ndarray) -> np.ndarray:
    return positions @ _PICO_NATIVE_TO_RH.T


def _load_pico_bridge_cls():
    try:
        from pico_bridge import PicoBridge
    except ImportError as exc:
        raise RuntimeError(
            "pico_bridge is not installed. Install the PICO Bridge PC receiver package, "
            "for example by reinstalling somehand with dependencies: pip install -e ."
        ) from exc
    return PicoBridge


def pico_hand_to_landmarks(hand_state: np.ndarray) -> np.ndarray:
    """Convert a PICO Bridge (26,7) hand state to MediaPipe-style (21,3) landmarks."""
    state = np.asarray(hand_state, dtype=np.float64)
    if state.shape != (26, 7):
        state = state.reshape(26, 7)
    positions = _transform_positions(state[:, :3])
    landmarks = np.empty((21, 3), dtype=np.float64)
    for mp_idx, pico_idx in enumerate(_PICO_BRIDGE_TO_MEDIAPIPE):
        landmarks[mp_idx] = positions[pico_idx]
    return landmarks


def _hand_frame_from_pico_frame(frame: Any, hand_side: str):
    side = normalize_hand_side(hand_side)
    return frame.left_hand if side == "left" else frame.right_hand


def pico_frame_to_detection(frame: Any, hand_side: str) -> HandDetection | None:
    side = normalize_hand_side(hand_side)
    hand = _hand_frame_from_pico_frame(frame, side)
    if not bool(getattr(hand, "active", False)):
        return None
    landmarks_3d = pico_hand_to_landmarks(getattr(hand, "joints"))
    landmarks_2d = np.zeros((21, 2), dtype=np.float64)
    return HandDetection(
        landmarks_3d=landmarks_3d,
        landmarks_2d=landmarks_2d,
        hand_side=side,
    )


class PicoBridgeReceiver:
    """Owns one in-process PICO Bridge PC receiver."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 63901,
        discovery: bool = True,
        advertise_ip: str | None = None,
        timeout: float = 60.0,
    ):
        self.host = host
        self.port = int(port)
        self.discovery = bool(discovery)
        self.advertise_ip = advertise_ip
        self.timeout = float(timeout)
        self._closed = False
        bridge_cls = _load_pico_bridge_cls()
        self._bridge = bridge_cls(
            host=self.host,
            port=self.port,
            discovery=self.discovery,
            advertise_ip=self.advertise_ip,
            video=None,
            start_timeout=self.timeout,
        )
        self._bridge.start()

    @property
    def fps(self) -> int:
        stats = self.stats_snapshot()
        fps = float(stats.get("fps", 0.0) or 0.0)
        return int(round(fps)) if fps > 0 else 80

    def is_available(self) -> bool:
        return not self._closed

    def wait_frame(self, *, timeout: float | None = None, after_seq: int | None = None):
        return self._bridge.wait_frame(timeout=self.timeout if timeout is None else timeout, after_seq=after_seq)

    def latest_frame(self):
        return self._bridge.latest_frame()

    def close(self) -> None:
        self._closed = True
        self._bridge.close()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._bridge, "stats", None)
        if not callable(stats_fn):
            return {}
        stats = stats_fn()
        if hasattr(stats, "__dataclass_fields__"):
            return {name: getattr(stats, name) for name in stats.__dataclass_fields__}
        if hasattr(stats, "_asdict"):
            return dict(stats._asdict())
        return dict(stats) if isinstance(stats, dict) else {}


class PicoHandProvider:
    """Expose one hand from PICO Bridge as a hand landmark detector."""

    def __init__(
        self,
        hand_side: str,
        timeout: float = 60.0,
        *,
        host: str = "0.0.0.0",
        port: int = 63901,
        discovery: bool = True,
        advertise_ip: str | None = None,
    ):
        self.hand_side = normalize_hand_side(hand_side)
        self._timeout = float(timeout)
        self._receiver = PicoBridgeReceiver(
            host=host,
            port=port,
            discovery=discovery,
            advertise_ip=advertise_ip,
            timeout=timeout,
        )
        self._last_served_seq = 0

    @property
    def fps(self) -> int:
        return self._receiver.fps

    def is_available(self) -> bool:
        return self._receiver.is_available()

    def get_detection(self) -> HandDetection:
        deadline = time.monotonic() + self._timeout
        after_seq = self._last_served_seq if self._last_served_seq > 0 else None
        last_timeout: TimeoutError | None = None
        while time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.0)
            try:
                frame = self._receiver.wait_frame(timeout=remaining, after_seq=after_seq)
            except TimeoutError as exc:
                last_timeout = exc
                break
            self._last_served_seq = int(getattr(frame, "seq", self._last_served_seq + 1))
            detection = pico_frame_to_detection(frame, self.hand_side)
            if detection is not None:
                return detection
            after_seq = self._last_served_seq
        raise TimeoutError(
            f"No active PICO Bridge {self.hand_side} hand frame within {self._timeout}s"
        ) from last_timeout

    def latest_detection_snapshot(self) -> tuple[int, HandDetection] | None:
        frame = self._receiver.latest_frame()
        if frame is None:
            return None
        detection = pico_frame_to_detection(frame, self.hand_side)
        if detection is None:
            return None
        return int(getattr(frame, "seq", 0)), detection

    def close(self) -> None:
        self._receiver.close()

    def stats_snapshot(self) -> dict[str, object]:
        return self._receiver.stats_snapshot()


def create_pico_provider(
    hand_side: str,
    timeout: float = 60.0,
    *,
    host: str = "0.0.0.0",
    port: int = 63901,
    discovery: bool = True,
    advertise_ip: str | None = None,
) -> PicoHandProvider:
    return PicoHandProvider(
        hand_side=hand_side,
        timeout=timeout,
        host=host,
        port=port,
        discovery=discovery,
        advertise_ip=advertise_ip,
    )
