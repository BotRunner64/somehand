"""Adapter for PICO VR hand tracking data via xrobotoolkit_sdk."""

from __future__ import annotations

import threading
import time

import numpy as np

from .domain.hand_side import normalize_hand_side
from .hand_detector import HandDetection

PICO_HAND_JOINT_NAMES: list[str] = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]

# PICO 26-joint indices -> MediaPipe 21-landmark indices.
# Skips Palm(1), *Metacarpal joints (6,11,16,21).
_PICO_TO_MEDIAPIPE: list[int] = [
    0,   # MP[0]  wrist       <- PICO[0]
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

# Unity -> right-hand coordinate system (same as xrobot_utils).
_UNITY_TO_RH = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _transform_positions(positions: np.ndarray) -> np.ndarray:
    """Transform (N,3) positions from Unity to right-hand coordinate system."""
    return positions @ _UNITY_TO_RH.T


def pico_hand_to_landmarks(hand_state: np.ndarray) -> np.ndarray:
    """Convert a PICO (26,7) hand state to MediaPipe-style (21,3) landmarks.

    Positions are transformed from Unity to right-hand coordinate system.
    """
    positions_unity = hand_state[:, :3]  # (26, 3)
    positions = _transform_positions(positions_unity)
    landmarks = np.empty((21, 3), dtype=np.float64)
    for mp_idx, pico_idx in enumerate(_PICO_TO_MEDIAPIPE):
        landmarks[mp_idx] = positions[pico_idx]
    return landmarks


class PicoHandProvider:
    """Adapter that exposes PICO VR hand tracking as a hand landmark detector."""

    def __init__(self, hand_side: str, timeout: float = 60.0):
        self.hand_side = normalize_hand_side(hand_side)
        self._timeout = timeout

        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise RuntimeError(
                "xrobotoolkit_sdk is not installed. "
                "Install it to use PICO hand tracking input."
            ) from exc

        self._xrt = xrt
        self._xrt.init()

        self._running = True
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._ready = threading.Event()
        self._latest_state: np.ndarray | None = None
        self._frame_index = 0
        self._last_served = 0
        self._stats = {
            "polls": 0,
            "active_frames": 0,
            "inactive_frames": 0,
        }

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    @property
    def fps(self) -> int:
        return 120

    def is_available(self) -> bool:
        return self._running and self._thread.is_alive()

    def get_detection(self) -> HandDetection:
        if not self._ready.wait(timeout=self._timeout):
            stats = self.stats_snapshot()
            raise TimeoutError(
                f"No PICO hand tracking data received within {self._timeout}s. "
                f"SDK polls={stats.get('polls', 0)}, active_frames={stats.get('active_frames', 0)}, "
                f"inactive_frames={stats.get('inactive_frames', 0)}. "
                "This usually means the SDK connected, but the headset is not outputting gesture data yet "
                "(for example: XRoboToolkit client not connected/foreground, hand tracking permission off, "
                "or the headset is still in controller mode instead of gesture mode)."
            )
        deadline = time.monotonic() + self._timeout
        with self._cond:
            while self._running and self._frame_index <= self._last_served:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No new PICO hand frame within {self._timeout}s"
                    )
                self._cond.wait(timeout=remaining)

            assert self._latest_state is not None
            self._last_served = self._frame_index
            state = self._latest_state.copy()

        return self._state_to_detection(state)

    def latest_detection_snapshot(self) -> tuple[int, HandDetection] | None:
        with self._lock:
            if self._latest_state is None or self._frame_index <= 0:
                return None
            frame_index = self._frame_index
            state = self._latest_state.copy()
        return frame_index, self._state_to_detection(state)

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        try:
            self._xrt.close()
        except BaseException:
            pass

    def stats_snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._stats)

    def _state_to_detection(self, state: np.ndarray) -> HandDetection:
        landmarks_3d = pico_hand_to_landmarks(state)
        landmarks_2d = np.zeros((21, 2), dtype=np.float64)
        return HandDetection(
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            hand_side=self.hand_side,
        )

    def _poll_loop(self) -> None:
        xrt = self._xrt
        get_state = (
            xrt.get_left_hand_tracking_state
            if self.hand_side == "left"
            else xrt.get_right_hand_tracking_state
        )
        get_active = (
            xrt.get_left_hand_is_active
            if self.hand_side == "left"
            else xrt.get_right_hand_is_active
        )

        while self._running:
            try:
                is_active = get_active()
                with self._lock:
                    self._stats["polls"] += 1
                if not is_active:
                    with self._lock:
                        self._stats["inactive_frames"] += 1
                    time.sleep(0.002)
                    continue

                raw = get_state()
                state = np.asarray(raw, dtype=np.float64).reshape(26, 7)

                with self._cond:
                    self._latest_state = state
                    self._frame_index += 1
                    self._stats["active_frames"] += 1
                    self._cond.notify_all()
                self._ready.set()
                time.sleep(1.0 / 120)
            except Exception:
                if not self._running:
                    break
                time.sleep(0.01)


def create_pico_provider(
    hand_side: str,
    timeout: float = 60.0,
) -> PicoHandProvider:
    """Factory: create a PicoHandProvider."""
    return PicoHandProvider(hand_side=hand_side, timeout=timeout)
