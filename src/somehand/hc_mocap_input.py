"""Adapters for hc_mocap hand data."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Final

import numpy as np
from scipy.spatial.transform import Rotation as R

from .domain.hand_side import normalize_hand_side
from .hand_detector import HandDetection
from .paths import DEFAULT_HC_MOCAP_REFERENCE_BVH
_OUTPUT_ROTATION_MATRIX = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_HEIGHT_OFFSET = np.array([0.0, 0.0, 0.9526], dtype=np.float64)

def _point(frame: dict[str, tuple[np.ndarray, np.ndarray]], joint_name: str) -> np.ndarray:
    if joint_name not in frame:
        raise KeyError(f"hc_mocap frame is missing joint '{joint_name}'")
    return np.asarray(frame[joint_name][0], dtype=np.float64)


def hc_mocap_frame_to_landmarks(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    hand_side: str,
) -> np.ndarray:
    """Convert a Teleopit hc_mocap frame into MediaPipe-style 21 landmarks.

    hc_mocap exposes wrist + three joints per finger. We do not extrapolate
    fingertip positions; the MediaPipe tip slots reuse the last available
    measured joint.
    """
    side = "L" if normalize_hand_side(hand_side) == "left" else "R"

    names = [
        f"hc_Hand_{side}",
        f"hc_Thumb1_{side}",
        f"hc_Thumb2_{side}",
        f"hc_Thumb3_{side}",
        None,
        f"hc_Index1_{side}",
        f"hc_Index2_{side}",
        f"hc_Index3_{side}",
        None,
        f"hc_Middle1_{side}",
        f"hc_Middle2_{side}",
        f"hc_Middle3_{side}",
        None,
        f"hc_Ring1_{side}",
        f"hc_Ring2_{side}",
        f"hc_Ring3_{side}",
        None,
        f"hc_Pinky1_{side}",
        f"hc_Pinky2_{side}",
        f"hc_Pinky3_{side}",
        None,
    ]

    landmarks = np.empty((21, 3), dtype=np.float64)
    for i, joint_name in enumerate(names):
        if joint_name is not None:
            landmarks[i] = _point(frame, joint_name)

    # Use End Site positions for fingertips (computed from last joint rotation + offset)
    tip_mapping = [
        (4, f"hc_Thumb3_{side}_EndSite", 3),
        (8, f"hc_Index3_{side}_EndSite", 7),
        (12, f"hc_Middle3_{side}_EndSite", 11),
        (16, f"hc_Ring3_{side}_EndSite", 15),
        (20, f"hc_Pinky3_{side}_EndSite", 19),
    ]
    for tip_idx, end_site_key, fallback_idx in tip_mapping:
        if end_site_key in frame:
            landmarks[tip_idx] = _point(frame, end_site_key)
        else:
            landmarks[tip_idx] = landmarks[fallback_idx]
    return landmarks


class HCMocapHandProvider:
    """Adapter that exposes hc_mocap hand data like a hand landmark detector."""

    def __init__(self, provider: object, hand_side: str):
        self._provider = provider
        self.hand_side = normalize_hand_side(hand_side)
        self._latest_detection: HandDetection | None = None
        self._latest_frame_index = 0
        self._snapshot_lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    @property
    def fps(self) -> int:
        return int(getattr(self._provider, "fps", 30))

    def get_detection(self) -> HandDetection:
        frame = self._provider.get_frame()
        detection = self._frame_to_detection(frame)
        with self._snapshot_lock:
            self._latest_frame_index += 1
            self._latest_detection = detection
        return detection

    def latest_detection_snapshot(self) -> tuple[int, HandDetection] | None:
        snapshot_fn = getattr(self._provider, "latest_frame_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if snapshot is None:
                return None
            frame_index, frame = snapshot
            return frame_index, self._frame_to_detection(frame)

        with self._snapshot_lock:
            if self._latest_detection is None or self._latest_frame_index <= 0:
                return None
            return self._latest_frame_index, self._latest_detection

    def close(self) -> None:
        close_fn = getattr(self._provider, "close", None)
        if callable(close_fn):
            close_fn()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._provider, "stats_snapshot", None)
        if callable(stats_fn):
            return dict(stats_fn())
        return {}

    def _frame_to_detection(
        self,
        frame: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> HandDetection:
        landmarks_3d = hc_mocap_frame_to_landmarks(frame, self.hand_side)
        landmarks_2d = np.zeros((21, 2), dtype=np.float64)
        return HandDetection(
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            hand_side=self.hand_side,
        )


class _BvhSkeleton:
    def __init__(
        self,
        joint_names: list[str],
        parents: np.ndarray,
        offsets: np.ndarray,
        channels: list[list[str]],
        frame_time: float,
        end_sites: dict[int, np.ndarray] | None = None,
    ):
        self.joint_names = joint_names
        self.parents = parents
        self.offsets = offsets
        self.channels = channels
        self.frame_time = frame_time
        self.expected_floats = int(sum(len(ch) for ch in channels))
        self.end_sites = end_sites or {}  # parent_joint_idx -> local offset


_ROOT_CHANNELS: Final[tuple[str, ...]] = (
    "Xposition",
    "Yposition",
    "Zposition",
    "Zrotation",
    "Xrotation",
    "Yrotation",
)
_JOINT_CHANNELS: Final[tuple[str, ...]] = ("Zrotation", "Xrotation", "Yrotation")
_BUILTIN_HC_MOCAP_JOINTS: Final[list[tuple[str, str | None, tuple[float, float, float]]]] = [
    ("hc_Abdomen", None, (0.0, 0.0, 0.0)),
    ("hc_Hip_L", "hc_Abdomen", (-0.090058, -0.014508, -0.005261)),
    ("hc_Knee_L", "hc_Hip_L", (0.0, -0.42572, 0.0)),
    ("hc_Foot_L", "hc_Knee_L", (0.0, -0.401965, 0.0)),
    ("LeftToeBase", "hc_Foot_L", (0.008324, -0.164266, -0.10654)),
    ("hc_Hip_R", "hc_Abdomen", (0.090058, -0.014508, -0.005261)),
    ("hc_Knee_R", "hc_Hip_R", (0.0, -0.42572, 0.0)),
    ("hc_Foot_R", "hc_Knee_R", (0.0, -0.401965, 0.0)),
    ("RightToeBase", "hc_Foot_R", (-0.008324, -0.164266, -0.10654)),
    ("Spine", "hc_Abdomen", (0.0, 0.108057, -0.00891)),
    ("hc_Chest", "Spine", (0.0, 0.005886, 0.192453)),
    ("hc_Chest1", "hc_Chest", (0.0, 0.132667, 0.019818)),
    ("LeftShoulder", "hc_Chest1", (-0.03782, 0.121714, -0.007377)),
    ("hc_Shoulder_L", "LeftShoulder", (0.0, 0.0, 0.161531)),
    ("hc_Elbow_L", "hc_Shoulder_L", (0.0, -0.298399, 0.0)),
    ("hc_Hand_L", "hc_Elbow_L", (0.0, -0.269751, 0.0)),
    ("hc_Index1_L", "hc_Hand_L", (-0.027655, -0.120344, -0.008508)),
    ("hc_Index2_L", "hc_Index1_L", (0.0, 0.0, 0.042875)),
    ("hc_Index3_L", "hc_Index2_L", (0.0, 0.0, 0.033938)),
    ("hc_Middle1_L", "hc_Hand_L", (-0.001078, -0.123213, -0.003106)),
    ("hc_Middle2_L", "hc_Middle1_L", (0.0, 0.0, 0.046404)),
    ("hc_Middle3_L", "hc_Middle2_L", (0.0, 0.0, 0.036488)),
    ("hc_Pinky1_L", "hc_Hand_L", (0.040938, -0.105318, -0.013548)),
    ("hc_Pinky2_L", "hc_Pinky1_L", (0.0, 0.0, 0.03571)),
    ("hc_Pinky3_L", "hc_Pinky2_L", (0.000881, 0.0, 0.029796)),
    ("hc_Ring1_L", "hc_Hand_L", (0.022144, -0.117419, -0.007786)),
    ("hc_Ring2_L", "hc_Ring1_L", (0.0, 0.0, 0.044193)),
    ("hc_Ring3_L", "hc_Ring2_L", (0.000638, 0.0, 0.034791)),
    ("hc_Thumb1_L", "hc_Hand_L", (-0.027649, -0.047882, -0.020461)),
    ("hc_Thumb2_L", "hc_Thumb1_L", (0.0, -0.038697, 0.0)),
    ("hc_Thumb3_L", "hc_Thumb2_L", (0.0, -0.040622, 0.0)),
    ("neck", "hc_Chest1", (0.0, 0.163912, 0.023766)),
    ("hc_Head", "neck", (0.0, -0.018972, 0.09095)),
    ("RightShoulder1", "hc_Chest1", (0.03782, 0.121711, -0.007377)),
    ("hc_Shoulder_R", "RightShoulder1", (0.0, 0.0, 0.16153)),
    ("hc_Elbow_R", "hc_Shoulder_R", (0.0, -0.298397, 0.0)),
    ("hc_Hand_R", "hc_Elbow_R", (0.0, -0.269752, 0.0)),
    ("hc_Index1_R", "hc_Hand_R", (0.023345, -0.121506, -0.003388)),
    ("hc_Index2_R", "hc_Index1_R", (0.0, 0.0, 0.042879)),
    ("hc_Index3_R", "hc_Index2_R", (0.0, 0.0, 0.033935)),
    ("hc_Middle1_R", "hc_Hand_R", (-0.002997, -0.123175, 0.003474)),
    ("hc_Middle2_R", "hc_Middle1_R", (0.0, 0.0, 0.046401)),
    ("hc_Middle3_R", "hc_Middle2_R", (0.0, 0.0, 0.036488)),
    ("hc_Pinky1_R", "hc_Hand_R", (-0.044907, -0.104415, -0.005811)),
    ("hc_Pinky2_R", "hc_Pinky1_R", (0.0, 0.0, 0.035707)),
    ("hc_Pinky3_R", "hc_Pinky2_R", (0.0, 0.0, 0.029808)),
    ("hc_Ring1_R", "hc_Hand_R", (-0.026234, -0.116837, -0.000348)),
    ("hc_Ring2_R", "hc_Ring1_R", (0.0, 0.0, 0.044191)),
    ("hc_Ring3_R", "hc_Ring2_R", (0.0, 0.0, 0.034797)),
    ("hc_Thumb1_R", "hc_Hand_R", (0.025022, -0.04981, -0.019209)),
    ("hc_Thumb2_R", "hc_Thumb1_R", (0.0, -0.038698, 0.0)),
    ("hc_Thumb3_R", "hc_Thumb2_R", (0.0, -0.04062, 0.0)),
]
_BUILTIN_HC_MOCAP_END_SITES: Final[dict[str, tuple[float, float, float]]] = {
    "LeftToeBase": (0.008324, -0.164266, -0.10654),
    "RightToeBase": (-0.008324, -0.164266, -0.10654),
    "hc_Index3_L": (0.0, 0.0, 0.033938),
    "hc_Middle3_L": (0.0, 0.0, 0.036488),
    "hc_Pinky3_L": (0.000881, 0.0, 0.029796),
    "hc_Ring3_L": (0.000638, 0.0, 0.034791),
    "hc_Thumb3_L": (0.0, -0.040622, 0.0),
    "hc_Head": (0.0, -0.018972, 0.09095),
    "hc_Index3_R": (0.0, 0.0, 0.033935),
    "hc_Middle3_R": (0.0, 0.0, 0.036488),
    "hc_Pinky3_R": (0.0, 0.0, 0.029808),
    "hc_Ring3_R": (0.0, 0.0, 0.034797),
    "hc_Thumb3_R": (0.0, -0.04062, 0.0),
}


def _builtin_hc_mocap_skeleton() -> _BvhSkeleton:
    name_to_index: dict[str, int] = {}
    joint_names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}

    for index, (name, parent_name, offset) in enumerate(_BUILTIN_HC_MOCAP_JOINTS):
        joint_names.append(name)
        name_to_index[name] = index
        parents.append(-1 if parent_name is None else name_to_index[parent_name])
        offsets.append(np.array(offset, dtype=np.float64))
        channels.append(list(_ROOT_CHANNELS if parent_name is None else _JOINT_CHANNELS))

    for joint_name, offset in _BUILTIN_HC_MOCAP_END_SITES.items():
        end_sites[name_to_index[joint_name]] = np.array(offset, dtype=np.float64)

    return _BvhSkeleton(
        joint_names=joint_names,
        parents=np.array(parents, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.float64),
        channels=channels,
        frame_time=1.0 / 60.0,
        end_sites=end_sites,
    )


def _parse_bvh_reference(reference_bvh: str) -> _BvhSkeleton:
    if reference_bvh == DEFAULT_HC_MOCAP_REFERENCE_BVH:
        return _builtin_hc_mocap_skeleton()

    reference_path = Path(reference_bvh).expanduser()
    if not reference_path.exists():
        legacy_default_path = Path("assets/ref_with_toe.bvh")
        if reference_path == legacy_default_path or reference_path == legacy_default_path.resolve():
            return _builtin_hc_mocap_skeleton()
        raise FileNotFoundError(f"Reference BVH not found: {reference_bvh}")
    lines = reference_path.read_text().splitlines()
    motion_idx = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")

    joint_names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}

    def parse_node(i: int, parent_idx: int) -> int:
        header = lines[i].strip().split()
        joint_type, joint_name = header[0], header[1]
        assert joint_type in {"ROOT", "JOINT"}

        joint_idx = len(joint_names)
        joint_names.append(joint_name)
        parents.append(parent_idx)
        offsets.append(np.zeros(3, dtype=np.float64))
        channels.append([])

        i += 1
        assert lines[i].strip() == "{"
        i += 1
        while True:
            stripped = lines[i].strip()
            if stripped.startswith("OFFSET"):
                offsets[joint_idx] = np.fromstring(stripped.split("OFFSET", 1)[1], sep=" ")
                i += 1
            elif stripped.startswith("CHANNELS"):
                parts = stripped.split()
                count = int(parts[1])
                channels[joint_idx] = parts[2: 2 + count]
                i += 1
            elif stripped.startswith("JOINT"):
                i = parse_node(i, joint_idx)
            elif stripped.startswith("End Site"):
                i += 1
                assert lines[i].strip() == "{"
                i += 1
                end_offset = np.zeros(3, dtype=np.float64)
                while lines[i].strip() != "}":
                    es_line = lines[i].strip()
                    if es_line.startswith("OFFSET"):
                        end_offset = np.fromstring(
                            es_line.split("OFFSET", 1)[1], sep=" "
                        )
                    i += 1
                end_sites[joint_idx] = end_offset
                i += 1
            elif stripped == "}":
                return i + 1
            else:
                i += 1

    parse_idx = 0
    while parse_idx < motion_idx:
        stripped = lines[parse_idx].strip()
        if stripped.startswith("ROOT"):
            parse_idx = parse_node(parse_idx, -1)
        else:
            parse_idx += 1

    frame_time = 1.0 / 30.0
    for line in lines[motion_idx:]:
        stripped = line.strip()
        if stripped.startswith("Frame Time:"):
            frame_time = float(stripped.split(":", 1)[1].strip())
            break

    return _BvhSkeleton(
        joint_names=joint_names,
        parents=np.array(parents, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.float64),
        channels=channels,
        frame_time=frame_time,
        end_sites=end_sites,
    )


def _rotation_from_channels(channel_names: list[str], channel_values: list[float]) -> R:
    axes = [name[0].lower() for name in channel_names if name.endswith("rotation")]
    values = [value for name, value in zip(channel_names, channel_values) if name.endswith("rotation")]
    if not axes:
        return R.identity()
    return R.from_euler("".join(axes).upper(), values, degrees=True)


def _frame_from_bvh_values(
    skeleton: _BvhSkeleton,
    values: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    local_positions = skeleton.offsets.copy()
    local_rotations: list[R] = [R.identity() for _ in skeleton.joint_names]

    cursor = 0
    for joint_idx, channel_names in enumerate(skeleton.channels):
        joint_values = values[cursor: cursor + len(channel_names)]
        cursor += len(channel_names)

        if not channel_names:
            continue

        translation = local_positions[joint_idx].copy()
        for name, value in zip(channel_names, joint_values):
            axis = name[0].lower()
            if name.endswith("position"):
                if axis == "x":
                    translation[0] = value
                elif axis == "y":
                    translation[1] = value
                elif axis == "z":
                    translation[2] = value
        local_positions[joint_idx] = translation
        local_rotations[joint_idx] = _rotation_from_channels(channel_names, joint_values.tolist())

    global_positions = np.zeros_like(local_positions)
    global_rotations: list[R] = [R.identity() for _ in skeleton.joint_names]

    for joint_idx, parent_idx in enumerate(skeleton.parents):
        if parent_idx < 0:
            global_positions[joint_idx] = local_positions[joint_idx]
            global_rotations[joint_idx] = local_rotations[joint_idx]
            continue

        global_rotations[joint_idx] = global_rotations[parent_idx] * local_rotations[joint_idx]
        global_positions[joint_idx] = (
            global_positions[parent_idx]
            + global_rotations[parent_idx].apply(local_positions[joint_idx])
        )

    frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output_rot = R.from_matrix(_OUTPUT_ROTATION_MATRIX)
    for joint_idx, joint_name in enumerate(skeleton.joint_names):
        position = global_positions[joint_idx] @ _OUTPUT_ROTATION_MATRIX.T + _HEIGHT_OFFSET
        rotated = output_rot * global_rotations[joint_idx]
        quat_xyzw = rotated.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        frame[joint_name] = (position, quat_wxyz)

    # Compute End Site (fingertip) positions from parent rotation + offset
    for joint_idx, end_offset in skeleton.end_sites.items():
        joint_name = skeleton.joint_names[joint_idx]
        tip_global = (
            global_positions[joint_idx]
            + global_rotations[joint_idx].apply(end_offset)
        )
        tip_position = tip_global @ _OUTPUT_ROTATION_MATRIX.T + _HEIGHT_OFFSET
        tip_key = joint_name + "_EndSite"
        frame[tip_key] = (tip_position, frame[joint_name][1])

    return frame


class _DirectHCMocapUDPProvider:
    def __init__(
        self,
        reference_bvh: str | None,
        host: str = "",
        port: int = 1118,
        timeout: float = 30.0,
    ):
        self._skeleton = _parse_bvh_reference(reference_bvh or DEFAULT_HC_MOCAP_REFERENCE_BVH)
        self._timeout = timeout
        self._running = True
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._ready = threading.Event()
        self._latest_frame: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._frame_index = 0
        self._last_served_frame_index = 0
        self._stats = {
            "packets_received": 0,
            "packets_valid": 0,
            "packets_bad_size": 0,
            "packets_bad_decode": 0,
            "last_packet_bytes": 0,
            "last_float_count": 0,
            "expected_float_count": self._skeleton.expected_floats,
            "last_sender": None,
            "last_packet_time": None,
        }

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(2.0)
        self._sock.bind((host, port))

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    @property
    def fps(self) -> int:
        return int(round(1.0 / self._skeleton.frame_time)) if self._skeleton.frame_time > 0 else 30

    def is_available(self) -> bool:
        return self._running and self._thread.is_alive()

    def get_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not self._ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No UDP hc_mocap data received within {self._timeout}s")
        deadline = time.monotonic() + self._timeout
        with self._cond:
            while self._running and self._frame_index <= self._last_served_frame_index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No new UDP hc_mocap frame received within {self._timeout}s"
                    )
                self._cond.wait(timeout=remaining)

            assert self._latest_frame is not None
            self._last_served_frame_index = self._frame_index
            return self._latest_frame

    def latest_frame_snapshot(self) -> tuple[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None:
        with self._lock:
            if self._latest_frame is None or self._frame_index <= 0:
                return None
            return self._frame_index, dict(self._latest_frame)

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)

    def stats_snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._stats)

    def _recv_loop(self) -> None:
        while self._running:
            try:
                raw, sender = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            with self._lock:
                self._stats["packets_received"] += 1
                self._stats["last_packet_bytes"] = len(raw)
                self._stats["last_sender"] = f"{sender[0]}:{sender[1]}"
                self._stats["last_packet_time"] = time.time()

            try:
                text = raw.decode("utf-8").strip()
                if not text:
                    continue
                values = np.fromstring(text, sep=" ", dtype=np.float64)
            except UnicodeDecodeError:
                with self._lock:
                    self._stats["packets_bad_decode"] += 1
                continue

            with self._lock:
                self._stats["last_float_count"] = int(values.size)

            if values.size != self._skeleton.expected_floats:
                with self._lock:
                    self._stats["packets_bad_size"] += 1
                continue

            frame = _frame_from_bvh_values(self._skeleton, values)
            with self._cond:
                self._latest_frame = frame
                self._frame_index += 1
                self._stats["packets_valid"] += 1
                self._cond.notify_all()
            self._ready.set()

def create_hc_mocap_udp_provider(
    *,
    reference_bvh: str | None,
    hand_side: str,
    host: str = "",
    port: int = 1118,
    timeout: float = 30.0,
) -> HCMocapHandProvider:
    provider = _DirectHCMocapUDPProvider(
        reference_bvh=reference_bvh,
        host=host,
        port=port,
        timeout=timeout,
    )
    return HCMocapHandProvider(provider, hand_side)
