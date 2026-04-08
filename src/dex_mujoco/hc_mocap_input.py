"""Adapters for hc_mocap hand data."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .hand_detector import HandDetection

_DEFAULT_TELEOPIT_ROOT = (
    Path(__file__).resolve().parents[2].parent.parent / "teleop_projects" / "Teleopit"
)
_OUTPUT_ROTATION_MATRIX = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_HEIGHT_OFFSET = np.array([0.0, 0.0, 0.9526], dtype=np.float64)


def _ensure_teleopit_importable(teleopit_root: str | None = None) -> None:
    candidate_roots: list[Path] = []
    if teleopit_root:
        candidate_roots.append(Path(teleopit_root).resolve())
    env_root = os.environ.get("TELEOPIT_ROOT")
    if env_root:
        candidate_roots.append(Path(env_root).resolve())
    candidate_roots.append(_DEFAULT_TELEOPIT_ROOT.resolve())

    for root in candidate_roots:
        if not root.exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            import teleopit  # noqa: F401
            return
        except ImportError:
            continue

    raise RuntimeError(
        "Unable to import Teleopit. Install it into the current environment, set "
        "TELEOPIT_ROOT, or pass --teleopit-root."
    )


def _point(frame: dict[str, tuple[np.ndarray, np.ndarray]], joint_name: str) -> np.ndarray:
    if joint_name not in frame:
        raise KeyError(f"hc_mocap frame is missing joint '{joint_name}'")
    return np.asarray(frame[joint_name][0], dtype=np.float64)


def hc_mocap_frame_to_landmarks(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    handedness: str,
) -> np.ndarray:
    """Convert a Teleopit hc_mocap frame into MediaPipe-style 21 landmarks.

    hc_mocap exposes wrist + three joints per finger. We do not extrapolate
    fingertip positions; the MediaPipe tip slots reuse the last available
    measured joint.
    """
    side = "L" if handedness == "Left" else "R"

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

    def __init__(self, provider: object, handedness: str):
        self._provider = provider
        self.handedness = handedness

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    @property
    def fps(self) -> int:
        return int(getattr(self._provider, "fps", 30))

    def get_detection(self) -> HandDetection:
        frame = self._provider.get_frame()
        landmarks_3d = hc_mocap_frame_to_landmarks(frame, self.handedness)
        landmarks_2d = np.zeros((21, 2), dtype=np.float64)
        return HandDetection(
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            handedness=self.handedness,
        )

    def close(self) -> None:
        close_fn = getattr(self._provider, "close", None)
        if callable(close_fn):
            close_fn()

    def stats_snapshot(self) -> dict[str, object]:
        stats_fn = getattr(self._provider, "stats_snapshot", None)
        if callable(stats_fn):
            return dict(stats_fn())
        return {}


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


def _parse_bvh_reference(reference_bvh: str) -> _BvhSkeleton:
    lines = Path(reference_bvh).read_text().splitlines()
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
        reference_bvh: str,
        host: str = "",
        port: int = 1118,
        timeout: float = 30.0,
    ):
        self._skeleton = _parse_bvh_reference(reference_bvh)
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


def create_hc_mocap_bvh_provider(
    *,
    bvh_path: str,
    handedness: str,
    teleopit_root: str | None = None,
) -> HCMocapHandProvider:
    _ensure_teleopit_importable(teleopit_root)
    from teleopit.inputs.bvh_provider import BVHInputProvider

    provider = BVHInputProvider(bvh_path=bvh_path, human_format="hc_mocap")
    return HCMocapHandProvider(provider, handedness)


def create_hc_mocap_udp_provider(
    *,
    reference_bvh: str,
    handedness: str,
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
    return HCMocapHandProvider(provider, handedness)
