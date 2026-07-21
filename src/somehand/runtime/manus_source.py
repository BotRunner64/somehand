"""MANUS ROS 2 input adapter for somehand."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from somehand.core import HandFrame, SourceFrame, normalize_hand_side


# MANUS 25 节点 → MediaPipe 风格 21 点。
#
# 0: wrist
# 1-4: thumb
# 5-8: index
# 9-12: middle
# 13-16: ring
# 17-20: pinky
# MANUS 25-node skeleton -> MediaPipe-style 21 landmarks.
#
# MANUS semantics:
#   thumb: MCP, PIP, IP, TIP
#   other fingers: MCP, PIP, IP, DIP, TIP
#
# MediaPipe-style output keeps four joints per finger.  For the four
# non-thumb fingers, MANUS IP is the additional intermediate point and is
# intentionally omitted.
LANDMARK_KEYS = [
    ("thumb", "mcp"),
    ("thumb", "pip"),
    ("thumb", "ip"),
    ("thumb", "tip"),

    ("index", "mcp"),
    ("index", "pip"),
    ("index", "dip"),
    ("index", "tip"),

    ("middle", "mcp"),
    ("middle", "pip"),
    ("middle", "dip"),
    ("middle", "tip"),

    ("ring", "mcp"),
    ("ring", "pip"),
    ("ring", "dip"),
    ("ring", "tip"),

    ("pinky", "mcp"),
    ("pinky", "pip"),
    ("pinky", "dip"),
    ("pinky", "tip"),
]


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def manus_message_to_hand_frame(msg: Any) -> HandFrame:
    """Convert a ManusGlove ROS 2 message to a somehand HandFrame."""

    if len(msg.raw_nodes) != msg.raw_node_count:
        raise ValueError(
            f"raw_node_count={msg.raw_node_count}, "
            f"actual={len(msg.raw_nodes)}"
        )

    nodes_by_id = {
        int(node.node_id): node
        for node in msg.raw_nodes
    }

    if 0 not in nodes_by_id:
        raise ValueError("MANUS message does not contain wrist node_id=0")

    nodes_by_type = {
        (
            _normalize(node.chain_type),
            _normalize(node.joint_type),
        ): node
        for node in msg.raw_nodes
    }

    ordered_nodes = [nodes_by_id[0]]
    missing: list[str] = []

    for chain_name, joint_name in LANDMARK_KEYS:
        node = nodes_by_type.get((chain_name, joint_name))

        if node is None:
            missing.append(f"{chain_name}/{joint_name}")
        else:
            ordered_nodes.append(node)

    if missing:
        raise ValueError(
            "Missing MANUS nodes: " + ", ".join(missing)
        )

    landmarks = np.asarray(
        [
            [
                node.pose.position.x,
                node.pose.position.y,
                node.pose.position.z,
            ]
            for node in ordered_nodes
        ],
        dtype=np.float64,
    )

    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3), got {landmarks.shape}"
        )

    if not np.all(np.isfinite(landmarks)):
        raise ValueError("MANUS landmarks contain NaN or infinity")

    return HandFrame(
        landmarks_3d=landmarks,
        landmarks_2d=None,
        hand_side=normalize_hand_side(msg.side),
    )


class ManusRos2InputSource:
    """Read MANUS glove frames from a ROS 2 topic."""

    def __init__(
        self,
        *,
        topic: str,
        hand_side: str,
        timeout: float = 2.0,
        nominal_fps: int = 120,
    ) -> None:
        # Lazy import: users running webcam mode should not need ROS 2.
        try:
            import rclpy
            from manus_ros2_msgs.msg import ManusGlove
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as exc:
            raise RuntimeError(
                "MANUS input requires ROS 2 Humble, rclpy and "
                "manus_ros2_msgs. Source the ROS environments first."
            ) from exc

        self.source_desc = f"ros2://{topic}"
        self.hand_side = normalize_hand_side(hand_side)
        self.timeout = float(timeout)
        self._fps = int(nominal_fps)

        self._rclpy = rclpy
        self._latest_msg = None
        self._available = True
        self._received_count = 0
        self._converted_count = 0

        self._owns_context = not rclpy.ok()

        if self._owns_context:
            rclpy.init(args=None)

        self._node = rclpy.create_node(
            f"somehand_manus_input_{self.hand_side}"
        )

        self._subscription = self._node.create_subscription(
            ManusGlove,
            topic,
            self._message_callback,
            qos_profile_sensor_data,
        )

    @property
    def fps(self) -> int:
        return self._fps

    def _message_callback(self, msg: Any) -> None:
        self._received_count += 1

        try:
            message_side = normalize_hand_side(msg.side)
        except ValueError:
            return

        if message_side != self.hand_side:
            return

        # Only preserve the latest frame to avoid queue buildup.
        self._latest_msg = msg

    def is_available(self) -> bool:
        return (
            self._available
            and self._rclpy.ok()
        )

    def get_frame(self) -> SourceFrame:
        if not self.is_available():
            raise StopIteration

        deadline = time.monotonic() + self.timeout

        while self._latest_msg is None:
            if not self._rclpy.ok():
                self._available = False
                raise StopIteration

            remaining = deadline - time.monotonic()

            if remaining <= 0.0:
                # No glove frame during this interval. Keep session alive.
                return SourceFrame(detection=None)

            self._rclpy.spin_once(
                self._node,
                timeout_sec=min(0.05, remaining),
            )

        msg = self._latest_msg
        self._latest_msg = None

        frame = manus_message_to_hand_frame(msg)
        self._converted_count += 1

        return SourceFrame(detection=frame)

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        if not self._available:
            return

        self._available = False

        try:
            self._node.destroy_node()
        finally:
            if self._owns_context and self._rclpy.ok():
                self._rclpy.shutdown()

    def stats_snapshot(self) -> dict[str, object]:
        return {
            "messages_received": self._received_count,
            "frames_converted": self._converted_count,
        }


def create_manus_ros2_source(
    *,
    topic: str,
    hand_side: str,
    timeout: float,
) -> ManusRos2InputSource:
    return ManusRos2InputSource(
        topic=topic,
        hand_side=hand_side,
        timeout=timeout,
    )
