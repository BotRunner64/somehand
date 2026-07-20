"""MANUS ROS 2 input adapter for somehand."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from somehand.core import (
    BiHandFrame,
    BiHandSourceFrame,
    HandFrame,
    SourceFrame,
    normalize_hand_side,
)


# MANUS 25-node skeleton -> MediaPipe-style 21 landmarks.
#
# Two skeleton layouts are supported.
#
# Legacy/synthetic layout:
#   thumb: MCP, PIP, IP, TIP
#   other fingers: MCP, PIP, IP, DIP, TIP
#
# Real MetaGlove layout observed from hardware:
#   thumb: MCP, PIP, DIP, TIP
#   other fingers: MCP, PIP, IP, DIP, TIP
#
# For the real MetaGlove, MANUS includes an additional metacarpal
# point between the wrist and the anatomical finger MCP landmark.
# Therefore the MediaPipe-style four points for index/middle/ring/
# pinky are selected as PIP, IP, DIP, TIP.
LEGACY_LANDMARK_KEYS = [
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


REAL_METAGLOVE_LANDMARK_KEYS = [
    ("thumb", "mcp"),
    ("thumb", "pip"),
    ("thumb", "dip"),
    ("thumb", "tip"),

    ("index", "pip"),
    ("index", "ip"),
    ("index", "dip"),
    ("index", "tip"),

    ("middle", "pip"),
    ("middle", "ip"),
    ("middle", "dip"),
    ("middle", "tip"),

    ("ring", "pip"),
    ("ring", "ip"),
    ("ring", "dip"),
    ("ring", "tip"),

    ("pinky", "pip"),
    ("pinky", "ip"),
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

    is_real_metaglove = (
        ("thumb", "dip") in nodes_by_type
        and ("thumb", "ip") not in nodes_by_type
    )

    landmark_keys = (
        REAL_METAGLOVE_LANDMARK_KEYS
        if is_real_metaglove
        else LEGACY_LANDMARK_KEYS
    )

    ordered_nodes = [nodes_by_id[0]]
    missing: list[str] = []

    for chain_name, joint_name in landmark_keys:
        node = nodes_by_type.get(
            (chain_name, joint_name)
        )

        if node is None:
            missing.append(
                f"{chain_name}/{joint_name}"
            )
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
        self._timeout_count = 0

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
                self._timeout_count += 1

                print(
                    "MANUS input timeout: "
                    f"no matching {self.hand_side} frame received "
                    f"from {self.source_desc} for "
                    f"{self.timeout:.3f}s; stopping session.",
                    flush=True,
                )

                # A live robot must never continue using the last
                # command after the MANUS stream disappears.
                raise StopIteration

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
            "timeouts": self._timeout_count,
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


class BiHandManusRos2InputSource:
    """Read fresh left and right MANUS frames from two ROS 2 topics.

    A bi-hand frame is emitted only after both sides provide a fresh message.
    If either side stops for ``timeout`` seconds, ``StopIteration`` is raised
    so the session exits rather than reusing stale glove data.
    """

    def __init__(
        self,
        *,
        left_topic: str,
        right_topic: str,
        timeout: float = 2.0,
        nominal_fps: int = 120,
    ) -> None:
        if left_topic == right_topic:
            raise ValueError("left_topic and right_topic must differ")
        if float(timeout) <= 0.0:
            raise ValueError("timeout must be > 0")
        if int(nominal_fps) <= 0:
            raise ValueError("nominal_fps must be > 0")

        try:
            import rclpy
            from manus_ros2_msgs.msg import ManusGlove
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as exc:
            raise RuntimeError(
                "MANUS bi-hand input requires ROS 2 Humble, rclpy and "
                "manus_ros2_msgs. Source the ROS environments first."
            ) from exc

        self.left_topic = str(left_topic)
        self.right_topic = str(right_topic)
        self.source_desc = (
            f"ros2://left={self.left_topic};right={self.right_topic}"
        )
        self.timeout = float(timeout)
        self._fps = int(nominal_fps)
        self._rclpy = rclpy
        self._latest_msgs: dict[str, Any | None] = {
            "left": None,
            "right": None,
        }
        self._available = True
        self._received_count = {"left": 0, "right": 0}
        self._converted_count = {"left": 0, "right": 0}
        self._side_mismatch_count = {"left": 0, "right": 0}
        self._timeout_count = 0

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)

        self._node = rclpy.create_node("somehand_manus_bihand_input")
        self._left_subscription = self._node.create_subscription(
            ManusGlove,
            self.left_topic,
            self._left_message_callback,
            qos_profile_sensor_data,
        )
        self._right_subscription = self._node.create_subscription(
            ManusGlove,
            self.right_topic,
            self._right_message_callback,
            qos_profile_sensor_data,
        )

    @property
    def fps(self) -> int:
        return self._fps

    def _left_message_callback(self, msg: Any) -> None:
        self._store_message("left", msg)

    def _right_message_callback(self, msg: Any) -> None:
        self._store_message("right", msg)

    def _store_message(self, expected_side: str, msg: Any) -> None:
        self._received_count[expected_side] += 1
        try:
            message_side = normalize_hand_side(msg.side)
        except ValueError:
            self._side_mismatch_count[expected_side] += 1
            return
        if message_side != expected_side:
            self._side_mismatch_count[expected_side] += 1
            return
        self._latest_msgs[expected_side] = msg

    def is_available(self) -> bool:
        return self._available and self._rclpy.ok()

    def get_frame(self) -> BiHandSourceFrame:
        if not self.is_available():
            raise StopIteration

        deadline = time.monotonic() + self.timeout
        while (
            self._latest_msgs["left"] is None
            or self._latest_msgs["right"] is None
        ):
            if not self._rclpy.ok():
                self._available = False
                raise StopIteration

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._timeout_count += 1
                missing = [
                    side
                    for side in ("left", "right")
                    if self._latest_msgs[side] is None
                ]
                print(
                    "MANUS bi-hand input timeout: "
                    f"no fresh matching {'+'.join(missing)} frame received "
                    f"for {self.timeout:.3f}s; stopping session.",
                    flush=True,
                )
                raise StopIteration

            self._rclpy.spin_once(
                self._node,
                timeout_sec=min(0.05, remaining),
            )

        left_msg = self._latest_msgs["left"]
        right_msg = self._latest_msgs["right"]
        self._latest_msgs["left"] = None
        self._latest_msgs["right"] = None

        left_frame = manus_message_to_hand_frame(left_msg)
        right_frame = manus_message_to_hand_frame(right_msg)
        if left_frame.hand_side != "left":
            raise ValueError(
                f"Expected left MANUS frame, got {left_frame.hand_side!r}"
            )
        if right_frame.hand_side != "right":
            raise ValueError(
                f"Expected right MANUS frame, got {right_frame.hand_side!r}"
            )

        self._converted_count["left"] += 1
        self._converted_count["right"] += 1
        return BiHandSourceFrame(
            detection=BiHandFrame(
                left=left_frame,
                right=right_frame,
            )
        )

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
            "left_messages_received": self._received_count["left"],
            "right_messages_received": self._received_count["right"],
            "left_frames_converted": self._converted_count["left"],
            "right_frames_converted": self._converted_count["right"],
            "left_side_mismatches": self._side_mismatch_count["left"],
            "right_side_mismatches": self._side_mismatch_count["right"],
            "timeouts": self._timeout_count,
        }


def create_bihand_manus_ros2_source(
    *,
    left_topic: str,
    right_topic: str,
    timeout: float,
) -> BiHandManusRos2InputSource:
    return BiHandManusRos2InputSource(
        left_topic=left_topic,
        right_topic=right_topic,
        timeout=timeout,
    )
