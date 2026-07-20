"""Unit tests for the MANUS ROS 2 input conversion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from somehand.runtime.manus_source import ManusRos2InputSource, manus_message_to_hand_frame


MANUS_SPECS = [
    ("Hand", "Invalid"),
    ("Thumb", "MCP"),
    ("Thumb", "PIP"),
    ("Thumb", "IP"),
    ("Thumb", "TIP"),
    ("Index", "MCP"),
    ("Index", "PIP"),
    ("Index", "IP"),
    ("Index", "DIP"),
    ("Index", "TIP"),
    ("Middle", "MCP"),
    ("Middle", "PIP"),
    ("Middle", "IP"),
    ("Middle", "DIP"),
    ("Middle", "TIP"),
    ("Ring", "MCP"),
    ("Ring", "PIP"),
    ("Ring", "IP"),
    ("Ring", "DIP"),
    ("Ring", "TIP"),
    ("Pinky", "MCP"),
    ("Pinky", "PIP"),
    ("Pinky", "IP"),
    ("Pinky", "DIP"),
    ("Pinky", "TIP"),
]

EXPECTED_NODE_IDS = [
    0,
    1, 2, 3, 4,
    5, 6, 8, 9,
    10, 11, 13, 14,
    15, 16, 18, 19,
    20, 21, 23, 24,
]


def _node(node_id: int, chain: str, joint: str):
    return SimpleNamespace(
        node_id=node_id,
        chain_type=chain,
        joint_type=joint,
        pose=SimpleNamespace(
            position=SimpleNamespace(
                x=float(node_id),
                y=float(node_id) + 0.1,
                z=float(node_id) + 0.2,
            )
        ),
    )


def _message(side: str = "Right"):
    nodes = [
        _node(node_id, chain, joint)
        for node_id, (chain, joint) in enumerate(MANUS_SPECS)
    ]

    return SimpleNamespace(
        side=side,
        raw_nodes=nodes,
        raw_node_count=len(nodes),
    )


def test_complete_manus_message_maps_to_media_pipe_order():
    frame = manus_message_to_hand_frame(_message())

    assert frame.landmarks_3d.shape == (21, 3)
    assert np.all(np.isfinite(frame.landmarks_3d))

    np.testing.assert_allclose(
        frame.landmarks_3d[:, 0],
        np.asarray(EXPECTED_NODE_IDS, dtype=np.float64),
    )


def test_real_thumb_dip_is_accepted_as_thumb_ip_landmark():
    msg = _message()

    # Real MetaGlove messages use Thumb/DIP where the synthetic fixture
    # historically used Thumb/IP.
    msg.raw_nodes[3].joint_type = "DIP"

    frame = manus_message_to_hand_frame(msg)

    np.testing.assert_allclose(
        frame.landmarks_3d[3],
        np.asarray([3.0, 3.1, 3.2]),
    )


def test_raw_node_count_mismatch_is_rejected():
    msg = _message()
    msg.raw_node_count += 1

    with pytest.raises(ValueError, match="raw_node_count"):
        manus_message_to_hand_frame(msg)


def test_nan_landmark_is_rejected():
    msg = _message()
    msg.raw_nodes[8].pose.position.x = float("nan")

    with pytest.raises(ValueError, match="NaN or infinity"):
        manus_message_to_hand_frame(msg)


def test_real_metaglove_shifts_non_thumb_landmarks():
    msg = _message()

    # Real MetaGlove uses Thumb/DIP instead of the legacy Thumb/IP.
    msg.raw_nodes[3].joint_type = "DIP"

    frame = manus_message_to_hand_frame(msg)

    def node_position(node_index):
        position = msg.raw_nodes[node_index].pose.position
        return np.asarray(
            [
                position.x,
                position.y,
                position.z,
            ],
            dtype=np.float64,
        )

    # Real MetaGlove non-thumb layout:
    # MCP, PIP, IP, DIP, TIP
    #
    # MediaPipe output:
    # MCP, PIP, DIP, TIP
    #
    # Therefore the first MANUS MCP point is omitted.
    expected_raw_indices = [
        6, 7, 8, 9,        # index
        11, 12, 13, 14,    # middle
        16, 17, 18, 19,    # ring
        21, 22, 23, 24,    # pinky
    ]

    for output_index, raw_index in enumerate(
        expected_raw_indices,
        start=5,
    ):
        np.testing.assert_allclose(
            frame.landmarks_3d[output_index],
            node_position(raw_index),
        )



class _FakeRclpy:
    def ok(self) -> bool:
        return True

    def spin_once(
        self,
        node,
        *,
        timeout_sec: float,
    ) -> None:
        return None


def test_manus_source_timeout_raises_stop_iteration(
    capsys,
):
    source = ManusRos2InputSource.__new__(
        ManusRos2InputSource
    )

    source.source_desc = "ros2:///manus_glove_0"
    source.hand_side = "right"
    source.timeout = 0.0
    source._rclpy = _FakeRclpy()
    source._node = object()
    source._latest_msg = None
    source._available = True
    source._timeout_count = 0

    with pytest.raises(StopIteration):
        source.get_frame()

    captured = capsys.readouterr()

    assert "MANUS input timeout" in captured.out
    assert source._timeout_count == 1
    assert source._available is True
