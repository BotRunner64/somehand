"""Unit tests for the MANUS ROS 2 input conversion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from somehand.runtime.manus_source import manus_message_to_hand_frame


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


def test_thumb_uses_ip_not_nonexistent_dip():
    msg = _message()

    # Remove the real Thumb/IP semantic and replace it with Thumb/DIP.
    msg.raw_nodes[3].joint_type = "DIP"

    with pytest.raises(ValueError, match=r"thumb/ip"):
        manus_message_to_hand_frame(msg)


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
