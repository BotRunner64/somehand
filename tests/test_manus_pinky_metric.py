"""Tests for per-finger MANUS curl metrics."""

from __future__ import annotations

import json

import numpy as np

from somehand.application.manus_calibration import (
    FINGER_NAMES,
    finger_curl_degrees,
    finger_joint_sum_degrees,
    load_manus_calibration_profile,
)


def test_joint_sum_detects_mcp_only_flexion():
    landmarks = np.zeros(
        (21, 3),
        dtype=np.float64,
    )

    # Wrist -> MCP points along +X.
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[17] = [1.0, 0.0, 0.0]

    # The entire pinky points along +Y. This represents an
    # MCP-only bend: internal segments remain parallel.
    landmarks[18] = [1.0, 1.0, 0.0]
    landmarks[19] = [1.0, 2.0, 0.0]
    landmarks[20] = [1.0, 3.0, 0.0]

    indices = (17, 18, 19, 20)

    assert finger_curl_degrees(
        landmarks,
        indices,
    ) == 0.0

    assert finger_joint_sum_degrees(
        landmarks,
        indices,
    ) == 90.0


def test_profile_supports_per_finger_metric(tmp_path):
    path = tmp_path / "profile.json"

    fingers = {
        name: {
            "open_deg": 0.0,
            "comfortable_closed_deg": 90.0,
            "metric": (
                "joint_sum"
                if name == "pinky"
                else "current"
            ),
        }
        for name in FINGER_NAMES
    }

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "side": "right",
                "fingers": fingers,
            }
        ),
        encoding="utf-8",
    )

    profile = load_manus_calibration_profile(
        path,
        expected_side="right",
    )

    assert profile.fingers["ring"].metric == "current"
    assert profile.fingers["pinky"].metric == "joint_sum"
