import numpy as np
import pytest
from pathlib import Path

from dex_mujoco.hc_mocap_input import (
    _frame_from_bvh_values,
    _parse_bvh_reference,
    HCMocapHandProvider,
    hc_mocap_frame_to_landmarks,
)


def _joint(position):
    return (np.array(position, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))


def test_hc_mocap_frame_to_landmarks_maps_left_hand_joints():
    frame = {
        "hc_Hand_L": _joint([0.0, 0.0, 0.0]),
        "hc_Thumb1_L": _joint([1.0, 0.0, 0.0]),
        "hc_Thumb2_L": _joint([2.0, 0.0, 0.0]),
        "hc_Thumb3_L": _joint([3.0, 0.0, 0.0]),
        "hc_Index1_L": _joint([0.0, 1.0, 0.0]),
        "hc_Index2_L": _joint([0.0, 2.0, 0.0]),
        "hc_Index3_L": _joint([0.0, 3.0, 0.0]),
        "hc_Middle1_L": _joint([0.0, 1.0, 1.0]),
        "hc_Middle2_L": _joint([0.0, 2.0, 1.0]),
        "hc_Middle3_L": _joint([0.0, 3.0, 1.0]),
        "hc_Ring1_L": _joint([0.0, 1.0, 2.0]),
        "hc_Ring2_L": _joint([0.0, 2.0, 2.0]),
        "hc_Ring3_L": _joint([0.0, 3.0, 2.0]),
        "hc_Pinky1_L": _joint([0.0, 1.0, 3.0]),
        "hc_Pinky2_L": _joint([0.0, 2.0, 3.0]),
        "hc_Pinky3_L": _joint([0.0, 3.0, 3.0]),
    }

    landmarks = hc_mocap_frame_to_landmarks(frame, "left")

    assert landmarks.shape == (21, 3)
    np.testing.assert_allclose(landmarks[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[3], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[4], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(landmarks[5], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(landmarks[7], [0.0, 3.0, 0.0])
    np.testing.assert_allclose(landmarks[8], [0.0, 3.0, 0.0])


def test_hc_mocap_provider_exposes_latest_detection_snapshot():
    frame = {
        "hc_Hand_L": _joint([0.0, 0.0, 0.0]),
        "hc_Thumb1_L": _joint([1.0, 0.0, 0.0]),
        "hc_Thumb2_L": _joint([2.0, 0.0, 0.0]),
        "hc_Thumb3_L": _joint([3.0, 0.0, 0.0]),
        "hc_Index1_L": _joint([0.0, 1.0, 0.0]),
        "hc_Index2_L": _joint([0.0, 2.0, 0.0]),
        "hc_Index3_L": _joint([0.0, 3.0, 0.0]),
        "hc_Middle1_L": _joint([0.0, 1.0, 1.0]),
        "hc_Middle2_L": _joint([0.0, 2.0, 1.0]),
        "hc_Middle3_L": _joint([0.0, 3.0, 1.0]),
        "hc_Ring1_L": _joint([0.0, 1.0, 2.0]),
        "hc_Ring2_L": _joint([0.0, 2.0, 2.0]),
        "hc_Ring3_L": _joint([0.0, 3.0, 2.0]),
        "hc_Pinky1_L": _joint([0.0, 1.0, 3.0]),
        "hc_Pinky2_L": _joint([0.0, 2.0, 3.0]),
        "hc_Pinky3_L": _joint([0.0, 3.0, 3.0]),
    }

    class _StubProvider:
        fps = 30

        def is_available(self):
            return True

        def get_frame(self):
            return frame

        def latest_frame_snapshot(self):
            return 7, frame

        def close(self):
            return None

    provider = HCMocapHandProvider(_StubProvider(), "left")
    snapshot = provider.latest_detection_snapshot()

    assert snapshot is not None
    frame_index, detection = snapshot
    assert frame_index == 7
    assert detection.hand_side == "left"
    assert detection.landmarks_3d.shape == (21, 3)
    np.testing.assert_allclose(detection.landmarks_3d[0], [0.0, 0.0, 0.0])

@pytest.mark.skipif(
    not Path("/home/wubingqian/project/teleop_projects/Teleopit/data/hc_mocap_bvh/motion-20260211203358.bvh").exists(),
    reason="local hc_mocap sample BVH not available",
)
def test_local_bvh_reference_parser_matches_motion_line():
    bvh_path = Path(
        "/home/wubingqian/project/teleop_projects/Teleopit/data/hc_mocap_bvh/motion-20260211203358.bvh"
    )
    skeleton = _parse_bvh_reference(str(bvh_path))

    motion_started = False
    sample_line = None
    for line in bvh_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "MOTION":
            motion_started = True
            continue
        if not motion_started or stripped.startswith("Frames:") or stripped.startswith("Frame Time:") or not stripped:
            continue
        sample_line = stripped
        break

    assert sample_line is not None
    values = np.fromstring(sample_line, sep=" ", dtype=np.float64)
    assert values.size == skeleton.expected_floats

    frame = _frame_from_bvh_values(skeleton, values)
    assert "hc_Hand_R" in frame
    assert "hc_Index3_R" in frame
    assert frame["hc_Hand_R"][0].shape == (3,)
