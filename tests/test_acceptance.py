import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.acceptance import mirror_pose_to_left, rotation_matrix, synthetic_hand_pose
from dex_mujoco.domain.preprocessing import preprocess_landmarks
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import compute_target_directions


def test_config_resolves_absolute_mjcf_path():
    config = RetargetingConfig.load("configs/retargeting/right/linkerhand_l20_right.yaml")
    assert Path(config.hand.mjcf_path).is_absolute()
    assert Path(config.hand.mjcf_path).exists()


def test_wrist_local_preprocess_is_rotation_invariant():
    config = RetargetingConfig.load("configs/retargeting/right/linkerhand_l20_right.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    base_pose = synthetic_hand_pose("open")
    base_dirs = compute_target_directions(
        base_pose,
        vector_pairs,
        hand_side="right",
    )
    rotated_pose = base_pose @ rotation_matrix("z", 70.0).T
    rotated_dirs = compute_target_directions(
        rotated_pose,
        vector_pairs,
        hand_side="right",
    )
    cosine = float(np.mean(np.sum(base_dirs * rotated_dirs, axis=1)))
    assert cosine > 0.98


def test_left_and_right_inputs_match_after_mirroring():
    config = RetargetingConfig.load("configs/retargeting/right/linkerhand_l20_right.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    right_pose = synthetic_hand_pose("pinch")
    left_pose = mirror_pose_to_left(right_pose)
    right_dirs = compute_target_directions(
        right_pose,
        vector_pairs,
        hand_side="right",
    )
    left_dirs = compute_target_directions(
        left_pose,
        vector_pairs,
        hand_side="left",
    )
    cosine = float(np.mean(np.sum(right_dirs * left_dirs, axis=1)))
    assert cosine > 0.98


def test_wrist_local_preprocess_matches_reference_operator_frame():
    pose = synthetic_hand_pose("pinch")
    centered = pose - pose[0:1, :]
    points = centered[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, normal)
    z_axis = z_axis / np.linalg.norm(z_axis)
    if np.dot(z_axis, points[1] - points[2]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    operator2robot = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    expected = centered @ np.stack([x_axis, normal, z_axis], axis=1) @ operator2robot
    actual = preprocess_landmarks(pose, hand_side="right")
    assert np.allclose(actual, expected)
