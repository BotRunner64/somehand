"""Pure landmark preprocessing and direction-target utilities."""

from __future__ import annotations

import numpy as np

from .hand_side import normalize_hand_side

_OPERATOR2ROBOT_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _mediapipe_to_mujoco(landmarks_3d: np.ndarray) -> np.ndarray:
    out = np.empty_like(landmarks_3d)
    out[:, 0] = -landmarks_3d[:, 2]
    out[:, 1] = landmarks_3d[:, 0]
    out[:, 2] = -landmarks_3d[:, 1]
    return out


def _estimate_wrist_frame(landmarks_3d: np.ndarray) -> np.ndarray:
    points = landmarks_3d[[0, 5, 9], :]
    x_vector = points[0] - points[2]

    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    normal = vh[-1]

    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        raise ValueError("Cannot estimate palm normal from degenerate landmarks")
    normal = normal / normal_norm

    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Cannot estimate palm x-axis from degenerate landmarks")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Cannot estimate palm z-axis from degenerate landmarks")
    z_axis = z_axis / z_norm

    if np.dot(z_axis, points[1] - points[2]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, normal, z_axis], axis=1)


def preprocess_landmarks(
    landmarks_3d: np.ndarray,
    hand_side: str = "right",
) -> np.ndarray:
    normalize_hand_side(hand_side)

    centered = landmarks_3d - landmarks_3d[0:1, :]
    try:
        wrist_frame = _estimate_wrist_frame(centered)
        return centered @ wrist_frame @ _OPERATOR2ROBOT_RIGHT
    except ValueError:
        return _mediapipe_to_mujoco(centered)


def compute_target_directions(
    landmarks_3d: np.ndarray,
    human_vector_pairs: list[tuple[int, int]],
    hand_side: str = "right",
) -> np.ndarray:
    landmarks = preprocess_landmarks(landmarks_3d, hand_side=hand_side)
    directions = np.empty((len(human_vector_pairs), 3), dtype=np.float64)
    for i, (origin_idx, target_idx) in enumerate(human_vector_pairs):
        vector = landmarks[target_idx] - landmarks[origin_idx]
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            directions[i] = 0.0
        else:
            directions[i] = vector / norm
    return directions
