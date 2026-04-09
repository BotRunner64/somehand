"""Acceptance helpers for validating retargeting quality."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .constants import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    LITTLE_DIP,
    LITTLE_MCP,
    LITTLE_PIP,
    LITTLE_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from .vector_retargeting import compute_target_directions


@dataclass
class AcceptanceResult:
    name: str
    passed: bool
    metrics: dict[str, float | int | str]
    detail: str = ""


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Create a 3D rotation matrix for a principal axis."""
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported axis: {axis}")


def synthetic_hand_pose(pose: str = "open") -> np.ndarray:
    """Create a simple canonical MediaPipe-style hand pose for regression checks."""
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[WRIST] = [0.0, 0.0, 0.0]

    thumb = np.array(
        [[0.030, -0.010, 0.000], [0.052, -0.016, 0.000], [0.072, -0.022, 0.000], [0.092, -0.028, 0.000]]
    )
    index = np.array(
        [[0.022, -0.022, 0.000], [0.022, -0.054, 0.000], [0.022, -0.086, 0.000], [0.022, -0.118, 0.000]]
    )
    middle = np.array(
        [[0.000, -0.022, 0.000], [0.000, -0.060, 0.000], [0.000, -0.098, 0.000], [0.000, -0.136, 0.000]]
    )
    ring = np.array(
        [[-0.022, -0.020, 0.000], [-0.022, -0.053, 0.000], [-0.022, -0.086, 0.000], [-0.022, -0.118, 0.000]]
    )
    little = np.array(
        [[-0.044, -0.016, 0.000], [-0.044, -0.043, 0.000], [-0.044, -0.070, 0.000], [-0.044, -0.097, 0.000]]
    )

    if pose == "pinch":
        thumb[-1] = [0.048, -0.060, 0.000]
        index[-1] = [0.032, -0.082, 0.000]
        index[-2] = [0.027, -0.072, 0.000]
    elif pose == "fist":
        thumb[-1] = [0.042, -0.052, 0.000]
        index[1:] = [[0.020, -0.040, 0.006], [0.018, -0.050, 0.014], [0.012, -0.056, 0.026]]
        middle[1:] = [[0.000, -0.044, 0.008], [0.000, -0.054, 0.018], [0.000, -0.060, 0.032]]
        ring[1:] = [[-0.020, -0.040, 0.010], [-0.020, -0.050, 0.020], [-0.020, -0.056, 0.032]]
        little[1:] = [[-0.040, -0.034, 0.010], [-0.040, -0.042, 0.022], [-0.040, -0.046, 0.034]]
    elif pose != "open":
        raise ValueError(f"Unsupported synthetic pose: {pose}")

    pts[[THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]] = thumb
    pts[[INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]] = index
    pts[[MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]] = middle
    pts[[RING_MCP, RING_PIP, RING_DIP, RING_TIP]] = ring
    pts[[LITTLE_MCP, LITTLE_PIP, LITTLE_DIP, LITTLE_TIP]] = little
    return pts


def mirror_pose_to_left(landmarks_3d: np.ndarray) -> np.ndarray:
    mirrored = landmarks_3d.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def mean_direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(a * b, axis=1)))


def rotation_invariance_score(config, vector_pairs: list[tuple[int, int]]) -> float:
    """Score whether rigid palm rotations preserve target directions."""
    scores = []
    for pose_name in ("open", "pinch", "fist"):
        pose = synthetic_hand_pose(pose_name)
        base_dirs = compute_target_directions(
            pose,
            vector_pairs,
            hand_side="right",
        )
        for axis, angle in (("x", 50.0), ("y", 35.0), ("z", 70.0)):
            rotated = pose @ rotation_matrix(axis, angle).T
            dirs = compute_target_directions(
                rotated,
                vector_pairs,
                hand_side="right",
            )
            scores.append(mean_direction_cosine(base_dirs, dirs))
    return float(min(scores))


def bilateral_preprocess_consistency_score(config, vector_pairs: list[tuple[int, int]]) -> float:
    pose = synthetic_hand_pose("pinch")
    right_dirs = compute_target_directions(
        pose,
        vector_pairs,
        hand_side="right",
    )
    left_dirs = compute_target_directions(
        mirror_pose_to_left(pose),
        vector_pairs,
        hand_side="left",
    )
    return mean_direction_cosine(right_dirs, left_dirs)


def static_jitter_score(retargeter, pose: np.ndarray, num_steps: int = 24, warmup: int = 8) -> float:
    retargeter.hand_model.reset()
    retargeter.landmark_filter.reset()
    retargeter._last_qpos = None

    qpos_traj = []
    for _ in range(num_steps):
        retargeter.update_targets(pose, hand_side="right")
        qpos_traj.append(retargeter.solve().copy())

    tail = np.array(qpos_traj[warmup:])
    deltas = np.diff(tail, axis=0)
    if len(deltas) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(deltas, axis=1)))


def current_alignment_metrics(retargeter) -> dict[str, float]:
    robot_vectors = retargeter._get_robot_vectors()
    target_directions = retargeter.get_target_directions()
    weights = np.asarray(retargeter.config.vector_weights, dtype=np.float64)
    cosines = np.empty(len(robot_vectors), dtype=np.float64)
    for i, vector in enumerate(robot_vectors):
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            cosines[i] = -1.0
        else:
            cosines[i] = float(np.dot(vector / norm, target_directions[i]))
    return {
        "weighted_cosine": float(np.average(cosines, weights=weights)),
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
    }


def solver_quality_score(retargeter) -> dict[str, float]:
    retargeter.hand_model.reset()
    retargeter.landmark_filter.reset()
    retargeter._last_qpos = None

    weighted_scores = []
    mean_scores = []
    min_scores = []
    losses = []
    for pose_name in ("open", "pinch", "fist"):
        retargeter.update_targets(synthetic_hand_pose(pose_name), hand_side="right")
        retargeter.solve()
        metrics = current_alignment_metrics(retargeter)
        weighted_scores.append(metrics["weighted_cosine"])
        mean_scores.append(metrics["mean_cosine"])
        min_scores.append(metrics["min_cosine"])
        losses.append(retargeter.compute_error())

    return {
        "mean_weighted_cosine": float(np.mean(weighted_scores)),
        "min_weighted_cosine": float(np.min(weighted_scores)),
        "mean_cosine": float(np.mean(mean_scores)),
        "min_cosine": float(np.min(min_scores)),
        "mean_loss": float(np.mean(losses)),
    }


def throughput_score(retargeter, num_steps: int = 60) -> float:
    retargeter.hand_model.reset()
    retargeter.landmark_filter.reset()
    retargeter._last_qpos = None

    poses = []
    base = synthetic_hand_pose("open")
    for i in range(num_steps):
        pose = base.copy()
        phase = 2.0 * np.pi * i / max(num_steps - 1, 1)
        curl = 0.015 * (0.5 + 0.5 * np.sin(phase))
        pose[[INDEX_DIP, INDEX_TIP], 2] += curl
        pose[[MIDDLE_DIP, MIDDLE_TIP], 2] += 0.8 * curl
        pose[THUMB_TIP, 0] -= 0.3 * curl
        pose[THUMB_TIP, 1] -= 0.8 * curl
        poses.append(pose)

    tic = perf_counter()
    for pose in poses:
        retargeter.update_targets(pose, hand_side="right")
        retargeter.solve()
    elapsed = perf_counter() - tic
    return float(num_steps / elapsed)
