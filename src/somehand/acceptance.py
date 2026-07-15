"""Acceptance helpers for validating retargeting quality."""

from dataclasses import dataclass
from time import perf_counter

import mujoco
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
from .domain.preprocessing import compute_target_directions
from .infrastructure.model_name_resolver import ModelNameResolver
_LEFT_RIGHT_ROBOT_MIRROR = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


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
        thumb[0] = [0.020, -0.016, 0.000]
        thumb[1] = [0.034, -0.032, 0.000]
        thumb[2] = [0.042, -0.048, 0.000]
        thumb[3] = [0.038, -0.062, 0.000]
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
    return mean_direction_cosine(right_dirs, left_dirs @ _LEFT_RIGHT_ROBOT_MIRROR)


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
    metrics = {
        "weighted_cosine": float(np.average(cosines, weights=weights)),
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
    }
    frame_primary_targets, frame_secondary_targets = retargeter.get_frame_target_directions()
    if frame_primary_targets is not None and frame_secondary_targets is not None:
        primary_vectors, secondary_vectors = retargeter._get_robot_frame_vectors()
        primary_cosines = np.empty(len(primary_vectors), dtype=np.float64)
        secondary_cosines = np.empty(len(secondary_vectors), dtype=np.float64)
        for index, vector in enumerate(primary_vectors):
            norm = np.linalg.norm(vector)
            if norm < 1e-8:
                primary_cosines[index] = -1.0
            else:
                primary_cosines[index] = float(np.dot(vector / norm, frame_primary_targets[index]))
        for index, vector in enumerate(secondary_vectors):
            norm = np.linalg.norm(vector)
            if norm < 1e-8:
                secondary_cosines[index] = -1.0
            else:
                secondary_cosines[index] = float(np.dot(vector / norm, frame_secondary_targets[index]))
        metrics["thumb_frame_primary_cosine"] = float(np.mean(primary_cosines))
        metrics["thumb_frame_secondary_cosine"] = float(np.mean(secondary_cosines))
    metrics.update(closure_metrics(retargeter))
    return metrics


def _resolve_generic_point(retargeter, semantic_name: str, *, obj_type) -> tuple[int, bool] | None:
    resolver = ModelNameResolver(retargeter.hand_model.model, hand_side=retargeter.config.hand.side)
    resolved = resolver.resolve_optional(semantic_name, obj_type=obj_type, role="Acceptance metric")
    if resolved is None:
        return None
    point_id = mujoco.mj_name2id(retargeter.hand_model.model, obj_type, resolved)
    if point_id < 0:
        return None
    return point_id, obj_type == mujoco.mjtObj.mjOBJ_SITE


def _point_position(retargeter, point: tuple[int, bool]) -> np.ndarray:
    index, is_site = point
    if is_site:
        return retargeter.hand_model.data.site_xpos[index]
    return retargeter.hand_model.data.xpos[index]


def closure_metrics(retargeter) -> dict[str, float]:
    thumb_tip = _resolve_generic_point(retargeter, "thumb_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE)
    metrics: dict[str, float] = {}
    scale = max(retargeter.get_robot_scale(), 1e-8)

    index_tip = _resolve_generic_point(retargeter, "index_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE)
    if thumb_tip is not None and index_tip is not None:
        metrics["pinch_thumb_index_gap"] = float(
            np.linalg.norm(_point_position(retargeter, thumb_tip) - _point_position(retargeter, index_tip))
        )
        metrics["pinch_thumb_index_gap_scaled"] = metrics["pinch_thumb_index_gap"] / scale

    closure_distances: list[float] = []
    for finger in ("index", "middle", "ring", "pinky"):
        tip = _resolve_generic_point(retargeter, f"{finger}_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE)
        base = _resolve_generic_point(retargeter, f"{finger}_base", obj_type=mujoco.mjtObj.mjOBJ_BODY)
        if tip is None or base is None:
            continue
        distance = float(np.linalg.norm(_point_position(retargeter, tip) - _point_position(retargeter, base)))
        closure_distances.append(distance)
        metrics[f"{finger}_tip_to_base"] = distance
        metrics[f"{finger}_tip_to_base_scaled"] = distance / scale
    if closure_distances:
        metrics["fist_mean_tip_to_base"] = float(np.mean(closure_distances))
        metrics["fist_max_tip_to_base"] = float(np.max(closure_distances))
        metrics["fist_mean_tip_to_base_scaled"] = metrics["fist_mean_tip_to_base"] / scale
        metrics["fist_max_tip_to_base_scaled"] = metrics["fist_max_tip_to_base"] / scale
    return metrics


def solver_quality_score(retargeter) -> dict[str, float]:
    retargeter.hand_model.reset()
    retargeter.landmark_filter.reset()
    retargeter._last_qpos = None

    weighted_scores = []
    mean_scores = []
    min_scores = []
    losses = []
    frame_primary_scores = []
    frame_secondary_scores = []
    pose_metrics: dict[str, dict[str, float]] = {}
    for pose_name in ("open", "pinch", "fist"):
        retargeter.hand_model.reset()
        retargeter.landmark_filter.reset()
        retargeter._last_qpos = None
        retargeter._prev_activations = None
        retargeter.update_targets(synthetic_hand_pose(pose_name), hand_side="right")
        retargeter.solve()
        metrics = current_alignment_metrics(retargeter)
        pose_metrics[pose_name] = metrics
        weighted_scores.append(metrics["weighted_cosine"])
        mean_scores.append(metrics["mean_cosine"])
        min_scores.append(metrics["min_cosine"])
        losses.append(retargeter.compute_error())
        if "thumb_frame_primary_cosine" in metrics:
            frame_primary_scores.append(metrics["thumb_frame_primary_cosine"])
        if "thumb_frame_secondary_cosine" in metrics:
            frame_secondary_scores.append(metrics["thumb_frame_secondary_cosine"])

    result = {
        "mean_weighted_cosine": float(np.mean(weighted_scores)),
        "min_weighted_cosine": float(np.min(weighted_scores)),
        "mean_cosine": float(np.mean(mean_scores)),
        "min_cosine": float(np.min(min_scores)),
        "mean_loss": float(np.mean(losses)),
    }
    if frame_primary_scores:
        result["mean_thumb_frame_primary_cosine"] = float(np.mean(frame_primary_scores))
    if frame_secondary_scores:
        result["mean_thumb_frame_secondary_cosine"] = float(np.mean(frame_secondary_scores))
    if "pinch_thumb_index_gap_scaled" in pose_metrics.get("pinch", {}):
        result["pinch_thumb_index_gap_scaled"] = float(pose_metrics["pinch"]["pinch_thumb_index_gap_scaled"])
    if "fist_mean_tip_to_base_scaled" in pose_metrics.get("fist", {}):
        result["fist_mean_tip_to_base_scaled"] = float(pose_metrics["fist"]["fist_mean_tip_to_base_scaled"])
    if "fist_mean_tip_to_base_scaled" in pose_metrics.get("open", {}) and "fist_mean_tip_to_base_scaled" in pose_metrics.get("fist", {}):
        result["fist_closure_ratio"] = float(
            pose_metrics["fist"]["fist_mean_tip_to_base_scaled"]
            / max(pose_metrics["open"]["fist_mean_tip_to_base_scaled"], 1e-8)
        )
    return result


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
