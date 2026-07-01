"""MuJoCo user-scene geometry for retargeting vectors."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from .viewer_camera import IDENTITY_MAT

VectorPair = tuple[int, int]

HUMAN_VECTOR_RGBA = np.array([0.05, 0.85, 1.0, 0.92], dtype=np.float32)
ROBOT_VECTOR_RGBA = np.array([1.0, 0.58, 0.12, 0.88], dtype=np.float32)
TARGET_VECTOR_RGBA = np.array([0.0, 0.95, 1.0, 0.72], dtype=np.float32)
VARIABLE_LOW_RGBA = np.array([0.22, 0.38, 0.62, 0.88], dtype=np.float32)
VARIABLE_HIGH_RGBA = np.array([1.0, 0.08, 0.04, 0.9], dtype=np.float32)
VECTOR_RADIUS = 0.002
TARGET_VECTOR_RADIUS = 0.0013
TIP_RADIUS = 0.0035
VARIABLE_MARKER_RADIUS = 0.005


def append_vector_segments(
    scene,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    rgba: np.ndarray,
    radius: float = VECTOR_RADIUS,
    tip_radius: float = TIP_RADIUS,
) -> None:
    start_points = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    end_points = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
    count = min(len(start_points), len(end_points))
    if count == 0:
        return
    required_geoms = 2 * count
    if scene.ngeom + required_geoms > scene.maxgeom:
        raise RuntimeError(
            f"Scene only supports {scene.maxgeom} geoms, "
            f"but vector overlay needs at least {scene.ngeom + required_geoms}"
        )

    for start, end in zip(start_points[:count], end_points[:count], strict=True):
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            continue
        if np.linalg.norm(end - start) < 1e-7:
            continue

        segment = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            segment,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            IDENTITY_MAT,
            rgba,
        )
        mujoco.mjv_connector(
            segment,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            start,
            end,
        )
        segment.rgba[:] = rgba
        scene.ngeom += 1

        tip = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            tip,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, tip_radius, dtype=np.float64),
            end,
            IDENTITY_MAT,
            rgba,
        )
        scene.ngeom += 1


def append_landmark_vector_geoms(
    scene,
    landmarks: np.ndarray,
    vector_pairs: Sequence[VectorPair],
    *,
    rgba: np.ndarray = HUMAN_VECTOR_RGBA,
) -> None:
    points = np.asarray(landmarks, dtype=np.float64)
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for origin_idx, target_idx in vector_pairs:
        if origin_idx >= len(points) or target_idx >= len(points):
            continue
        starts.append(points[origin_idx])
        ends.append(points[target_idx])
    append_vector_segments(
        scene,
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
        rgba=rgba,
    )


def target_direction_ends(
    starts: np.ndarray,
    current_ends: np.ndarray,
    target_directions: np.ndarray | None,
    *,
    max_length: float | None = None,
) -> np.ndarray | None:
    if target_directions is None:
        return None
    origins = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    tasks = np.asarray(current_ends, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(target_directions, dtype=np.float64).reshape(-1, 3)
    count = min(len(origins), len(tasks), len(directions))
    if count == 0:
        return None

    ends = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        direction = directions[index]
        direction_norm = np.linalg.norm(direction)
        current_length = np.linalg.norm(tasks[index] - origins[index])
        if max_length is not None:
            current_length = min(current_length, float(max_length))
        if direction_norm < 1e-7 or current_length < 1e-7:
            ends[index] = origins[index]
        else:
            ends[index] = origins[index] + direction / direction_norm * current_length
    return ends


def variable_marker_rgba(value: float, low: float, high: float) -> np.ndarray:
    if high <= low:
        normalized = 0.0
    else:
        normalized = (float(value) - float(low)) / (float(high) - float(low))
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return ((1.0 - normalized) * VARIABLE_LOW_RGBA + normalized * VARIABLE_HIGH_RGBA).astype(np.float32)


def append_variable_markers(
    scene,
    positions: np.ndarray,
    rgba_values: np.ndarray,
    *,
    radius: float = VARIABLE_MARKER_RADIUS,
) -> None:
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(rgba_values, dtype=np.float32).reshape(-1, 4)
    count = min(len(points), len(colors))
    if count == 0:
        return
    if scene.ngeom + count > scene.maxgeom:
        raise RuntimeError(
            f"Scene only supports {scene.maxgeom} geoms, "
            f"but variable overlay needs at least {scene.ngeom + count}"
        )
    for point, rgba in zip(points[:count], colors[:count], strict=True):
        if not np.isfinite(point).all():
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius, dtype=np.float64),
            point,
            IDENTITY_MAT,
            rgba,
        )
        scene.ngeom += 1


__all__ = [
    "HUMAN_VECTOR_RGBA",
    "ROBOT_VECTOR_RGBA",
    "TARGET_VECTOR_RGBA",
    "TARGET_VECTOR_RADIUS",
    "TIP_RADIUS",
    "VARIABLE_HIGH_RGBA",
    "VARIABLE_LOW_RGBA",
    "VARIABLE_MARKER_RADIUS",
    "VECTOR_RADIUS",
    "VectorPair",
    "append_landmark_vector_geoms",
    "append_variable_markers",
    "append_vector_segments",
    "target_direction_ends",
    "variable_marker_rgba",
]
