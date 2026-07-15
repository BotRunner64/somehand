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
DISTANCE_RGBA = np.array([0.8, 0.25, 1.0, 0.78], dtype=np.float32)
FRAME_PRIMARY_RGBA = np.array([1.0, 0.08, 0.05, 0.82], dtype=np.float32)
FRAME_SECONDARY_RGBA = np.array([0.1, 0.9, 0.18, 0.82], dtype=np.float32)
FRAME_NORMAL_RGBA = np.array([0.15, 0.35, 1.0, 0.82], dtype=np.float32)
ANGLE_RGBA = np.array([1.0, 0.92, 0.12, 0.9], dtype=np.float32)
VARIABLE_LOW_RGBA = np.array([0.22, 0.38, 0.62, 0.88], dtype=np.float32)
VARIABLE_HIGH_RGBA = np.array([1.0, 0.08, 0.04, 0.9], dtype=np.float32)
VECTOR_RADIUS = 0.002
TARGET_VECTOR_RADIUS = 0.0013
DIAGNOSTIC_THIN_RADIUS = 0.0014
LANDMARK_FRAME_AXIS_RADIUS = 0.0028
LANDMARK_FRAME_AXIS_LENGTH = 0.05
TIP_RADIUS = 0.0035
VARIABLE_MARKER_RADIUS = 0.005
ANGLE_MARKER_RADIUS = 0.0065


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


def append_landmark_frame_geoms(
    scene,
    landmarks: np.ndarray,
    frame_triples: Sequence[tuple[int, int, int]],
    *,
    axis_length: float = LANDMARK_FRAME_AXIS_LENGTH,
    radius: float = LANDMARK_FRAME_AXIS_RADIUS,
) -> None:
    points = np.asarray(landmarks, dtype=np.float64)
    for origin_idx, primary_idx, secondary_idx in frame_triples:
        if max(origin_idx, primary_idx, secondary_idx) >= len(points):
            continue
        origin = points[origin_idx]
        primary = points[primary_idx]
        secondary = points[secondary_idx]
        if not (np.isfinite(origin).all() and np.isfinite(primary).all() and np.isfinite(secondary).all()):
            continue
        primary_vector = primary - origin
        primary_norm = np.linalg.norm(primary_vector)
        if primary_norm < 1e-8:
            continue
        primary_axis = primary_vector / primary_norm
        secondary_vector = secondary - origin
        secondary_axis = secondary_vector - np.dot(secondary_vector, primary_axis) * primary_axis
        secondary_norm = np.linalg.norm(secondary_axis)
        if secondary_norm < 1e-8:
            continue
        secondary_axis = secondary_axis / secondary_norm
        normal_axis = np.cross(primary_axis, secondary_axis)
        before_ngeom = scene.ngeom
        append_vector_segments(
            scene,
            np.asarray([origin, origin, origin], dtype=np.float64),
            np.asarray(
                [
                    origin + primary_axis * axis_length,
                    origin + secondary_axis * axis_length,
                    origin + normal_axis * axis_length,
                ],
                dtype=np.float64,
            ),
            rgba=FRAME_PRIMARY_RGBA,
            radius=radius,
            tip_radius=TIP_RADIUS * 1.2,
        )
        if scene.ngeom - before_ngeom != 6:
            continue
        scene.geoms[scene.ngeom - 4].rgba[:] = FRAME_SECONDARY_RGBA
        scene.geoms[scene.ngeom - 3].rgba[:] = FRAME_SECONDARY_RGBA
        scene.geoms[scene.ngeom - 2].rgba[:] = FRAME_NORMAL_RGBA
        scene.geoms[scene.ngeom - 1].rgba[:] = FRAME_NORMAL_RGBA


def append_landmark_angle_geoms(
    scene,
    landmarks: np.ndarray,
    angle_triples: Sequence[tuple[int, int, int]],
) -> None:
    points = np.asarray(landmarks, dtype=np.float64)
    for first_idx, middle_idx, second_idx in angle_triples:
        if max(first_idx, middle_idx, second_idx) >= len(points):
            continue
        middle = points[middle_idx]
        append_vector_segments(
            scene,
            np.asarray([middle, middle], dtype=np.float64),
            np.asarray([points[first_idx], points[second_idx]], dtype=np.float64),
            rgba=ANGLE_RGBA,
            radius=DIAGNOSTIC_THIN_RADIUS,
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
    "ANGLE_MARKER_RADIUS",
    "ANGLE_RGBA",
    "DIAGNOSTIC_THIN_RADIUS",
    "DISTANCE_RGBA",
    "FRAME_NORMAL_RGBA",
    "FRAME_PRIMARY_RGBA",
    "FRAME_SECONDARY_RGBA",
    "HUMAN_VECTOR_RGBA",
    "LANDMARK_FRAME_AXIS_LENGTH",
    "LANDMARK_FRAME_AXIS_RADIUS",
    "ROBOT_VECTOR_RGBA",
    "TARGET_VECTOR_RGBA",
    "TARGET_VECTOR_RADIUS",
    "TIP_RADIUS",
    "VARIABLE_HIGH_RGBA",
    "VARIABLE_LOW_RGBA",
    "VARIABLE_MARKER_RADIUS",
    "VECTOR_RADIUS",
    "VectorPair",
    "append_landmark_angle_geoms",
    "append_landmark_frame_geoms",
    "append_landmark_vector_geoms",
    "append_variable_markers",
    "append_vector_segments",
    "target_direction_ends",
    "variable_marker_rgba",
]
