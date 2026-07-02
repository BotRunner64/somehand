"""Camera and landmark-geometry helpers for MuJoCo viewers."""

from __future__ import annotations

import numpy as np
import mujoco

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
LANDMARK_COLORS = np.array(
    [
        [255, 255, 255, 220],
        [255, 170, 90, 220], [255, 170, 90, 220], [255, 170, 90, 220], [255, 170, 90, 220],
        [90, 220, 255, 220], [90, 220, 255, 220], [90, 220, 255, 220], [90, 220, 255, 220],
        [120, 255, 160, 220], [120, 255, 160, 220], [120, 255, 160, 220], [120, 255, 160, 220],
        [255, 230, 110, 220], [255, 230, 110, 220], [255, 230, 110, 220], [255, 230, 110, 220],
        [255, 130, 210, 220], [255, 130, 210, 220], [255, 130, 210, 220], [255, 130, 210, 220],
    ],
    dtype=np.float32,
) / 255.0
BONE_COLORS = np.array([LANDMARK_COLORS[end] for _, end in HAND_CONNECTIONS], dtype=np.float32)
LEFT_LANDMARK_COLORS = np.array(
    [
        [255, 220, 200, 220],
        [255, 180, 120, 220], [255, 180, 120, 220], [255, 180, 120, 220], [255, 180, 120, 220],
        [255, 190, 150, 220], [255, 190, 150, 220], [255, 190, 150, 220], [255, 190, 150, 220],
        [255, 205, 170, 220], [255, 205, 170, 220], [255, 205, 170, 220], [255, 205, 170, 220],
        [255, 220, 190, 220], [255, 220, 190, 220], [255, 220, 190, 220], [255, 220, 190, 220],
        [255, 235, 210, 220], [255, 235, 210, 220], [255, 235, 210, 220], [255, 235, 210, 220],
    ],
    dtype=np.float32,
) / 255.0
RIGHT_LANDMARK_COLORS = np.array(
    [
        [220, 255, 220, 220],
        [120, 255, 140, 220], [120, 255, 140, 220], [120, 255, 140, 220], [120, 255, 140, 220],
        [140, 255, 170, 220], [140, 255, 170, 220], [140, 255, 170, 220], [140, 255, 170, 220],
        [160, 255, 190, 220], [160, 255, 190, 220], [160, 255, 190, 220], [160, 255, 190, 220],
        [180, 255, 210, 220], [180, 255, 210, 220], [180, 255, 210, 220], [180, 255, 210, 220],
        [200, 255, 230, 220], [200, 255, 230, 220], [200, 255, 230, 220], [200, 255, 230, 220],
    ],
    dtype=np.float32,
) / 255.0
LEFT_BONE_COLORS = np.array([LEFT_LANDMARK_COLORS[end] for _, end in HAND_CONNECTIONS], dtype=np.float32)
RIGHT_BONE_COLORS = np.array([RIGHT_LANDMARK_COLORS[end] for _, end in HAND_CONNECTIONS], dtype=np.float32)
IDENTITY_MAT = np.eye(3, dtype=np.float64).reshape(-1)
POINT_RADIUS = 0.006
BONE_RADIUS = 0.0035
CAMERA_MARGIN = 1.15
MIN_CAMERA_DISTANCE = 0.18
MIN_FRAMING_RADIUS = 0.01
DEFAULT_HAND_CAMERA = {
    "distance": 0.55,
    "azimuth": 145.0,
    "elevation": -20.0,
    "lookat": (0.0, 0.0, 0.0),
}
DEFAULT_BIHAND_CAMERA = {
    "distance": 0.60,
    "azimuth": -90.0,
    "elevation": -5.0,
}
DEFAULT_BIHAND_LANDMARK_CAMERA = {
    "distance": 0.60,
    "azimuth": -90.0,
    "elevation": -5.0,
    "lookat": (0.0, 0.04, 0.02),
}
DEFAULT_LANDMARK_CAMERA = dict(DEFAULT_HAND_CAMERA)
LANDMARK_VIEWER_XML = """
<mujoco model="input_landmarks">
  <visual>
    <global offwidth="800" offheight="600"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.08" rgba="0.12 0.12 0.12 1"/>
  </worldbody>
</mujoco>
"""


def _with_alpha(colors: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None:
        return colors
    rgba = np.array(colors, copy=True)
    rgba[:, 3] = float(alpha)
    return rgba


def append_single_landmark_geoms(
    scene,
    landmarks: np.ndarray,
    *,
    point_alpha: float | None = None,
    bone_alpha: float | None = None,
    point_radius: float = POINT_RADIUS,
    bone_radius: float = BONE_RADIUS,
) -> None:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")

    required_geoms = len(LANDMARK_COLORS) + len(HAND_CONNECTIONS)
    if scene.ngeom + required_geoms > scene.maxgeom:
        raise RuntimeError(
            f"Scene only supports {scene.maxgeom} geoms, "
            f"but single-hand overlay needs {scene.ngeom + required_geoms}"
        )

    point_colors = _with_alpha(LANDMARK_COLORS, point_alpha)
    bone_colors = _with_alpha(BONE_COLORS, bone_alpha)

    for point, rgba in zip(points, point_colors, strict=True):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, point_radius, dtype=np.float64),
            point,
            IDENTITY_MAT,
            rgba,
        )
        scene.ngeom += 1

    for (start_idx, end_idx), rgba in zip(HAND_CONNECTIONS, bone_colors, strict=True):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            IDENTITY_MAT,
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            bone_radius,
            points[start_idx],
            points[end_idx],
        )
        geom.rgba[:] = rgba
        scene.ngeom += 1


def append_bihand_landmark_geoms(
    scene,
    hands: np.ndarray,
    *,
    point_alpha: float | None = None,
    bone_alpha: float | None = None,
    point_radius: float = POINT_RADIUS,
    bone_radius: float = BONE_RADIUS,
) -> None:
    points = np.asarray(hands, dtype=np.float64)
    if points.shape != (2, 21, 3):
        raise ValueError(f"Expected landmarks with shape (2, 21, 3), got {points.shape}")

    required_geoms = 2 * (len(LANDMARK_COLORS) + len(HAND_CONNECTIONS))
    if scene.ngeom + required_geoms > scene.maxgeom:
        raise RuntimeError(
            f"Scene only supports {scene.maxgeom} geoms, "
            f"but bi-hand overlay needs at most {scene.ngeom + required_geoms}"
        )

    for hand_points, point_colors, bone_colors in (
        (points[0], LEFT_LANDMARK_COLORS, LEFT_BONE_COLORS),
        (points[1], RIGHT_LANDMARK_COLORS, RIGHT_BONE_COLORS),
    ):
        point_colors = _with_alpha(point_colors, point_alpha)
        bone_colors = _with_alpha(bone_colors, bone_alpha)
        finite_mask = np.isfinite(hand_points).all(axis=1)
        for idx, (point, rgba) in enumerate(zip(hand_points, point_colors, strict=True)):
            if not finite_mask[idx]:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, point_radius, dtype=np.float64),
                point,
                IDENTITY_MAT,
                rgba,
            )
            scene.ngeom += 1

        for (start_idx, end_idx), rgba in zip(HAND_CONNECTIONS, bone_colors, strict=True):
            if not (finite_mask[start_idx] and finite_mask[end_idx]):
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                IDENTITY_MAT,
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                bone_radius,
                hand_points[start_idx],
                hand_points[end_idx],
            )
            geom.rgba[:] = rgba
            scene.ngeom += 1


def configure_free_camera(
    camera,
    *,
    distance: float,
    azimuth: float,
    elevation: float,
    lookat: tuple[float, float, float],
) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.lookat[:] = np.asarray(lookat, dtype=np.float64)


def configure_default_hand_camera(camera) -> None:
    configure_free_camera(camera, **DEFAULT_HAND_CAMERA)


def camera_aspect_ratio(model) -> float:
    width = max(int(model.vis.global_.offwidth), 1)
    height = max(int(model.vis.global_.offheight), 1)
    return width / height


def camera_distance_for_radius(radius: float, *, fovy_degrees: float, aspect_ratio: float) -> float:
    safe_radius = max(float(radius), MIN_FRAMING_RADIUS)
    half_vertical = np.deg2rad(max(float(fovy_degrees), 1.0) * 0.5)
    half_horizontal = np.arctan(np.tan(half_vertical) * max(float(aspect_ratio), 1e-3))
    limiting_half_angle = max(min(half_vertical, half_horizontal), np.deg2rad(5.0))
    return max(MIN_CAMERA_DISTANCE, CAMERA_MARGIN * safe_radius / np.sin(limiting_half_angle))


def compute_bounding_sphere(
    points: np.ndarray,
    *,
    radii: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    finite_points = np.asarray(points, dtype=np.float64)
    if finite_points.ndim != 2 or finite_points.shape[1] != 3 or finite_points.shape[0] == 0:
        raise ValueError(f"Expected points with shape (N, 3), got {finite_points.shape}")

    if radii is None:
        safe_radii = np.zeros(finite_points.shape[0], dtype=np.float64)
    else:
        safe_radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if safe_radii.shape[0] != finite_points.shape[0]:
            raise ValueError("radii must have the same length as points")

    mask = np.isfinite(finite_points).all(axis=1) & np.isfinite(safe_radii) & (safe_radii >= 0.0)
    if not np.any(mask):
        return np.zeros(3, dtype=np.float64), 0.0

    finite_points = finite_points[mask]
    safe_radii = safe_radii[mask]
    mins = np.min(finite_points - safe_radii[:, None], axis=0)
    maxs = np.max(finite_points + safe_radii[:, None], axis=0)
    center = 0.5 * (mins + maxs)
    radius = np.max(np.linalg.norm(finite_points - center, axis=1) + safe_radii)
    return center, float(radius)


def try_frame_camera_to_points(
    camera,
    *,
    model,
    points: np.ndarray,
    radii: np.ndarray | None = None,
    azimuth: float,
    elevation: float,
    aspect_ratio: float | None = None,
) -> bool:
    lookat, radius = compute_bounding_sphere(points, radii=radii)
    if radius <= 0.0:
        return False

    configure_free_camera(
        camera,
        distance=camera_distance_for_radius(
            radius,
            fovy_degrees=float(model.vis.global_.fovy),
            aspect_ratio=camera_aspect_ratio(model) if aspect_ratio is None else float(aspect_ratio),
        ),
        azimuth=azimuth,
        elevation=elevation,
        lookat=tuple(lookat),
    )
    return True


def try_frame_hand_camera(
    camera,
    *,
    model,
    data,
    aspect_ratio: float | None = None,
    azimuth: float | None = None,
    elevation: float | None = None,
) -> bool:
    centers: list[np.ndarray] = []
    radii: list[float] = []

    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        if geom_type in {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_HFIELD),
        }:
            continue

        radius = float(model.geom_rbound[geom_id])
        if radius <= 0.0:
            radius = float(np.linalg.norm(model.geom_size[geom_id]))
        if not np.isfinite(radius) or radius <= 0.0:
            continue

        centers.append(np.array(data.geom_xpos[geom_id], copy=True))
        radii.append(radius)

    if not centers:
        return False

    return try_frame_camera_to_points(
        camera,
        model=model,
        points=np.asarray(centers, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        azimuth=DEFAULT_HAND_CAMERA["azimuth"] if azimuth is None else float(azimuth),
        elevation=DEFAULT_HAND_CAMERA["elevation"] if elevation is None else float(elevation),
        aspect_ratio=aspect_ratio,
    )


__all__ = [
    "HAND_CONNECTIONS",
    "LANDMARK_COLORS",
    "BONE_COLORS",
    "LEFT_LANDMARK_COLORS",
    "RIGHT_LANDMARK_COLORS",
    "LEFT_BONE_COLORS",
    "RIGHT_BONE_COLORS",
    "IDENTITY_MAT",
    "POINT_RADIUS",
    "BONE_RADIUS",
    "CAMERA_MARGIN",
    "MIN_CAMERA_DISTANCE",
    "MIN_FRAMING_RADIUS",
    "DEFAULT_HAND_CAMERA",
    "DEFAULT_BIHAND_CAMERA",
    "DEFAULT_BIHAND_LANDMARK_CAMERA",
    "DEFAULT_LANDMARK_CAMERA",
    "LANDMARK_VIEWER_XML",
    "append_single_landmark_geoms",
    "append_bihand_landmark_geoms",
    "camera_aspect_ratio",
    "camera_distance_for_radius",
    "compute_bounding_sphere",
    "configure_default_hand_camera",
    "configure_free_camera",
    "try_frame_camera_to_points",
    "try_frame_hand_camera",
]
