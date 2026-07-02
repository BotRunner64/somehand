"""Hand and bi-hand MuJoCo viewer implementations."""

from __future__ import annotations

import mujoco
import numpy as np

from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.model_name_resolver import ModelNameResolver

from .viewer_camera import DEFAULT_BIHAND_CAMERA, DEFAULT_HAND_CAMERA, configure_free_camera, try_frame_hand_camera
from .viewer_passive import ManagedPassiveViewer, compile_model_with_name, mujoco_key_callback, set_viewer_overlay_label, set_viewer_window_title
from .vector_visualization import (
    ANGLE_MARKER_RADIUS,
    ANGLE_RGBA,
    DIAGNOSTIC_THIN_RADIUS,
    DISTANCE_RGBA,
    FRAME_NORMAL_RGBA,
    FRAME_PRIMARY_RGBA,
    FRAME_SECONDARY_RGBA,
    ROBOT_VECTOR_RGBA,
    TARGET_VECTOR_RADIUS,
    TARGET_VECTOR_RGBA,
    append_variable_markers,
    append_vector_segments,
    target_direction_ends,
    variable_marker_rgba,
)

RobotVectorSpec = tuple[int, str, str, str, str]
ResolvedVectorPoint = tuple[int, bool, int, bool, int]
RobotDistanceSpec = tuple[int, str, str, str, str]
RobotFrameSpec = tuple[int, str, str, str, str, str, str]
RobotAngleSpec = tuple[int, str]
ResolvedDistancePoint = tuple[int, bool, int, bool, int]
ResolvedFramePoint = tuple[int, bool, np.ndarray, np.ndarray, int]
ResolvedAnglePoint = tuple[int, int, int, float, float]
VariableMarkerSpec = tuple[int, int, float, float]
DIAGNOSTIC_ALPHA = 0.28
TARGET_VECTOR_MAX_LENGTH = 0.035
FINGERTIP_SITE_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _quat_to_rotation_matrix(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotate_direction_targets(targets: np.ndarray | None, rotation: np.ndarray) -> np.ndarray | None:
    if targets is None:
        return None
    directions = np.asarray(targets, dtype=np.float64)
    if directions.size == 0:
        return directions.copy()
    return directions.reshape(-1, 3) @ rotation.T


class HandVisualizer:
    """Real-time MuJoCo visualization of the retargeted robot hand."""

    def __init__(
        self,
        hand_model: HandModel,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
        viewer_mode: str = "normal",
        hand_side: str | None = None,
        robot_vector_specs: list[RobotVectorSpec] | None = None,
        robot_distance_specs: list[RobotDistanceSpec] | None = None,
        robot_frame_specs: list[RobotFrameSpec] | None = None,
        robot_angle_specs: list[RobotAngleSpec] | None = None,
    ):
        self.hand_model = hand_model
        self._diagnostic = viewer_mode == "diagnostic"
        if window_title or self._diagnostic:
            self.model, self.data = compile_model_with_name(hand_model.mjcf_path, window_title or "somehand_diagnostic")
        else:
            self.model = hand_model.model
            self.data = hand_model.data
        if self._diagnostic:
            apply_model_alpha(self.model, DIAGNOSTIC_ALPHA)
        set_fingertip_site_visibility(self.model, visible=self._diagnostic)
        self._overlay_label = overlay_label
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        set_viewer_window_title(self.viewer, window_title)
        set_viewer_overlay_label(self.viewer, self._overlay_label)
        self._vector_points = resolve_robot_vector_points(
            self.model,
            robot_vector_specs or [],
            hand_side=hand_side,
        )
        self._distance_points = resolve_robot_distance_points(
            self.model,
            robot_distance_specs or [],
            hand_side=hand_side,
        )
        self._frame_points = resolve_robot_frame_points(
            self.model,
            robot_frame_specs or [],
            hand_side=hand_side,
        )
        self._angle_points = resolve_robot_angle_points(
            self.model,
            robot_angle_specs or [],
            hand_side=hand_side,
        )
        self._variable_markers = resolve_variable_markers(self.model) if self._diagnostic else []
        self._configure_camera(**DEFAULT_HAND_CAMERA)
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(
        self,
        qpos: np.ndarray,
        target_directions: np.ndarray | None = None,
        *,
        target_frame_primary_directions: np.ndarray | None = None,
        target_frame_secondary_directions: np.ndarray | None = None,
        target_distances: np.ndarray | None = None,
        target_angles: np.ndarray | None = None,
    ):
        with self.viewer.lock():
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and try_frame_hand_camera(self.viewer.cam, model=self.model, data=self.data):
                self._camera_initialized = True
            self._update_vector_overlay(
                target_directions,
                target_frame_primary_directions=target_frame_primary_directions,
                target_frame_secondary_directions=target_frame_secondary_directions,
                target_distances=target_distances,
                target_angles=target_angles,
            )
        set_viewer_overlay_label(self.viewer, self._overlay_label)
        self.viewer.sync()

    def _update_vector_overlay(
        self,
        target_directions: np.ndarray | None,
        *,
        target_frame_primary_directions: np.ndarray | None = None,
        target_frame_secondary_directions: np.ndarray | None = None,
        target_distances: np.ndarray | None = None,
        target_angles: np.ndarray | None = None,
    ) -> None:
        scene = self.viewer.user_scn
        if scene is None:
            return
        scene.ngeom = 0
        vector_points = getattr(self, "_vector_points", [])
        distance_points = getattr(self, "_distance_points", [])
        frame_points = getattr(self, "_frame_points", [])
        angle_points = getattr(self, "_angle_points", [])
        variable_markers = getattr(self, "_variable_markers", [])
        if vector_points:
            starts, current_ends, target_indices = robot_vector_points(self.model, self.data, vector_points)
            append_vector_segments(scene, starts, current_ends, rgba=ROBOT_VECTOR_RGBA)
            target_starts, target_current_ends, selected_targets = select_target_vectors(
                starts,
                current_ends,
                target_directions,
                target_indices,
            )
            target_ends = target_direction_ends(
                target_starts,
                target_current_ends,
                selected_targets,
                max_length=TARGET_VECTOR_MAX_LENGTH,
            )
            if target_ends is not None:
                append_vector_segments(
                    scene,
                    target_starts[: len(target_ends)],
                    target_ends,
                    rgba=TARGET_VECTOR_RGBA,
                    radius=TARGET_VECTOR_RADIUS,
                )
        if distance_points:
            starts, ends, target_indices = robot_distance_points(self.model, self.data, distance_points)
            append_vector_segments(scene, starts, ends, rgba=DISTANCE_RGBA, radius=DIAGNOSTIC_THIN_RADIUS)
            target_ends = distance_target_ends(starts, ends, target_distances, target_indices)
            if target_ends is not None:
                append_vector_segments(
                    scene,
                    starts[: len(target_ends)],
                    target_ends,
                    rgba=DISTANCE_RGBA,
                    radius=TARGET_VECTOR_RADIUS,
                    tip_radius=ANGLE_MARKER_RADIUS * 0.6,
                )
        if frame_points:
            append_frame_axes(
                scene,
                self.data,
                frame_points,
                target_frame_primary_directions=target_frame_primary_directions,
                target_frame_secondary_directions=target_frame_secondary_directions,
            )
        if angle_points:
            positions, colors = angle_marker_points(self.model, self.data, angle_points, target_angles)
            append_variable_markers(scene, positions, colors, radius=ANGLE_MARKER_RADIUS)
        if variable_markers:
            positions, colors = variable_marker_points(self.model, self.data, variable_markers)
            append_variable_markers(scene, positions, colors)

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class BiHandScene:
    """Combined MuJoCo scene containing left and right hand models."""

    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
        viewer_mode: str = "normal",
        left_hand_side: str | None = None,
        right_hand_side: str | None = None,
        left_robot_vector_specs: list[RobotVectorSpec] | None = None,
        right_robot_vector_specs: list[RobotVectorSpec] | None = None,
        left_robot_distance_specs: list[RobotDistanceSpec] | None = None,
        right_robot_distance_specs: list[RobotDistanceSpec] | None = None,
        left_robot_frame_specs: list[RobotFrameSpec] | None = None,
        right_robot_frame_specs: list[RobotFrameSpec] | None = None,
        left_robot_angle_specs: list[RobotAngleSpec] | None = None,
        right_robot_angle_specs: list[RobotAngleSpec] | None = None,
    ):
        self.left_hand_model = left_hand_model
        self.right_hand_model = right_hand_model
        self.left_pos = tuple(float(value) for value in left_pos)
        self.right_pos = tuple(float(value) for value in right_pos)
        self.left_quat = tuple(float(value) for value in left_quat)
        self.right_quat = tuple(float(value) for value in right_quat)
        self.left_rotation = _quat_to_rotation_matrix(self.left_quat)
        self.right_rotation = _quat_to_rotation_matrix(self.right_quat)
        self._diagnostic = viewer_mode == "diagnostic"
        self.model, self.data = self._build_model()
        if self._diagnostic:
            apply_model_alpha(self.model, DIAGNOSTIC_ALPHA)
        set_fingertip_site_visibility(self.model, visible=self._diagnostic)
        self.left_qpos_indices = self._resolve_qpos_indices(left_hand_model, prefix="left_")
        self.right_qpos_indices = self._resolve_qpos_indices(right_hand_model, prefix="right_")
        self.left_vector_points = resolve_robot_vector_points(
            self.model,
            left_robot_vector_specs or [],
            hand_side=left_hand_side,
            source_model=left_hand_model.model,
            prefix="left_",
        )
        self.right_vector_points = resolve_robot_vector_points(
            self.model,
            right_robot_vector_specs or [],
            hand_side=right_hand_side,
            source_model=right_hand_model.model,
            prefix="right_",
        )
        self.left_distance_points = resolve_robot_distance_points(
            self.model,
            left_robot_distance_specs or [],
            hand_side=left_hand_side,
            source_model=left_hand_model.model,
            prefix="left_",
        )
        self.right_distance_points = resolve_robot_distance_points(
            self.model,
            right_robot_distance_specs or [],
            hand_side=right_hand_side,
            source_model=right_hand_model.model,
            prefix="right_",
        )
        self.left_frame_points = resolve_robot_frame_points(
            self.model,
            left_robot_frame_specs or [],
            hand_side=left_hand_side,
            source_model=left_hand_model.model,
            prefix="left_",
        )
        self.right_frame_points = resolve_robot_frame_points(
            self.model,
            right_robot_frame_specs or [],
            hand_side=right_hand_side,
            source_model=right_hand_model.model,
            prefix="right_",
        )
        self.left_angle_points = resolve_robot_angle_points(
            self.model,
            left_robot_angle_specs or [],
            hand_side=left_hand_side,
            source_model=left_hand_model.model,
            prefix="left_",
        )
        self.right_angle_points = resolve_robot_angle_points(
            self.model,
            right_robot_angle_specs or [],
            hand_side=right_hand_side,
            source_model=right_hand_model.model,
            prefix="right_",
        )
        self.left_variable_markers = resolve_variable_markers(self.model, prefix="left_") if self._diagnostic else []
        self.right_variable_markers = resolve_variable_markers(self.model, prefix="right_") if self._diagnostic else []

    def _build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec()
        spec.modelname = "somehand_bihand"
        spec.visual.global_.offwidth = max(
            int(self.left_hand_model.model.vis.global_.offwidth),
            int(self.right_hand_model.model.vis.global_.offwidth),
        )
        spec.visual.global_.offheight = max(
            int(self.left_hand_model.model.vis.global_.offheight),
            int(self.right_hand_model.model.vis.global_.offheight),
        )

        left_frame = spec.worldbody.add_frame()
        left_frame.pos = list(self.left_pos)
        left_frame.quat = list(self.left_quat)
        right_frame = spec.worldbody.add_frame()
        right_frame.pos = list(self.right_pos)
        right_frame.quat = list(self.right_quat)

        spec.attach(
            mujoco.MjSpec.from_file(self.left_hand_model.mjcf_path),
            frame=left_frame,
            prefix="left_",
        )
        spec.attach(
            mujoco.MjSpec.from_file(self.right_hand_model.mjcf_path),
            frame=right_frame,
            prefix="right_",
        )

        model = spec.compile()
        data = mujoco.MjData(model)
        return model, data

    def _resolve_qpos_indices(self, hand_model: HandModel, *, prefix: str) -> np.ndarray:
        qpos_indices: list[int] = []
        for joint_name in hand_model.get_joint_names():
            source_joint_id = mujoco.mj_name2id(hand_model.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_type = int(hand_model.model.jnt_type[source_joint_id])
            width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            combined_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}")
            combined_qpos_adr = int(self.model.jnt_qposadr[combined_joint_id])
            qpos_indices.extend(range(combined_qpos_adr, combined_qpos_adr + width))
        return np.array(qpos_indices, dtype=np.int32)

    def update(self, left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
        self.data.qpos[self.left_qpos_indices] = left_qpos
        self.data.qpos[self.right_qpos_indices] = right_qpos
        mujoco.mj_forward(self.model, self.data)


class BiHandVisualizer:
    """Real-time MuJoCo visualization of both retargeted robot hands."""

    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        key_callback=None,
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        camera_lookat: tuple[float, float, float] = (0.0, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
        viewer_mode: str = "normal",
        left_hand_side: str | None = None,
        right_hand_side: str | None = None,
        left_robot_vector_specs: list[RobotVectorSpec] | None = None,
        right_robot_vector_specs: list[RobotVectorSpec] | None = None,
        left_robot_distance_specs: list[RobotDistanceSpec] | None = None,
        right_robot_distance_specs: list[RobotDistanceSpec] | None = None,
        left_robot_frame_specs: list[RobotFrameSpec] | None = None,
        right_robot_frame_specs: list[RobotFrameSpec] | None = None,
        left_robot_angle_specs: list[RobotAngleSpec] | None = None,
        right_robot_angle_specs: list[RobotAngleSpec] | None = None,
    ):
        self.scene = BiHandScene(
            left_hand_model,
            right_hand_model,
            left_pos=left_pos,
            right_pos=right_pos,
            left_quat=left_quat,
            right_quat=right_quat,
            viewer_mode=viewer_mode,
            left_hand_side=left_hand_side,
            right_hand_side=right_hand_side,
            left_robot_vector_specs=left_robot_vector_specs,
            right_robot_vector_specs=right_robot_vector_specs,
            left_robot_distance_specs=left_robot_distance_specs,
            right_robot_distance_specs=right_robot_distance_specs,
            left_robot_frame_specs=left_robot_frame_specs,
            right_robot_frame_specs=right_robot_frame_specs,
            left_robot_angle_specs=left_robot_angle_specs,
            right_robot_angle_specs=right_robot_angle_specs,
        )
        self.model = self.scene.model
        self.data = self.scene.data
        self._camera_lookat = tuple(float(value) for value in camera_lookat)
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
        )
        self._configure_camera(
            distance=DEFAULT_BIHAND_CAMERA["distance"],
            azimuth=DEFAULT_BIHAND_CAMERA["azimuth"],
            elevation=DEFAULT_BIHAND_CAMERA["elevation"],
            lookat=self._camera_lookat,
        )
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(
        self,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        *,
        left_target_directions: np.ndarray | None = None,
        right_target_directions: np.ndarray | None = None,
        left_target_frame_primary_directions: np.ndarray | None = None,
        right_target_frame_primary_directions: np.ndarray | None = None,
        left_target_frame_secondary_directions: np.ndarray | None = None,
        right_target_frame_secondary_directions: np.ndarray | None = None,
        left_target_distances: np.ndarray | None = None,
        right_target_distances: np.ndarray | None = None,
        left_target_angles: np.ndarray | None = None,
        right_target_angles: np.ndarray | None = None,
    ) -> None:
        with self.viewer.lock():
            self.scene.update(left_qpos, right_qpos)
            if not self._camera_initialized and try_frame_hand_camera(
                self.viewer.cam,
                model=self.model,
                data=self.data,
                azimuth=DEFAULT_BIHAND_CAMERA["azimuth"],
                elevation=DEFAULT_BIHAND_CAMERA["elevation"],
            ):
                self._camera_initialized = True
            self._update_vector_overlay(
                left_target_directions,
                right_target_directions,
                left_target_frame_primary_directions=left_target_frame_primary_directions,
                right_target_frame_primary_directions=right_target_frame_primary_directions,
                left_target_frame_secondary_directions=left_target_frame_secondary_directions,
                right_target_frame_secondary_directions=right_target_frame_secondary_directions,
                left_target_distances=left_target_distances,
                right_target_distances=right_target_distances,
                left_target_angles=left_target_angles,
                right_target_angles=right_target_angles,
            )
        self.viewer.sync()

    def _update_vector_overlay(
        self,
        left_target_directions: np.ndarray | None,
        right_target_directions: np.ndarray | None,
        *,
        left_target_frame_primary_directions: np.ndarray | None = None,
        right_target_frame_primary_directions: np.ndarray | None = None,
        left_target_frame_secondary_directions: np.ndarray | None = None,
        right_target_frame_secondary_directions: np.ndarray | None = None,
        left_target_distances: np.ndarray | None = None,
        right_target_distances: np.ndarray | None = None,
        left_target_angles: np.ndarray | None = None,
        right_target_angles: np.ndarray | None = None,
    ) -> None:
        scene = self.viewer.user_scn
        if scene is None:
            return
        scene.ngeom = 0
        left_target_directions = _rotate_direction_targets(left_target_directions, self.scene.left_rotation)
        right_target_directions = _rotate_direction_targets(right_target_directions, self.scene.right_rotation)
        left_target_frame_primary_directions = _rotate_direction_targets(
            left_target_frame_primary_directions,
            self.scene.left_rotation,
        )
        right_target_frame_primary_directions = _rotate_direction_targets(
            right_target_frame_primary_directions,
            self.scene.right_rotation,
        )
        left_target_frame_secondary_directions = _rotate_direction_targets(
            left_target_frame_secondary_directions,
            self.scene.left_rotation,
        )
        right_target_frame_secondary_directions = _rotate_direction_targets(
            right_target_frame_secondary_directions,
            self.scene.right_rotation,
        )
        for vector_points, target_directions in (
            (self.scene.left_vector_points, left_target_directions),
            (self.scene.right_vector_points, right_target_directions),
        ):
            if not vector_points:
                continue
            starts, current_ends, target_indices = robot_vector_points(self.model, self.data, vector_points)
            append_vector_segments(scene, starts, current_ends, rgba=ROBOT_VECTOR_RGBA)
            target_starts, target_current_ends, selected_targets = select_target_vectors(
                starts,
                current_ends,
                target_directions,
                target_indices,
            )
            target_ends = target_direction_ends(
                target_starts,
                target_current_ends,
                selected_targets,
                max_length=TARGET_VECTOR_MAX_LENGTH,
            )
            if target_ends is not None:
                append_vector_segments(
                    scene,
                    target_starts[: len(target_ends)],
                    target_ends,
                    rgba=TARGET_VECTOR_RGBA,
                    radius=TARGET_VECTOR_RADIUS,
                )
        for points, target_distances in (
            (self.scene.left_distance_points, left_target_distances),
            (self.scene.right_distance_points, right_target_distances),
        ):
            if not points:
                continue
            starts, ends, target_indices = robot_distance_points(self.model, self.data, points)
            append_vector_segments(scene, starts, ends, rgba=DISTANCE_RGBA, radius=DIAGNOSTIC_THIN_RADIUS)
            target_ends = distance_target_ends(starts, ends, target_distances, target_indices)
            if target_ends is not None:
                append_vector_segments(
                    scene,
                    starts[: len(target_ends)],
                    target_ends,
                    rgba=DISTANCE_RGBA,
                    radius=TARGET_VECTOR_RADIUS,
                    tip_radius=ANGLE_MARKER_RADIUS * 0.6,
                )
        for points, primary_targets, secondary_targets in (
            (
                self.scene.left_frame_points,
                left_target_frame_primary_directions,
                left_target_frame_secondary_directions,
            ),
            (
                self.scene.right_frame_points,
                right_target_frame_primary_directions,
                right_target_frame_secondary_directions,
            ),
        ):
            append_frame_axes(
                scene,
                self.data,
                points,
                target_frame_primary_directions=primary_targets,
                target_frame_secondary_directions=secondary_targets,
            )
        for points, target_angles in (
            (self.scene.left_angle_points, left_target_angles),
            (self.scene.right_angle_points, right_target_angles),
        ):
            if not points:
                continue
            positions, colors = angle_marker_points(self.model, self.data, points, target_angles)
            append_variable_markers(scene, positions, colors, radius=ANGLE_MARKER_RADIUS)
        for markers in (self.scene.left_variable_markers, self.scene.right_variable_markers):
            if not markers:
                continue
            positions, colors = variable_marker_points(self.model, self.data, markers)
            append_variable_markers(scene, positions, colors)

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


def resolve_robot_vector_points(
    model,
    vector_specs: list[RobotVectorSpec],
    *,
    hand_side: str | None,
    source_model=None,
    prefix: str = "",
) -> list[ResolvedVectorPoint]:
    if not vector_specs:
        return []
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right' when robot vector specs are provided")
    source = model if source_model is None else source_model
    resolver = ModelNameResolver(source, hand_side=hand_side)
    resolved: list[ResolvedVectorPoint] = []
    for target_index, origin_name, origin_type, task_name, task_type in vector_specs:
        origin_id, origin_is_site = _resolve_vector_point(
            source,
            model,
            resolver,
            origin_name,
            origin_type,
            prefix=prefix,
        )
        task_id, task_is_site = _resolve_vector_point(
            source,
            model,
            resolver,
            task_name,
            task_type,
            prefix=prefix,
        )
        if origin_id == task_id and origin_is_site == task_is_site:
            continue
        resolved.append((origin_id, origin_is_site, task_id, task_is_site, int(target_index)))
    return resolved


def resolve_robot_distance_points(
    model,
    distance_specs: list[RobotDistanceSpec],
    *,
    hand_side: str | None,
    source_model=None,
    prefix: str = "",
) -> list[ResolvedDistancePoint]:
    if not distance_specs:
        return []
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right' when robot distance specs are provided")
    source = model if source_model is None else source_model
    resolver = ModelNameResolver(source, hand_side=hand_side)
    resolved: list[ResolvedDistancePoint] = []
    for target_index, first_name, first_type, second_name, second_type in distance_specs:
        first_id, first_is_site = _resolve_vector_point(
            source,
            model,
            resolver,
            first_name,
            first_type,
            prefix=prefix,
        )
        second_id, second_is_site = _resolve_vector_point(
            source,
            model,
            resolver,
            second_name,
            second_type,
            prefix=prefix,
        )
        if first_id == second_id and first_is_site == second_is_site:
            continue
        resolved.append((first_id, first_is_site, second_id, second_is_site, int(target_index)))
    return resolved


def resolve_robot_frame_points(
    model,
    frame_specs: list[RobotFrameSpec],
    *,
    hand_side: str | None,
    source_model=None,
    prefix: str = "",
) -> list[ResolvedFramePoint]:
    if not frame_specs:
        return []
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right' when robot frame specs are provided")
    source = model if source_model is None else source_model
    source_data = mujoco.MjData(source)
    mujoco.mj_forward(source, source_data)
    resolver = ModelNameResolver(source, hand_side=hand_side)
    resolved: list[ResolvedFramePoint] = []
    for target_index, origin_name, origin_type, primary_name, primary_type, secondary_name, secondary_type in frame_specs:
        source_origin_id, source_origin_is_site = _resolve_vector_point(
            source,
            source,
            resolver,
            origin_name,
            origin_type,
            prefix="",
        )
        source_primary_id, source_primary_is_site = _resolve_vector_point(
            source,
            source,
            resolver,
            primary_name,
            primary_type,
            prefix="",
        )
        source_secondary_id, source_secondary_is_site = _resolve_vector_point(
            source,
            source,
            resolver,
            secondary_name,
            secondary_type,
            prefix="",
        )
        origin_position = _data_point_pos(source_data, source_origin_id, source_origin_is_site)
        primary_position = _data_point_pos(source_data, source_primary_id, source_primary_is_site)
        secondary_position = _data_point_pos(source_data, source_secondary_id, source_secondary_is_site)
        origin_rotation = _data_point_rot(source_data, source_origin_id, source_origin_is_site)
        primary_axis, secondary_axis = _orthonormalize_axes(
            origin_rotation.T @ (primary_position - origin_position),
            origin_rotation.T @ (secondary_position - origin_position),
        )
        if primary_axis is None or secondary_axis is None:
            continue
        origin_id, origin_is_site = _resolve_vector_point(
            source,
            model,
            resolver,
            origin_name,
            origin_type,
            prefix=prefix,
        )
        resolved.append((origin_id, origin_is_site, primary_axis, secondary_axis, int(target_index)))
    return resolved


def resolve_robot_angle_points(
    model,
    angle_specs: list[RobotAngleSpec],
    *,
    hand_side: str | None,
    source_model=None,
    prefix: str = "",
) -> list[ResolvedAnglePoint]:
    if not angle_specs:
        return []
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right' when robot angle specs are provided")
    source = model if source_model is None else source_model
    resolver = ModelNameResolver(source, hand_side=hand_side)
    resolved: list[ResolvedAnglePoint] = []
    for target_index, joint_name in angle_specs:
        resolved_name = resolver.resolve(joint_name, obj_type=mujoco.mjtObj.mjOBJ_JOINT, role="Angle visualization")
        target_name = f"{prefix}{resolved_name}" if prefix else resolved_name
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_name)
        if joint_id < 0 and source is model:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, resolved_name)
        if joint_id < 0:
            raise ValueError(f"Angle visualization joint '{target_name}' not found in model")
        low, high = model.jnt_range[joint_id]
        resolved.append((int(joint_id), int(model.jnt_qposadr[joint_id]), int(target_index), float(low), float(high)))
    return resolved


def _resolve_vector_point(
    source_model,
    target_model,
    resolver: ModelNameResolver,
    name: str,
    point_type: str,
    *,
    prefix: str,
) -> tuple[int, bool]:
    is_site = point_type == "site"
    obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
    resolved_name = resolver.resolve(name, obj_type=obj_type, role="Vector visualization")
    target_name = f"{prefix}{resolved_name}" if prefix else resolved_name
    point_id = mujoco.mj_name2id(target_model, obj_type, target_name)
    if point_id < 0 and source_model is target_model:
        point_id = mujoco.mj_name2id(target_model, obj_type, resolved_name)
    if point_id < 0:
        raise ValueError(f"Vector visualization point '{target_name}' not found in model")
    return int(point_id), is_site


def _data_point_pos(data, point_id: int, is_site: bool) -> np.ndarray:
    return np.array(data.site_xpos[point_id] if is_site else data.xpos[point_id], dtype=np.float64)


def _data_point_rot(data, point_id: int, is_site: bool) -> np.ndarray:
    if is_site:
        return data.site_xmat[point_id].reshape(3, 3).copy()
    return data.xmat[point_id].reshape(3, 3).copy()


def _orthonormalize_axes(primary_vector: np.ndarray, secondary_vector: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    primary_norm = np.linalg.norm(primary_vector)
    if primary_norm < 1e-8:
        return None, None
    primary_axis = primary_vector / primary_norm
    secondary_rejected = secondary_vector - np.dot(secondary_vector, primary_axis) * primary_axis
    secondary_norm = np.linalg.norm(secondary_rejected)
    if secondary_norm < 1e-8:
        return primary_axis, None
    return primary_axis, secondary_rejected / secondary_norm


def robot_vector_points(
    model,
    data,
    vector_points: list[ResolvedVectorPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.empty((len(vector_points), 3), dtype=np.float64)
    ends = np.empty((len(vector_points), 3), dtype=np.float64)
    target_indices = np.empty(len(vector_points), dtype=np.int32)
    for index, (origin_id, origin_is_site, task_id, task_is_site, target_index) in enumerate(vector_points):
        starts[index] = data.site_xpos[origin_id] if origin_is_site else data.xpos[origin_id]
        ends[index] = data.site_xpos[task_id] if task_is_site else data.xpos[task_id]
        target_indices[index] = target_index
    return starts, ends, target_indices


def robot_distance_points(
    model,
    data,
    distance_points: list[ResolvedDistancePoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.empty((len(distance_points), 3), dtype=np.float64)
    ends = np.empty((len(distance_points), 3), dtype=np.float64)
    target_indices = np.empty(len(distance_points), dtype=np.int32)
    for index, (first_id, first_is_site, second_id, second_is_site, target_index) in enumerate(distance_points):
        starts[index] = data.site_xpos[first_id] if first_is_site else data.xpos[first_id]
        ends[index] = data.site_xpos[second_id] if second_is_site else data.xpos[second_id]
        target_indices[index] = target_index
    return starts, ends, target_indices


def distance_target_ends(
    starts: np.ndarray,
    current_ends: np.ndarray,
    target_distances: np.ndarray | None,
    target_indices: np.ndarray,
) -> np.ndarray | None:
    if target_distances is None:
        return None
    distances = np.asarray(target_distances, dtype=np.float64)
    valid_mask = target_indices < len(distances)
    if not np.any(valid_mask):
        return None
    origins = starts[valid_mask]
    current = current_ends[valid_mask]
    valid_indices = target_indices[valid_mask]
    ends = np.empty((len(origins), 3), dtype=np.float64)
    for index, target_index in enumerate(valid_indices):
        vector = current[index] - origins[index]
        norm = np.linalg.norm(vector)
        if norm < 1e-7:
            ends[index] = origins[index]
        else:
            ends[index] = origins[index] + vector / norm * float(distances[target_index])
    return ends


def append_frame_axes(
    scene,
    data,
    frame_points: list[ResolvedFramePoint],
    *,
    target_frame_primary_directions: np.ndarray | None = None,
    target_frame_secondary_directions: np.ndarray | None = None,
    axis_length: float = 0.035,
) -> None:
    for origin_id, origin_is_site, local_primary, local_secondary, target_index in frame_points:
        origin = data.site_xpos[origin_id] if origin_is_site else data.xpos[origin_id]
        rotation = data.site_xmat[origin_id].reshape(3, 3) if origin_is_site else data.xmat[origin_id].reshape(3, 3)
        primary = rotation @ local_primary
        secondary = rotation @ local_secondary
        normal = np.cross(primary, secondary)
        append_vector_segments(
            scene,
            np.asarray([origin, origin, origin], dtype=np.float64),
            np.asarray(
                [
                    origin + primary * axis_length,
                    origin + secondary * axis_length,
                    origin + normal * axis_length,
                ],
                dtype=np.float64,
            ),
            rgba=FRAME_PRIMARY_RGBA,
            radius=DIAGNOSTIC_THIN_RADIUS,
        )
        scene.geoms[scene.ngeom - 4].rgba[:] = FRAME_SECONDARY_RGBA
        scene.geoms[scene.ngeom - 3].rgba[:] = FRAME_SECONDARY_RGBA
        scene.geoms[scene.ngeom - 2].rgba[:] = FRAME_NORMAL_RGBA
        scene.geoms[scene.ngeom - 1].rgba[:] = FRAME_NORMAL_RGBA
        target_primary = _select_target_axis(target_frame_primary_directions, target_index)
        target_secondary = _select_target_axis(target_frame_secondary_directions, target_index)
        if target_primary is not None:
            append_vector_segments(
                scene,
                np.asarray([origin], dtype=np.float64),
                np.asarray([origin + target_primary * axis_length], dtype=np.float64),
                rgba=TARGET_VECTOR_RGBA,
                radius=TARGET_VECTOR_RADIUS,
            )
        if target_secondary is not None:
            append_vector_segments(
                scene,
                np.asarray([origin], dtype=np.float64),
                np.asarray([origin + target_secondary * axis_length], dtype=np.float64),
                rgba=TARGET_VECTOR_RGBA,
                radius=TARGET_VECTOR_RADIUS,
            )


def _select_target_axis(target_axes: np.ndarray | None, target_index: int) -> np.ndarray | None:
    if target_axes is None or target_index >= len(target_axes):
        return None
    axis = np.asarray(target_axes[target_index], dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-7:
        return None
    return axis / norm


def select_target_vectors(
    starts: np.ndarray,
    current_ends: np.ndarray,
    target_directions: np.ndarray | None,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if target_directions is None:
        return starts[:0], current_ends[:0], None
    directions = np.asarray(target_directions, dtype=np.float64)
    valid_mask = target_indices < len(directions)
    if not np.any(valid_mask):
        return starts[:0], current_ends[:0], None
    valid_indices = target_indices[valid_mask]
    return starts[valid_mask], current_ends[valid_mask], directions[valid_indices]


def select_target_directions(target_directions: np.ndarray | None, target_indices: np.ndarray) -> np.ndarray | None:
    if target_directions is None:
        return None
    directions = np.asarray(target_directions, dtype=np.float64)
    valid = target_indices[target_indices < len(directions)]
    if len(valid) == 0:
        return None
    return directions[valid]


def resolve_variable_markers(model, *, prefix: str = "") -> list[VariableMarkerSpec]:
    markers: list[VariableMarkerSpec] = []
    scalar_joint_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if prefix and (joint_name is None or not joint_name.startswith(prefix)):
            continue
        if int(model.jnt_type[joint_id]) not in scalar_joint_types:
            continue
        if hasattr(model, "jnt_limited") and not bool(model.jnt_limited[joint_id]):
            continue
        low, high = model.jnt_range[joint_id]
        if not (np.isfinite(low) and np.isfinite(high) and high > low):
            continue
        markers.append((int(joint_id), int(model.jnt_qposadr[joint_id]), float(low), float(high)))
    return markers


def variable_marker_points(model, data, markers: list[VariableMarkerSpec]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(markers), 3), dtype=np.float64)
    colors = np.empty((len(markers), 4), dtype=np.float32)
    for index, (joint_id, qpos_id, low, high) in enumerate(markers):
        positions[index] = data.xanchor[joint_id]
        colors[index] = variable_marker_rgba(float(data.qpos[qpos_id]), low, high)
    return positions, colors


def angle_marker_points(
    model,
    data,
    markers: list[ResolvedAnglePoint],
    target_angles: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(markers), 3), dtype=np.float64)
    colors = np.empty((len(markers), 4), dtype=np.float32)
    targets = None if target_angles is None else np.asarray(target_angles, dtype=np.float64)
    for index, (joint_id, qpos_id, target_index, low, high) in enumerate(markers):
        positions[index] = data.xanchor[joint_id]
        if targets is not None and target_index < len(targets) and high > low:
            error = abs(float(data.qpos[qpos_id]) - float(targets[target_index])) / (high - low)
            colors[index] = variable_marker_rgba(error, 0.0, 1.0)
        else:
            colors[index] = ANGLE_RGBA
    return positions, colors


def apply_model_alpha(model, alpha: float) -> None:
    alpha = float(alpha)
    if getattr(model, "ngeom", 0):
        model.geom_rgba[:, 3] = alpha
    if getattr(model, "nmat", 0):
        model.mat_rgba[:, 3] = alpha


def set_fingertip_site_visibility(model, *, visible: bool) -> None:
    if not getattr(model, "nsite", 0):
        return
    alpha = 1.0 if visible else 0.0
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not site_name or not site_name.endswith("_tip"):
            continue
        model.site_rgba[site_id] = FINGERTIP_SITE_RGBA
        model.site_rgba[site_id, 3] = alpha


__all__ = [
    "FINGERTIP_SITE_RGBA",
    "HandVisualizer",
    "BiHandScene",
    "BiHandVisualizer",
    "angle_marker_points",
    "apply_model_alpha",
    "append_frame_axes",
    "distance_target_ends",
    "resolve_robot_angle_points",
    "resolve_robot_distance_points",
    "resolve_robot_frame_points",
    "resolve_robot_vector_points",
    "resolve_variable_markers",
    "robot_vector_points",
    "select_target_directions",
    "select_target_vectors",
    "set_fingertip_site_visibility",
    "variable_marker_points",
]
