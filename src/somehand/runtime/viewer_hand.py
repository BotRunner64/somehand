"""Hand and bi-hand MuJoCo viewer implementations."""

from __future__ import annotations

import mujoco
import numpy as np

from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.model_name_resolver import ModelNameResolver

from .viewer_camera import DEFAULT_BIHAND_CAMERA, DEFAULT_HAND_CAMERA, configure_free_camera, try_frame_hand_camera
from .viewer_passive import ManagedPassiveViewer, compile_model_with_name, mujoco_key_callback, set_viewer_overlay_label, set_viewer_window_title
from .vector_visualization import (
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
VariableMarkerSpec = tuple[int, int, float, float]
DIAGNOSTIC_ALPHA = 0.28
TARGET_VECTOR_MAX_LENGTH = 0.035


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

    def update(self, qpos: np.ndarray, target_directions: np.ndarray | None = None):
        with self.viewer.lock():
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and try_frame_hand_camera(self.viewer.cam, model=self.model, data=self.data):
                self._camera_initialized = True
            self._update_vector_overlay(target_directions)
        set_viewer_overlay_label(self.viewer, self._overlay_label)
        self.viewer.sync()

    def _update_vector_overlay(self, target_directions: np.ndarray | None) -> None:
        scene = self.viewer.user_scn
        if scene is None:
            return
        scene.ngeom = 0
        if self._vector_points:
            starts, current_ends, target_indices = robot_vector_points(self.model, self.data, self._vector_points)
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
        if self._variable_markers:
            positions, colors = variable_marker_points(self.model, self.data, self._variable_markers)
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
    ):
        self.left_hand_model = left_hand_model
        self.right_hand_model = right_hand_model
        self.left_pos = tuple(float(value) for value in left_pos)
        self.right_pos = tuple(float(value) for value in right_pos)
        self.left_quat = tuple(float(value) for value in left_quat)
        self.right_quat = tuple(float(value) for value in right_quat)
        self._diagnostic = viewer_mode == "diagnostic"
        self.model, self.data = self._build_model()
        if self._diagnostic:
            apply_model_alpha(self.model, DIAGNOSTIC_ALPHA)
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
            self._update_vector_overlay(left_target_directions, right_target_directions)
        self.viewer.sync()

    def _update_vector_overlay(
        self,
        left_target_directions: np.ndarray | None,
        right_target_directions: np.ndarray | None,
    ) -> None:
        scene = self.viewer.user_scn
        if scene is None:
            return
        scene.ngeom = 0
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


def apply_model_alpha(model, alpha: float) -> None:
    alpha = float(alpha)
    if getattr(model, "ngeom", 0):
        model.geom_rgba[:, 3] = alpha
    if getattr(model, "nmat", 0):
        model.mat_rgba[:, 3] = alpha


__all__ = [
    "HandVisualizer",
    "BiHandScene",
    "BiHandVisualizer",
    "apply_model_alpha",
    "resolve_robot_vector_points",
    "resolve_variable_markers",
    "robot_vector_points",
    "select_target_directions",
    "select_target_vectors",
    "variable_marker_points",
]
