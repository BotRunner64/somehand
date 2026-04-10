"""MuJoCo passive viewers for robot-hand and input-landmark visualization."""

from __future__ import annotations

import multiprocessing as mp
import atexit
import queue
import signal
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from .hand_model import HandModel

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
_LANDMARK_COLORS = np.array(
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
_BONE_COLORS = np.array([_LANDMARK_COLORS[end] for _, end in _HAND_CONNECTIONS], dtype=np.float32)
_LEFT_LANDMARK_COLORS = np.array(
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
_RIGHT_LANDMARK_COLORS = np.array(
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
_LEFT_BONE_COLORS = np.array([_LEFT_LANDMARK_COLORS[end] for _, end in _HAND_CONNECTIONS], dtype=np.float32)
_RIGHT_BONE_COLORS = np.array([_RIGHT_LANDMARK_COLORS[end] for _, end in _HAND_CONNECTIONS], dtype=np.float32)
_IDENTITY_MAT = np.eye(3, dtype=np.float64).reshape(-1)
_POINT_RADIUS = 0.006
_BONE_RADIUS = 0.0035
_CAMERA_MARGIN = 1.15
_MIN_CAMERA_DISTANCE = 0.18
_MIN_FRAMING_RADIUS = 0.01
_DEFAULT_HAND_CAMERA = {
    "distance": 0.55,
    "azimuth": 145.0,
    "elevation": -20.0,
    "lookat": (0.0, 0.0, 0.0),
}
_DEFAULT_BIHAND_CAMERA = {
    "distance": 0.60,
    "azimuth": -90.0,
    "elevation": -5.0,
}
_DEFAULT_BIHAND_LANDMARK_CAMERA = {
    "distance": 0.60,
    "azimuth": -90.0,
    "elevation": -5.0,
    "lookat": (0.0, 0.04, 0.02),
}
_DEFAULT_LANDMARK_CAMERA = {
    "distance": 0.32,
    "azimuth": 140.0,
    "elevation": -24.0,
    "lookat": (0.0, 0.0, 0.02),
}
_LANDMARK_VIEWER_XML = """
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


def _mujoco_key_callback(handler):
    if handler is None:
        return None

    def _callback(keycode: int) -> None:
        if keycode < 0 or keycode > 255:
            return
        handler(chr(keycode))

    return _callback


def _set_viewer_overlay_label(viewer, label: str | None) -> None:
    if not label:
        return
    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            label,
            "",
        )
    )


def _set_viewer_window_title(viewer, title: str | None) -> None:
    if not title:
        return
    get_sim = getattr(viewer, "_get_sim", None)
    if not callable(get_sim):
        return
    sim = get_sim()
    if sim is None:
        return
    try:
        sim.filename = title
    except Exception:
        return


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
    configure_free_camera(camera, **_DEFAULT_HAND_CAMERA)


def _camera_aspect_ratio(model) -> float:
    width = max(int(model.vis.global_.offwidth), 1)
    height = max(int(model.vis.global_.offheight), 1)
    return width / height


def _camera_distance_for_radius(radius: float, *, fovy_degrees: float, aspect_ratio: float) -> float:
    safe_radius = max(float(radius), _MIN_FRAMING_RADIUS)
    half_vertical = np.deg2rad(max(float(fovy_degrees), 1.0) * 0.5)
    half_horizontal = np.arctan(np.tan(half_vertical) * max(float(aspect_ratio), 1e-3))
    limiting_half_angle = max(min(half_vertical, half_horizontal), np.deg2rad(5.0))
    return max(_MIN_CAMERA_DISTANCE, _CAMERA_MARGIN * safe_radius / np.sin(limiting_half_angle))


def _compute_bounding_sphere(
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


def _try_frame_camera_to_points(
    camera,
    *,
    model,
    points: np.ndarray,
    radii: np.ndarray | None = None,
    azimuth: float,
    elevation: float,
    aspect_ratio: float | None = None,
) -> bool:
    lookat, radius = _compute_bounding_sphere(points, radii=radii)
    if radius <= 0.0:
        return False

    configure_free_camera(
        camera,
        distance=_camera_distance_for_radius(
            radius,
            fovy_degrees=float(model.vis.global_.fovy),
            aspect_ratio=_camera_aspect_ratio(model) if aspect_ratio is None else float(aspect_ratio),
        ),
        azimuth=azimuth,
        elevation=elevation,
        lookat=tuple(lookat),
    )
    return True


def _try_frame_hand_camera(
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

    return _try_frame_camera_to_points(
        camera,
        model=model,
        points=np.asarray(centers, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        azimuth=_DEFAULT_HAND_CAMERA["azimuth"] if azimuth is None else float(azimuth),
        elevation=_DEFAULT_HAND_CAMERA["elevation"] if elevation is None else float(elevation),
        aspect_ratio=aspect_ratio,
    )


def _launch_passive_internal_with_window_title(
    model,
    data,
    *,
    handle_return,
    key_callback=None,
    show_left_ui=False,
    show_right_ui=False,
    window_title: str | None = None,
) -> None:
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    user_scn = mujoco.MjvScene(model, mujoco.viewer._Simulate.MAX_GEOM)
    simulate = mujoco.viewer._Simulate(cam, opt, pert, user_scn, False, key_callback)

    simulate.ui0_enable = show_left_ui
    simulate.ui1_enable = show_right_ui

    if mujoco.viewer._MJPYTHON is None:
        if not mujoco.viewer.glfw.init():
            raise mujoco.FatalError("could not initialize GLFW")
        atexit.register(mujoco.viewer.glfw.terminate)

    def _loader():
        return model, data, window_title or ""

    notify_loaded = lambda: handle_return.put_nowait(mujoco.viewer.Handle(simulate, cam, opt, pert, user_scn))
    side_thread = threading.Thread(target=mujoco.viewer._reload, args=(simulate, _loader, notify_loaded))

    def _exit_simulate():
        simulate.exit()

    atexit.register(_exit_simulate)
    side_thread.start()
    simulate.render_loop()
    atexit.unregister(_exit_simulate)
    side_thread.join()
    simulate.destroy()


def _compile_model_with_name(mjcf_path: str, model_name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(mjcf_path)
    spec.modelname = model_name
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


class _ManagedPassiveViewer:
    """Wrap MuJoCo's passive viewer and wait for its render thread to exit."""

    def __init__(
        self,
        model,
        data,
        *,
        key_callback=None,
        show_left_ui=False,
        show_right_ui=False,
        window_title: str | None = None,
    ):
        if sys.platform == "darwin":
            self._handle = mujoco.viewer.launch_passive(
                model=model,
                data=data,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )
            self._thread = None
            return

        handle_return = queue.Queue(1)
        if not window_title:
            target = mujoco.viewer._launch_internal
            args = (model, data)
            kwargs = dict(
                run_physics_thread=False,
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )
        else:
            target = _launch_passive_internal_with_window_title
            args = (model, data)
            kwargs = dict(
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
                window_title=window_title,
            )
        self._thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs,
            name="dex-mujoco-passive-viewer",
            daemon=True,
        )
        self._thread.start()
        self._handle = handle_return.get()

    def __getattr__(self, name: str):
        return getattr(self._handle, name)

    def close(self, *, timeout: float = 2.0) -> None:
        self._handle.close()
        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)


class HandVisualizer:
    """Real-time MuJoCo visualization of the retargeted robot hand."""

    def __init__(
        self,
        hand_model: HandModel,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
    ):
        self.hand_model = hand_model
        if window_title:
            self.model, self.data = _compile_model_with_name(hand_model.mjcf_path, window_title)
        else:
            self.model = hand_model.model
            self.data = hand_model.data
        self._overlay_label = overlay_label
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=_mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        _set_viewer_window_title(self.viewer, window_title)
        _set_viewer_overlay_label(self.viewer, self._overlay_label)
        self._configure_camera(**_DEFAULT_HAND_CAMERA)
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

    def update(self, qpos: np.ndarray):
        """Update visualization with new robot joint positions."""
        with self.viewer.lock():
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and _try_frame_hand_camera(self.viewer.cam, model=self.model, data=self.data):
                self._camera_initialized = True
        _set_viewer_overlay_label(self.viewer, self._overlay_label)
        self.viewer.sync()

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
    ):
        self.left_hand_model = left_hand_model
        self.right_hand_model = right_hand_model
        self.left_pos = tuple(float(value) for value in left_pos)
        self.right_pos = tuple(float(value) for value in right_pos)
        self.left_quat = tuple(float(value) for value in left_quat)
        self.right_quat = tuple(float(value) for value in right_quat)
        self.model, self.data = self._build_model()
        self.left_qpos_indices = self._resolve_qpos_indices(left_hand_model, prefix="left_")
        self.right_qpos_indices = self._resolve_qpos_indices(right_hand_model, prefix="right_")

    def _build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec()
        spec.modelname = "dex_mujoco_bihand"
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
    ):
        self.scene = BiHandScene(
            left_hand_model,
            right_hand_model,
            left_pos=left_pos,
            right_pos=right_pos,
            left_quat=left_quat,
            right_quat=right_quat,
        )
        self.model = self.scene.model
        self.data = self.scene.data
        self._camera_lookat = tuple(float(value) for value in camera_lookat)
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=_mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
        )
        self._configure_camera(
            distance=_DEFAULT_BIHAND_CAMERA["distance"],
            azimuth=_DEFAULT_BIHAND_CAMERA["azimuth"],
            elevation=_DEFAULT_BIHAND_CAMERA["elevation"],
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

    def update(self, left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
        with self.viewer.lock():
            self.scene.update(left_qpos, right_qpos)
            if not self._camera_initialized and _try_frame_hand_camera(
                self.viewer.cam,
                model=self.model,
                data=self.data,
                azimuth=_DEFAULT_BIHAND_CAMERA["azimuth"],
                elevation=_DEFAULT_BIHAND_CAMERA["elevation"],
            ):
                self._camera_initialized = True
        self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class LandmarkVisualizer:
    """Real-time MuJoCo visualization of the input hand landmarks."""

    def __init__(self, *, window_title: str | None = None):
        self.model = mujoco.MjModel.from_xml_string(_LANDMARK_VIEWER_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        _set_viewer_window_title(self.viewer, window_title)
        self._max_overlay_geoms = len(_LANDMARK_COLORS) + len(_HAND_CONNECTIONS)
        if self.viewer.user_scn is None:
            raise RuntimeError("MuJoCo passive viewer does not expose a user scene")
        if self.viewer.user_scn.maxgeom < self._max_overlay_geoms:
            raise RuntimeError(
                f"MuJoCo viewer user scene only supports {self.viewer.user_scn.maxgeom} geoms, "
                f"but landmark overlay needs {self._max_overlay_geoms}"
            )
        self._configure_camera(**_DEFAULT_LANDMARK_CAMERA)
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

    def update(self, landmarks: np.ndarray):
        """Update visualization with input-hand landmarks."""
        with self.viewer.lock():
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and _try_frame_camera_to_points(
                self.viewer.cam,
                model=self.model,
                points=landmarks,
                azimuth=_DEFAULT_LANDMARK_CAMERA["azimuth"],
                elevation=_DEFAULT_LANDMARK_CAMERA["elevation"],
            ):
                self._camera_initialized = True
            self._update_landmark_overlay(landmarks)
        self.viewer.sync()

    def _update_landmark_overlay(self, landmarks: np.ndarray) -> None:
        scene = self.viewer.user_scn
        scene.ngeom = 0

        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (21, 3):
            raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")

        for point, rgba in zip(points, _LANDMARK_COLORS, strict=True):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, _POINT_RADIUS, dtype=np.float64),
                point,
                _IDENTITY_MAT,
                rgba,
            )
            scene.ngeom += 1

        for (start_idx, end_idx), rgba in zip(_HAND_CONNECTIONS, _BONE_COLORS, strict=True):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                _IDENTITY_MAT,
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                _BONE_RADIUS,
                points[start_idx],
                points[end_idx],
            )
            geom.rgba[:] = rgba
            scene.ngeom += 1

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class BiHandLandmarkVisualizer:
    """Real-time MuJoCo visualization of both input-hand landmark sets."""

    def __init__(self, *, window_title: str | None = None):
        self.model = mujoco.MjModel.from_xml_string(_LANDMARK_VIEWER_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        _set_viewer_window_title(self.viewer, window_title)
        self._max_overlay_geoms = 2 * (len(_LANDMARK_COLORS) + len(_HAND_CONNECTIONS))
        if self.viewer.user_scn is None:
            raise RuntimeError("MuJoCo passive viewer does not expose a user scene")
        if self.viewer.user_scn.maxgeom < self._max_overlay_geoms:
            raise RuntimeError(
                f"MuJoCo viewer user scene only supports {self.viewer.user_scn.maxgeom} geoms, "
                f"but bi-hand landmark overlay needs {self._max_overlay_geoms}"
            )
        self._configure_camera(**_DEFAULT_BIHAND_LANDMARK_CAMERA)
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

    def update(self, hands: np.ndarray):
        with self.viewer.lock():
            mujoco.mj_forward(self.model, self.data)
            finite_points = hands[np.isfinite(hands).all(axis=2)]
            if finite_points.size > 0 and not self._camera_initialized and _try_frame_camera_to_points(
                self.viewer.cam,
                model=self.model,
                points=finite_points.reshape(-1, 3),
                azimuth=_DEFAULT_BIHAND_LANDMARK_CAMERA["azimuth"],
                elevation=_DEFAULT_BIHAND_LANDMARK_CAMERA["elevation"],
            ):
                self._camera_initialized = True
            self._update_landmark_overlay(hands)
        self.viewer.sync()

    def _update_landmark_overlay(self, hands: np.ndarray) -> None:
        scene = self.viewer.user_scn
        scene.ngeom = 0
        points = np.asarray(hands, dtype=np.float64)
        if points.shape != (2, 21, 3):
            raise ValueError(f"Expected landmarks with shape (2, 21, 3), got {points.shape}")

        for hand_points, point_colors, bone_colors in (
            (points[0], _LEFT_LANDMARK_COLORS, _LEFT_BONE_COLORS),
            (points[1], _RIGHT_LANDMARK_COLORS, _RIGHT_BONE_COLORS),
        ):
            finite_mask = np.isfinite(hand_points).all(axis=1)
            for idx, (point, rgba) in enumerate(zip(hand_points, point_colors, strict=True)):
                if not finite_mask[idx]:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, _POINT_RADIUS, dtype=np.float64),
                    point,
                    _IDENTITY_MAT,
                    rgba,
                )
                scene.ngeom += 1

            for (start_idx, end_idx), rgba in zip(_HAND_CONNECTIONS, bone_colors, strict=True):
                if not (finite_mask[start_idx] and finite_mask[end_idx]):
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    _IDENTITY_MAT,
                    rgba,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    _BONE_RADIUS,
                    hand_points[start_idx],
                    hand_points[end_idx],
                )
                geom.rgba[:] = rgba
                scene.ngeom += 1

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


def _landmark_viewer_worker(frame_queue: mp.queues.Queue, window_title: str | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    visualizer = LandmarkVisualizer(window_title=window_title)
    latest_landmarks = np.zeros((21, 3), dtype=np.float64)

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                latest_landmarks = np.asarray(item, dtype=np.float64)
                drained = True

            if drained:
                visualizer.update(latest_landmarks)
            else:
                visualizer.update(latest_landmarks)
                time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncLandmarkVisualizer:
    """Landmark viewer running in a separate process for stability."""

    def __init__(self, *, window_title: str | None = None):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_landmark_viewer_worker,
            args=(self._queue, window_title),
            name="dex-mujoco-landmark-viewer",
        )
        self._process.start()

    @property
    def is_running(self) -> bool:
        return self._process.is_alive()

    def update(self, landmarks: np.ndarray) -> None:
        payload = np.asarray(landmarks, dtype=np.float64)
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


def _robot_hand_viewer_worker(
    mjcf_path: str,
    qpos_queue: mp.queues.Queue,
    overlay_label: str | None,
    window_title: str | None,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    hand_model = HandModel(mjcf_path)
    visualizer = HandVisualizer(hand_model, overlay_label=overlay_label, window_title=window_title)
    latest_qpos = hand_model.get_qpos()

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = qpos_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                latest_qpos = np.asarray(item, dtype=np.float64)
                drained = True

            if drained:
                visualizer.update(latest_qpos)
            else:
                visualizer.update(latest_qpos)
                time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncRobotHandVisualizer:
    """Robot-hand viewer running in a separate process for stability."""

    def __init__(
        self,
        mjcf_path: str,
        *,
        overlay_label: str | None = None,
        window_title: str | None = None,
    ):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_robot_hand_viewer_worker,
            args=(mjcf_path, self._queue, overlay_label, window_title),
            name="dex-mujoco-robot-hand-viewer",
        )
        self._process.start()

    @property
    def is_running(self) -> bool:
        return self._process.is_alive()

    def update(self, qpos: np.ndarray) -> None:
        payload = np.asarray(qpos, dtype=np.float64)
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


def _bihand_landmark_viewer_worker(frame_queue: mp.queues.Queue) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    visualizer = BiHandLandmarkVisualizer()
    latest_landmarks = np.full((2, 21, 3), np.nan, dtype=np.float64)

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                latest_landmarks = np.asarray(item, dtype=np.float64)
                drained = True

            if drained:
                visualizer.update(latest_landmarks)
            else:
                visualizer.update(latest_landmarks)
                time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncBiHandLandmarkVisualizer:
    """Bi-hand landmark viewer running in a separate process for stability."""

    def __init__(self):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_bihand_landmark_viewer_worker,
            args=(self._queue,),
            name="dex-mujoco-bihand-landmark-viewer",
        )
        self._process.start()

    @property
    def is_running(self) -> bool:
        return self._process.is_alive()

    def update(self, landmarks: np.ndarray) -> None:
        payload = np.asarray(landmarks, dtype=np.float64)
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
