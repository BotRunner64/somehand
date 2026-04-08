"""MuJoCo passive viewers for robot-hand and input-landmark visualization."""

from __future__ import annotations

import multiprocessing as mp
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
_IDENTITY_MAT = np.eye(3, dtype=np.float64).reshape(-1)
_POINT_RADIUS = 0.006
_BONE_RADIUS = 0.0035
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


class _ManagedPassiveViewer:
    """Wrap MuJoCo's passive viewer and wait for its render thread to exit."""

    def __init__(self, model, data, *, key_callback=None, show_left_ui=False, show_right_ui=False):
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
        self._thread = threading.Thread(
            target=mujoco.viewer._launch_internal,
            args=(model, data),
            kwargs=dict(
                run_physics_thread=False,
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            ),
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

    def __init__(self, hand_model: HandModel, *, key_callback=None):
        self.hand_model = hand_model
        self.model = hand_model.model
        self.data = hand_model.data
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            key_callback=_mujoco_key_callback(key_callback),
            show_left_ui=False,
            show_right_ui=False,
        )
        self._configure_camera(distance=0.55, azimuth=145.0, elevation=-20.0, lookat=(0.0, 0.0, 0.0))

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.distance = distance
            self.viewer.cam.azimuth = azimuth
            self.viewer.cam.elevation = elevation
            self.viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
        self.viewer.sync(state_only=True)

    def update(self, qpos: np.ndarray):
        """Update visualization with new robot joint positions."""
        with self.viewer.lock():
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class LandmarkVisualizer:
    """Real-time MuJoCo visualization of the input hand landmarks."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_LANDMARK_VIEWER_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = _ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._max_overlay_geoms = len(_LANDMARK_COLORS) + len(_HAND_CONNECTIONS)
        if self.viewer.user_scn is None:
            raise RuntimeError("MuJoCo passive viewer does not expose a user scene")
        if self.viewer.user_scn.maxgeom < self._max_overlay_geoms:
            raise RuntimeError(
                f"MuJoCo viewer user scene only supports {self.viewer.user_scn.maxgeom} geoms, "
                f"but landmark overlay needs {self._max_overlay_geoms}"
            )
        self._configure_camera(distance=0.32, azimuth=140.0, elevation=-24.0, lookat=(0.0, 0.0, 0.02))

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.distance = distance
            self.viewer.cam.azimuth = azimuth
            self.viewer.cam.elevation = elevation
            self.viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
        self.viewer.sync(state_only=True)

    def update(self, landmarks: np.ndarray):
        """Update visualization with input-hand landmarks."""
        with self.viewer.lock():
            mujoco.mj_forward(self.model, self.data)
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


def _landmark_viewer_worker(frame_queue: mp.queues.Queue) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    visualizer = LandmarkVisualizer()
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

    def __init__(self):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_landmark_viewer_worker,
            args=(self._queue,),
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
