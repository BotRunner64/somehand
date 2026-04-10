"""Runtime sinks for robot visualization and trajectory recording."""

from __future__ import annotations

import importlib
import os
import platform
from pathlib import Path

import cv2
import mujoco
import numpy as np

from dex_mujoco.domain import (
    BiHandFrame,
    BiHandFrameSink,
    BiHandOutputSink,
    BiHandRetargetingResult,
    HandFrame,
    HandFrameSink,
    OutputSink,
    RetargetingStepResult,
    preprocess_landmarks,
)
from dex_mujoco.visualization import (
    AsyncRobotHandVisualizer,
    AsyncLandmarkVisualizer,
    AsyncBiHandLandmarkVisualizer,
    BiHandScene,
    BiHandVisualizer,
    HandVisualizer,
    configure_free_camera,
    configure_default_hand_camera,
    _try_frame_hand_camera,
)

from .hand_model import HandModel


def _reload_renderer_cls_for_backend(backend: str | None):
    from mujoco.rendering.classic import gl_context as gl_context_module
    from mujoco.rendering.classic import renderer as renderer_module

    if backend is None:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = backend

    importlib.reload(gl_context_module)
    renderer_module = importlib.reload(renderer_module)
    return renderer_module.Renderer


def _create_offscreen_renderer(model, *, width: int, height: int):
    backend_env = os.environ.get("MUJOCO_GL")
    if backend_env:
        try:
            return mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:
            raise RuntimeError(
                "Cannot create MuJoCo replay renderer with the configured "
                f"MUJOCO_GL={backend_env!r}: {exc}"
            ) from exc

    if platform.system() == "Linux":
        try:
            renderer_cls = _reload_renderer_cls_for_backend("egl")
            return renderer_cls(model, height=height, width=width)
        except Exception:
            _reload_renderer_cls_for_backend(None)

    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            "Cannot create MuJoCo replay renderer. If you are running headless, "
            "try `MUJOCO_GL=egl`."
        ) from exc


def _fit_video_size(
    *,
    requested_width: int,
    requested_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    if requested_width <= max_width and requested_height <= max_height:
        return requested_width, requested_height

    scale = min(max_width / requested_width, max_height / requested_height)
    width = max(2, int(requested_width * scale))
    height = max(2, int(requested_height * scale))
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    return width, height


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


def _transform_points(
    points: np.ndarray,
    *,
    pos: tuple[float, float, float] | np.ndarray,
    quat: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    rotation = _quat_to_rotation_matrix(quat)
    translation = np.asarray(pos, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


class TrajectoryRecorder(OutputSink):
    def __init__(self):
        self.trajectory: list[np.ndarray] = []

    @property
    def is_running(self) -> bool:
        return True

    def on_result(self, result: RetargetingStepResult) -> None:
        self.trajectory.append(result.qpos.copy())

    def close(self) -> None:
        return None


class RobotHandOutputSink(OutputSink):
    def __init__(
        self,
        hand_model: HandModel,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
    ):
        self._visualizer = HandVisualizer(
            hand_model,
            key_callback=key_callback,
            overlay_label=overlay_label,
            window_title=window_title,
        )

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        self._visualizer.update(result.qpos)

    def close(self) -> None:
        self._visualizer.close()


class RobotHandTargetOutputSink(OutputSink):
    def __init__(
        self,
        hand_model: HandModel,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
    ):
        self._visualizer = AsyncRobotHandVisualizer(
            hand_model.mjcf_path,
            overlay_label=overlay_label,
            window_title=window_title,
        )

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        qpos = result.target_qpos if result.target_qpos is not None else result.qpos
        self._visualizer.update(qpos)

    def close(self) -> None:
        self._visualizer.close()


class RobotHandVideoOutputSink(OutputSink):
    def __init__(
        self,
        hand_model: HandModel,
        *,
        output_path: str,
        fps: int,
        width: int = 1280,
        height: int = 720,
        codec: str = "mp4v",
    ):
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = hand_model.model
        self._data = mujoco.MjData(self._model)
        width, height = _fit_video_size(
            requested_width=width,
            requested_height=height,
            max_width=max(int(self._model.vis.global_.offwidth), 1),
            max_height=max(int(self._model.vis.global_.offheight), 1),
        )
        self._frame_aspect_ratio = width / max(height, 1)
        self._renderer = _create_offscreen_renderer(self._model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        configure_default_hand_camera(self._camera)
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            cv2.VideoWriter_fourcc(*codec),
            float(max(fps, 1)),
            (width, height),
        )
        if not self._writer.isOpened():
            self._renderer.close()
            raise RuntimeError(f"Cannot open replay video writer for: {self._output_path}")
        self._frames_written = 0
        self._camera_initialized = False
        self._is_closed = False

    @property
    def is_running(self) -> bool:
        return not self._is_closed

    def on_result(self, result: RetargetingStepResult) -> None:
        if self._is_closed:
            return
        self._data.qpos[:] = result.qpos
        mujoco.mj_forward(self._model, self._data)
        if not self._camera_initialized and _try_frame_hand_camera(
            self._camera,
            model=self._model,
            data=self._data,
            aspect_ratio=self._frame_aspect_ratio,
        ):
            self._camera_initialized = True
        self._renderer.update_scene(self._data, camera=self._camera)
        frame_rgb = self._renderer.render()
        self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self._frames_written += 1

    def close(self) -> None:
        if self._is_closed:
            return
        self._writer.release()
        self._renderer.close()
        self._is_closed = True
        print(f"Saved replay video ({self._frames_written} frames) to {self._output_path}")


class AsyncLandmarkOutputSink(OutputSink, HandFrameSink):
    def __init__(self, *, window_title: str | None = None):
        self._visualizer = AsyncLandmarkVisualizer(window_title=window_title)

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        self._visualizer.update(result.processed_landmarks)

    def on_frame(self, frame: HandFrame) -> None:
        self._visualizer.update(
            preprocess_landmarks(
                frame.landmarks_3d,
                hand_side=frame.hand_side,
            )
        )

    def close(self) -> None:
        self._visualizer.close()


class AsyncBiHandLandmarkOutputSink(BiHandFrameSink):
    def __init__(
        self,
        *,
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
    ):
        self._visualizer = AsyncBiHandLandmarkVisualizer()
        self._left_pos = tuple(float(value) for value in left_pos)
        self._right_pos = tuple(float(value) for value in right_pos)
        self._left_quat = tuple(float(value) for value in left_quat)
        self._right_quat = tuple(float(value) for value in right_quat)

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_frame(self, frame: BiHandFrame) -> None:
        left = np.full((21, 3), np.nan, dtype=np.float64)
        right = np.full((21, 3), np.nan, dtype=np.float64)
        if frame.left is not None:
            left = preprocess_landmarks(
                frame.left.landmarks_3d,
                hand_side=frame.left.hand_side,
            )
            left = _transform_points(left, pos=self._left_pos, quat=self._left_quat)
        if frame.right is not None:
            right = preprocess_landmarks(
                frame.right.landmarks_3d,
                hand_side=frame.right.hand_side,
            )
            right = _transform_points(right, pos=self._right_pos, quat=self._right_quat)
        self._visualizer.update(np.stack([left, right], axis=0))

    def close(self) -> None:
        self._visualizer.close()


class _BiHandRenderHelper:
    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        panel_width: int,
        panel_height: int,
        left_pos: tuple[float, float, float],
        right_pos: tuple[float, float, float],
        camera_lookat: tuple[float, float, float],
        left_quat: tuple[float, float, float, float],
        right_quat: tuple[float, float, float, float],
    ):
        self._scene = BiHandScene(
            left_hand_model,
            right_hand_model,
            left_pos=left_pos,
            right_pos=right_pos,
            left_quat=left_quat,
            right_quat=right_quat,
        )
        left_width, left_height = _fit_video_size(
            requested_width=panel_width,
            requested_height=panel_height,
            max_width=max(int(self._scene.model.vis.global_.offwidth), 1),
            max_height=max(int(self._scene.model.vis.global_.offheight), 1),
        )
        self._panel_width = left_width
        self._panel_height = left_height
        self._model = self._scene.model
        self._data = self._scene.data
        self._renderer = _create_offscreen_renderer(
            self._model,
            height=self._panel_height,
            width=self._panel_width,
        )
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        configure_free_camera(
            self._camera,
            distance=0.60,
            azimuth=-90.0,
            elevation=-5.0,
            lookat=camera_lookat,
        )
        self._camera_initialized = False

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._panel_width, self._panel_height

    def render(self, result: BiHandRetargetingResult) -> np.ndarray:
        self._scene.update(result.left.qpos, result.right.qpos)
        if not self._camera_initialized and _try_frame_hand_camera(
            self._camera,
            model=self._model,
            data=self._data,
            aspect_ratio=self._panel_width / max(self._panel_height, 1),
            azimuth=-90.0,
            elevation=-5.0,
        ):
            self._camera_initialized = True
        self._renderer.update_scene(self._data, camera=self._camera)
        return cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._renderer.close()


class BiHandOutputWindowSink(BiHandOutputSink):
    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        key_callback=None,
        panel_width: int = 640,
        panel_height: int = 720,
        window_name: str = "Bi-Hand Retargeting",
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        camera_lookat: tuple[float, float, float] = (0.0, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
    ):
        self._visualizer = BiHandVisualizer(
            left_hand_model,
            right_hand_model,
            key_callback=key_callback,
            left_pos=left_pos,
            right_pos=right_pos,
            camera_lookat=camera_lookat,
            left_quat=left_quat,
            right_quat=right_quat,
        )
        self._window_name = window_name

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: BiHandRetargetingResult) -> None:
        self._visualizer.update(result.left.qpos, result.right.qpos)

    def close(self) -> None:
        self._visualizer.close()


class BiHandVideoOutputSink(BiHandOutputSink):
    def __init__(
        self,
        left_hand_model: HandModel,
        right_hand_model: HandModel,
        *,
        output_path: str,
        fps: int,
        panel_width: int = 640,
        panel_height: int = 720,
        codec: str = "mp4v",
        left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02),
        right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02),
        camera_lookat: tuple[float, float, float] = (0.0, 0.04, 0.02),
        left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151),
        right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665),
    ):
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._renderer = _BiHandRenderHelper(
            left_hand_model,
            right_hand_model,
            panel_width=panel_width,
            panel_height=panel_height,
            left_pos=left_pos,
            right_pos=right_pos,
            camera_lookat=camera_lookat,
            left_quat=left_quat,
            right_quat=right_quat,
        )
        width, height = self._renderer.frame_size
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            cv2.VideoWriter_fourcc(*codec),
            float(max(fps, 1)),
            (width, height),
        )
        if not self._writer.isOpened():
            self._renderer.close()
            raise RuntimeError(f"Cannot open bi-hand replay video writer for: {self._output_path}")
        self._frames_written = 0
        self._is_closed = False

    @property
    def is_running(self) -> bool:
        return not self._is_closed

    def on_result(self, result: BiHandRetargetingResult) -> None:
        if self._is_closed:
            return
        self._writer.write(self._renderer.render(result))
        self._frames_written += 1

    def close(self) -> None:
        if self._is_closed:
            return
        self._writer.release()
        self._renderer.close()
        self._is_closed = True
        print(f"Saved bi-hand replay video ({self._frames_written} frames) to {self._output_path}")
