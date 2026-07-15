"""Rendering helpers for runtime output sinks."""

from __future__ import annotations

import importlib
import os
import platform

import cv2
import mujoco
import numpy as np

from somehand.runtime.viewer_camera import configure_free_camera, try_frame_hand_camera
from somehand.runtime.viewer_hand import BiHandScene


def reload_renderer_cls_for_backend(backend: str | None):
    from mujoco.rendering.classic import gl_context as gl_context_module
    from mujoco.rendering.classic import renderer as renderer_module

    if backend is None:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = backend

    importlib.reload(gl_context_module)
    renderer_module = importlib.reload(renderer_module)
    return renderer_module.Renderer


def create_offscreen_renderer(model, *, width: int, height: int):
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
            renderer_cls = reload_renderer_cls_for_backend("egl")
            return renderer_cls(model, height=height, width=width)
        except Exception:
            reload_renderer_cls_for_backend(None)

    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            "Cannot create MuJoCo replay renderer. If you are running headless, "
            "try `MUJOCO_GL=egl`."
        ) from exc


def fit_video_size(
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


def quat_to_rotation_matrix(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
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


def transform_points(
    points: np.ndarray,
    *,
    pos: tuple[float, float, float] | np.ndarray,
    quat: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    rotation = quat_to_rotation_matrix(quat)
    translation = np.asarray(pos, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


class BiHandRenderHelper:
    def __init__(
        self,
        left_hand_model,
        right_hand_model,
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
        left_width, left_height = fit_video_size(
            requested_width=panel_width,
            requested_height=panel_height,
            max_width=max(int(self._scene.model.vis.global_.offwidth), 1),
            max_height=max(int(self._scene.model.vis.global_.offheight), 1),
        )
        self._panel_width = left_width
        self._panel_height = left_height
        self._model = self._scene.model
        self._data = self._scene.data
        self._renderer = create_offscreen_renderer(
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

    def render(self, result) -> np.ndarray:
        self._scene.update(result.left.qpos, result.right.qpos)
        if not self._camera_initialized and try_frame_hand_camera(
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


__all__ = [
    "BiHandRenderHelper",
    "create_offscreen_renderer",
    "fit_video_size",
    "quat_to_rotation_matrix",
    "reload_renderer_cls_for_backend",
    "transform_points",
]
