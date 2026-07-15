"""Runtime sink implementations for visualization and recording."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from somehand.core import BiHandFrame, BiHandFrameSink, BiHandOutputSink, HandFrame, HandFrameSink, OutputSink, RetargetingStepResult, preprocess_landmarks
from somehand.runtime.viewer_async import AsyncBiHandLandmarkVisualizer, AsyncLandmarkVisualizer, AsyncRobotHandVisualizer
from somehand.runtime.viewer_camera import configure_default_hand_camera, try_frame_hand_camera
from somehand.runtime.viewer_hand import BiHandVisualizer, HandVisualizer, set_fingertip_site_visibility

from .sink_rendering import BiHandRenderHelper, create_offscreen_renderer, fit_video_size, transform_points


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
        hand_model,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
        viewer_mode: str = "normal",
        hand_side: str | None = None,
        robot_vector_specs: list[tuple[int, str, str, str, str]] | None = None,
        robot_distance_specs: list[tuple[int, str, str, str, str]] | None = None,
        robot_frame_specs: list[tuple[int, str, str, str, str, str, str]] | None = None,
        robot_angle_specs: list[tuple[int, str]] | None = None,
    ):
        self._visualizer = HandVisualizer(
            hand_model,
            key_callback=key_callback,
            overlay_label=overlay_label,
            window_title=window_title,
            viewer_mode=viewer_mode,
            hand_side=hand_side,
            robot_vector_specs=robot_vector_specs,
            robot_distance_specs=robot_distance_specs,
            robot_frame_specs=robot_frame_specs,
            robot_angle_specs=robot_angle_specs,
        )

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        self._visualizer.update(
            result.qpos,
            target_directions=result.target_directions,
            target_frame_primary_directions=getattr(result, "target_frame_primary_directions", None),
            target_frame_secondary_directions=getattr(result, "target_frame_secondary_directions", None),
            target_distances=getattr(result, "target_distances", None),
            target_angles=getattr(result, "target_angles", None),
        )

    def close(self) -> None:
        self._visualizer.close()


class RobotHandTargetOutputSink(OutputSink):
    def __init__(
        self,
        hand_model,
        *,
        key_callback=None,
        overlay_label: str | None = None,
        window_title: str | None = None,
        viewer_mode: str = "normal",
        hand_side: str | None = None,
        robot_vector_specs: list[tuple[int, str, str, str, str]] | None = None,
        robot_distance_specs: list[tuple[int, str, str, str, str]] | None = None,
        robot_frame_specs: list[tuple[int, str, str, str, str, str, str]] | None = None,
        robot_angle_specs: list[tuple[int, str]] | None = None,
    ):
        self._visualizer = AsyncRobotHandVisualizer(
            hand_model.mjcf_path,
            overlay_label=overlay_label,
            window_title=window_title,
            viewer_mode=viewer_mode,
            hand_side=hand_side,
            robot_vector_specs=robot_vector_specs,
            robot_distance_specs=robot_distance_specs,
            robot_frame_specs=robot_frame_specs,
            robot_angle_specs=robot_angle_specs,
        )

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        qpos = result.target_qpos if result.target_qpos is not None else result.qpos
        self._visualizer.update_with_vectors(
            qpos,
            result.target_directions,
            target_frame_primary_directions=getattr(result, "target_frame_primary_directions", None),
            target_frame_secondary_directions=getattr(result, "target_frame_secondary_directions", None),
            target_distances=getattr(result, "target_distances", None),
            target_angles=getattr(result, "target_angles", None),
        )

    def close(self) -> None:
        self._visualizer.close()


class RobotHandVideoOutputSink(OutputSink):
    def __init__(
        self,
        hand_model,
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
        set_fingertip_site_visibility(self._model, visible=False)
        self._data = mujoco.MjData(self._model)
        width, height = fit_video_size(
            requested_width=width,
            requested_height=height,
            max_width=max(int(self._model.vis.global_.offwidth), 1),
            max_height=max(int(self._model.vis.global_.offheight), 1),
        )
        self._frame_aspect_ratio = width / max(height, 1)
        self._renderer = create_offscreen_renderer(self._model, height=height, width=width)
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
        if not self._camera_initialized and try_frame_hand_camera(
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
    def __init__(
        self,
        *,
        window_title: str | None = None,
        vector_pairs: list[tuple[int, int]] | None = None,
        distance_pairs: list[tuple[int, int]] | None = None,
        frame_triples: list[tuple[int, int, int]] | None = None,
        angle_triples: list[tuple[int, int, int]] | None = None,
    ):
        self._visualizer = AsyncLandmarkVisualizer(
            window_title=window_title,
            vector_pairs=vector_pairs,
            distance_pairs=distance_pairs,
            frame_triples=frame_triples,
            angle_triples=angle_triples,
        )

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
        left_vector_pairs: list[tuple[int, int]] | None = None,
        right_vector_pairs: list[tuple[int, int]] | None = None,
        left_distance_pairs: list[tuple[int, int]] | None = None,
        right_distance_pairs: list[tuple[int, int]] | None = None,
        left_frame_triples: list[tuple[int, int, int]] | None = None,
        right_frame_triples: list[tuple[int, int, int]] | None = None,
        left_angle_triples: list[tuple[int, int, int]] | None = None,
        right_angle_triples: list[tuple[int, int, int]] | None = None,
    ):
        self._visualizer = AsyncBiHandLandmarkVisualizer(
            left_vector_pairs=left_vector_pairs,
            right_vector_pairs=right_vector_pairs,
            left_distance_pairs=left_distance_pairs,
            right_distance_pairs=right_distance_pairs,
            left_frame_triples=left_frame_triples,
            right_frame_triples=right_frame_triples,
            left_angle_triples=left_angle_triples,
            right_angle_triples=right_angle_triples,
        )
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
            left = transform_points(left, pos=self._left_pos, quat=self._left_quat)
        if frame.right is not None:
            right = preprocess_landmarks(
                frame.right.landmarks_3d,
                hand_side=frame.right.hand_side,
            )
            right = transform_points(right, pos=self._right_pos, quat=self._right_quat)
        self._visualizer.update(np.stack([left, right], axis=0))

    def on_result(self, result) -> None:
        left = transform_points(
            result.left.processed_landmarks,
            pos=self._left_pos,
            quat=self._left_quat,
        )
        right = transform_points(
            result.right.processed_landmarks,
            pos=self._right_pos,
            quat=self._right_quat,
        )
        self._visualizer.update(np.stack([left, right], axis=0))

    def close(self) -> None:
        self._visualizer.close()


class BiHandOutputWindowSink(BiHandOutputSink):
    def __init__(
        self,
        left_hand_model,
        right_hand_model,
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
        viewer_mode: str = "normal",
        left_hand_side: str | None = None,
        right_hand_side: str | None = None,
        left_robot_vector_specs: list[tuple[int, str, str, str, str]] | None = None,
        right_robot_vector_specs: list[tuple[int, str, str, str, str]] | None = None,
        left_robot_distance_specs: list[tuple[int, str, str, str, str]] | None = None,
        right_robot_distance_specs: list[tuple[int, str, str, str, str]] | None = None,
        left_robot_frame_specs: list[tuple[int, str, str, str, str, str, str]] | None = None,
        right_robot_frame_specs: list[tuple[int, str, str, str, str, str, str]] | None = None,
        left_robot_angle_specs: list[tuple[int, str]] | None = None,
        right_robot_angle_specs: list[tuple[int, str]] | None = None,
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
        self._window_name = window_name

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result) -> None:
        self._visualizer.update(
            result.left.qpos,
            result.right.qpos,
            left_target_directions=result.left.target_directions,
            right_target_directions=result.right.target_directions,
            left_target_frame_primary_directions=getattr(result.left, "target_frame_primary_directions", None),
            right_target_frame_primary_directions=getattr(result.right, "target_frame_primary_directions", None),
            left_target_frame_secondary_directions=getattr(result.left, "target_frame_secondary_directions", None),
            right_target_frame_secondary_directions=getattr(result.right, "target_frame_secondary_directions", None),
            left_target_distances=getattr(result.left, "target_distances", None),
            right_target_distances=getattr(result.right, "target_distances", None),
            left_target_angles=getattr(result.left, "target_angles", None),
            right_target_angles=getattr(result.right, "target_angles", None),
        )

    def close(self) -> None:
        self._visualizer.close()


class BiHandVideoOutputSink(BiHandOutputSink):
    def __init__(
        self,
        left_hand_model,
        right_hand_model,
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
        self._renderer = BiHandRenderHelper(
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

    def on_result(self, result) -> None:
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


__all__ = [
    "AsyncBiHandLandmarkOutputSink",
    "AsyncLandmarkOutputSink",
    "BiHandOutputWindowSink",
    "BiHandVideoOutputSink",
    "RobotHandOutputSink",
    "RobotHandTargetOutputSink",
    "RobotHandVideoOutputSink",
    "TrajectoryRecorder",
]
