"""Runtime sinks for robot visualization and trajectory recording."""

from __future__ import annotations

import numpy as np

from dex_mujoco.domain import HandFrame, HandFrameSink, OutputSink, RetargetingStepResult, preprocess_landmarks
from dex_mujoco.visualization import AsyncLandmarkVisualizer, HandVisualizer

from .hand_model import HandModel


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
    def __init__(self, hand_model: HandModel, *, key_callback=None):
        self._visualizer = HandVisualizer(hand_model, key_callback=key_callback)

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        self._visualizer.update(result.qpos)

    def close(self) -> None:
        self._visualizer.close()


class AsyncLandmarkOutputSink(OutputSink, HandFrameSink):
    def __init__(self):
        self._visualizer = AsyncLandmarkVisualizer()

    @property
    def is_running(self) -> bool:
        return self._visualizer.is_running

    def on_result(self, result: RetargetingStepResult) -> None:
        self._visualizer.update(result.processed_landmarks)

    def on_frame(self, frame: HandFrame) -> None:
        self._visualizer.update(
            preprocess_landmarks(
                frame.landmarks_3d,
                handedness=frame.handedness,
            )
        )

    def close(self) -> None:
        self._visualizer.close()
