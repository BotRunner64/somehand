"""Compatibility runtime adapter over the new layered pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from dex_mujoco.application import RetargetingEngine
from dex_mujoco.domain import HandFrame
from dex_mujoco.infrastructure import AsyncLandmarkOutputSink, RobotHandOutputSink, TrajectoryRecorder, save_trajectory_artifact


@dataclass(frozen=True)
class RuntimeOptions:
    config_path: str
    visualize: bool = False
    output_path: str | None = None


class RetargetRuntime:
    def __init__(self, options: RuntimeOptions, *, input_type: str):
        self.options = options
        self.input_type = input_type
        self.engine = RetargetingEngine.from_config_path(options.config_path, input_type=input_type)
        self.config = self.engine.config
        self.hand_model = self.engine.hand_model
        self.trajectory_recorder = TrajectoryRecorder()
        self.visualizer = RobotHandOutputSink(self.hand_model) if options.visualize else None
        self.landmark_visualizer = AsyncLandmarkOutputSink() if options.visualize else None

    @property
    def trajectory(self):
        return self.trajectory_recorder.trajectory

    @property
    def is_running(self) -> bool:
        viewers = [sink for sink in (self.visualizer, self.landmark_visualizer) if sink is not None]
        if not viewers:
            return True
        return all(sink.is_running for sink in viewers)

    def print_startup(
        self,
        *,
        source_desc: str,
        tracking_desc: str,
        extra_lines: list[str] | None = None,
    ) -> None:
        details = self.engine.describe()
        print(f"Model: {details['model_name']} ({details['dof']} DOF)")
        print(f"Retargeting: {details['vector_pairs']} vector pairs")
        print(f"Input source: {source_desc}")
        print(tracking_desc)
        if self.visualizer is not None:
            print("MuJoCo viewers: one for retargeted robot hand, one for input hand mocap")
        if extra_lines:
            for line in extra_lines:
                print(line)

    def process_detection(self, detection) -> None:
        frame = HandFrame(
            landmarks_3d=detection.landmarks_3d,
            landmarks_2d=detection.landmarks_2d,
            handedness=detection.handedness,
        )
        result = self.engine.process(frame)
        self.trajectory_recorder.on_result(result)
        if self.visualizer is not None:
            self.visualizer.on_result(result)
        if self.landmark_visualizer is not None:
            self.landmark_visualizer.on_result(result)

    def print_summary(self, *, num_frames: int, num_detected: int | None = None) -> None:
        if num_detected is None:
            print(f"Processed {num_frames} frames")
            return
        print(f"Processed {num_frames} frames, detected hand in {num_detected} frames")

    def save_output(
        self,
        *,
        source_desc: str,
        num_frames: int,
        handedness: str | None = None,
        num_detected: int | None = None,
    ) -> None:
        save_trajectory_artifact(
            self.options.output_path,
            self.trajectory,
            joint_names=self.hand_model.get_joint_names(),
            config_path=self.options.config_path,
            num_frames=num_frames,
            source_desc=source_desc,
            input_type=self.input_type,
            handedness=handedness,
            num_detected=num_detected,
        )

    def close(self) -> None:
        if self.landmark_visualizer is not None:
            self.landmark_visualizer.close()
        if self.visualizer is not None:
            self.visualizer.close()
