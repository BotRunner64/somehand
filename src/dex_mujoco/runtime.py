"""Shared runtime for all dex retargeting CLI entrypoints."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hand_model import HandModel
from .retargeting_config import RetargetingConfig
from .vector_retargeting import VectorRetargeter, preprocess_landmarks
from .visualization import AsyncLandmarkVisualizer, HandVisualizer


@dataclass(frozen=True)
class RuntimeOptions:
    config_path: str
    visualize: bool = False
    output_path: str | None = None


class RetargetRuntime:
    def __init__(self, options: RuntimeOptions, *, input_type: str):
        self.options = options
        self.input_type = input_type
        self.config = RetargetingConfig.load(options.config_path)
        if input_type == "hc_mocap" and self.config.preprocess.frame == "wrist_local":
            print(
                "hc_mocap input detected: overriding preprocess.frame "
                "from wrist_local to camera_aligned and using wrist-pose local landmarks."
            )
            self.config.preprocess.frame = "camera_aligned"

        self.hand_model = HandModel(self.config.hand.mjcf_path)
        self.retargeter = VectorRetargeter(self.hand_model, self.config)
        self.visualizer = HandVisualizer(self.hand_model) if options.visualize else None
        self.landmark_visualizer = AsyncLandmarkVisualizer() if options.visualize else None

        self.trajectory: list[np.ndarray] = []

    @property
    def is_running(self) -> bool:
        if self.visualizer is None and self.landmark_visualizer is None:
            return True
        return all(
            viewer.is_running
            for viewer in (self.visualizer, self.landmark_visualizer)
            if viewer is not None
        )

    def print_startup(
        self,
        *,
        source_desc: str,
        tracking_desc: str,
        extra_lines: list[str] | None = None,
    ) -> None:
        print(f"Model: {self.config.hand.name} ({self.hand_model.nq} DOF)")
        print(f"Retargeting: {len(self.config.human_vector_pairs)} vector pairs")
        print(f"Input source: {source_desc}")
        print(tracking_desc)
        if self.visualizer is not None:
            print("MuJoCo viewers: one for retargeted robot hand, one for input hand mocap")
        if extra_lines:
            for line in extra_lines:
                print(line)

    def process_detection(self, detection) -> np.ndarray:
        retarget_landmarks = detection.landmarks_3d_local
        if retarget_landmarks is None:
            retarget_landmarks = detection.landmarks_3d

        self.retargeter.update_targets(retarget_landmarks, detection.handedness)
        qpos = self.retargeter.solve()
        self.trajectory.append(qpos.copy())

        if self.visualizer is not None:
            landmarks_for_vis = preprocess_landmarks(
                retarget_landmarks,
                handedness=detection.handedness,
                frame=self.config.preprocess.frame,
            )
            self.visualizer.update(qpos)
            if self.landmark_visualizer is not None:
                self.landmark_visualizer.update(landmarks_for_vis)

        return qpos

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
        if not self.options.output_path or not self.trajectory:
            return

        output_path = Path(self.options.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trajectory": np.array(self.trajectory),
            "joint_names": self.hand_model.get_joint_names(),
            "config_path": self.options.config_path,
            "num_frames": num_frames,
            "input_source": source_desc,
            "input_type": self.input_type,
        }
        if handedness is not None:
            data["handedness"] = handedness
        if num_detected is not None:
            data["num_detected"] = num_detected

        with output_path.open("wb") as f:
            pickle.dump(data, f)
        print(f"Saved trajectory ({len(self.trajectory)} frames) to {output_path}")

    def close(self) -> None:
        if self.landmark_visualizer is not None:
            self.landmark_visualizer.close()
        if self.visualizer is not None:
            self.visualizer.close()
