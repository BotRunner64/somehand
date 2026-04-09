"""Application service that retargets left and right hands independently."""

from __future__ import annotations

import numpy as np

from dex_mujoco.domain import (
    BiHandFrame,
    BiHandRetargetingConfig,
    BiHandRetargetingResult,
    RetargetingStepResult,
)

from .engine import RetargetingEngine


def _copy_step_result(result: RetargetingStepResult) -> RetargetingStepResult:
    return RetargetingStepResult(
        qpos=np.array(result.qpos, copy=True),
        target_directions=None if result.target_directions is None else np.array(result.target_directions, copy=True),
        processed_landmarks=np.array(result.processed_landmarks, copy=True),
        hand_side=result.hand_side,
    )


class BiHandRetargetingEngine:
    """Stable application-layer entry for one-step bi-hand retargeting."""

    def __init__(self, config: BiHandRetargetingConfig, *, input_type: str):
        self.config = config
        self.input_type = input_type
        self.left_engine = RetargetingEngine.from_config_path(config.left_config_path, input_type=input_type)
        self.right_engine = RetargetingEngine.from_config_path(config.right_config_path, input_type=input_type)
        self._left_result = self._neutral_result(self.left_engine, hand_side="left")
        self._right_result = self._neutral_result(self.right_engine, hand_side="right")

    @classmethod
    def from_config_path(cls, config_path: str, *, input_type: str) -> "BiHandRetargetingEngine":
        return cls(BiHandRetargetingConfig.load(config_path), input_type=input_type)

    def describe(self) -> dict[str, object]:
        return {
            "left_model_name": self.left_engine.config.hand.name,
            "right_model_name": self.right_engine.config.hand.name,
            "left_dof": self.left_engine.hand_model.nq,
            "right_dof": self.right_engine.hand_model.nq,
        }

    def process(self, frame: BiHandFrame) -> BiHandRetargetingResult:
        left_detected = frame.left is not None
        right_detected = frame.right is not None

        if left_detected:
            self._left_result = self.left_engine.process(frame.left)
        if right_detected:
            self._right_result = self.right_engine.process(frame.right)

        return BiHandRetargetingResult(
            left=_copy_step_result(self._left_result),
            right=_copy_step_result(self._right_result),
            left_detected=left_detected,
            right_detected=right_detected,
        )

    @staticmethod
    def _neutral_result(engine: RetargetingEngine, *, hand_side: str) -> RetargetingStepResult:
        return RetargetingStepResult(
            qpos=engine.hand_model.get_qpos().copy(),
            target_directions=None,
            processed_landmarks=np.zeros((21, 3), dtype=np.float64),
            hand_side=hand_side,
        )
