"""Application service that turns hand frames into robot joint targets."""

from __future__ import annotations

from dex_mujoco.domain import HandFrame, RetargetingConfig, RetargetingStepResult, preprocess_landmarks
from dex_mujoco.infrastructure.hand_model import HandModel
from dex_mujoco.infrastructure.vector_solver import VectorRetargeter


class RetargetingEngine:
    """Stable application-layer entry for one-step retargeting."""

    def __init__(self, config: RetargetingConfig, *, input_type: str):
        self.config = config
        self.input_type = input_type
        self.hand_model = HandModel(config.hand.mjcf_path)
        self.retargeter = VectorRetargeter(self.hand_model, config)

    @classmethod
    def from_config_path(cls, config_path: str, *, input_type: str) -> "RetargetingEngine":
        return cls(RetargetingConfig.load(config_path), input_type=input_type)

    def describe(self) -> dict[str, int | str]:
        return {
            "model_name": self.config.hand.name,
            "dof": self.hand_model.nq,
            "vector_pairs": len(self.config.human_vector_pairs),
        }

    def process(self, frame: HandFrame) -> RetargetingStepResult:
        landmarks = frame.landmarks_3d
        if frame.hand_side != self.config.hand.side:
            raise ValueError(
                f"input hand side {frame.hand_side!r} does not match config hand side {self.config.hand.side!r}"
            )
        self.retargeter.update_targets(
            landmarks,
            hand_side=frame.hand_side,
        )
        qpos = self.retargeter.solve()
        processed_landmarks = preprocess_landmarks(
            landmarks,
            hand_side=frame.hand_side,
        )
        return RetargetingStepResult(
            qpos=qpos.copy(),
            target_directions=self.retargeter.get_target_directions(),
            processed_landmarks=processed_landmarks,
            hand_side=frame.hand_side,
        )
