"""Domain configuration models for retargeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .hand_side import HAND_SIDES, normalize_hand_side


@dataclass
class SolverConfig:
    max_iterations: int = 30
    norm_delta: float = 0.01
    output_alpha: float = 0.70


@dataclass
class PreprocessConfig:
    temporal_filter_alpha: float = 0.35


@dataclass
class HandConfig:
    name: str = ""
    side: str = ""
    mjcf_path: str = ""
    urdf_source: str = ""


@dataclass
class PinchConfig:
    enabled: bool = False
    d1: float = 0.03
    d2: float = 0.06
    weight: float = 5.0
    thumb_weight_boost: float = 1.5
    fingertip_sites: list[str] = field(default_factory=list)


@dataclass
class PositionConstraint:
    landmark: int = 0
    body: str = ""
    body_type: str = "body"
    weight: float = 1.0


@dataclass
class PositionConfig:
    enabled: bool = False
    weight: float = 8.0
    scale_landmarks: list[int] = field(default_factory=lambda: [0, 9])
    scale_bodies: list[str] = field(default_factory=lambda: ["world", "middle_proximal"])
    scale_body_types: list[str] = field(default_factory=lambda: ["body", "body"])
    constraints: list[PositionConstraint] = field(default_factory=list)


@dataclass
class VectorLossConfig:
    type: str = "direction"
    huber_delta: float = 0.02
    scaling: float = 1.0
    scale_landmarks: list[int] = field(default_factory=lambda: [0, 9])
    scale_bodies: list[str] = field(default_factory=lambda: ["world", "middle_proximal"])
    scale_body_types: list[str] = field(default_factory=lambda: ["body", "body"])


@dataclass
class AngleConstraint:
    landmarks: list[int] = field(default_factory=list)
    joint: str = ""
    weight: float = 1.0
    scale: float = 1.0
    invert: bool = False


@dataclass
class RetargetingConfig:
    hand: HandConfig = field(default_factory=HandConfig)
    human_vector_pairs: list[list[int]] = field(default_factory=list)
    origin_link_names: list[str] = field(default_factory=list)
    task_link_names: list[str] = field(default_factory=list)
    origin_link_types: list[str] = field(default_factory=list)
    task_link_types: list[str] = field(default_factory=list)
    vector_weights: list[float] = field(default_factory=list)
    vector_loss: VectorLossConfig = field(default_factory=VectorLossConfig)
    angle_constraints: list[AngleConstraint] = field(default_factory=list)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    @classmethod
    def load(cls, config_path: str) -> "RetargetingConfig":
        from dex_mujoco.infrastructure.config_loader import load_retargeting_config

        return load_retargeting_config(config_path)

    def validate(self) -> None:
        if not self.hand.side:
            raise ValueError("hand.side must be explicitly set to 'left' or 'right'")
        self.hand.side = normalize_hand_side(self.hand.side)
        n = len(self.human_vector_pairs)
        if len(self.origin_link_names) != n:
            raise ValueError(
                f"origin_link_names length ({len(self.origin_link_names)}) "
                f"must match human_vector_pairs length ({n})"
            )
        if len(self.task_link_names) != n:
            raise ValueError(
                f"task_link_names length ({len(self.task_link_names)}) "
                f"must match human_vector_pairs length ({n})"
            )
        if len(self.origin_link_types) != n:
            raise ValueError(
                f"origin_link_types length ({len(self.origin_link_types)}) "
                f"must match human_vector_pairs length ({n})"
            )
        if len(self.task_link_types) != n:
            raise ValueError(
                f"task_link_types length ({len(self.task_link_types)}) "
                f"must match human_vector_pairs length ({n})"
            )
        if len(self.vector_weights) != n:
            raise ValueError(
                f"vector_weights length ({len(self.vector_weights)}) "
                f"must match human_vector_pairs length ({n})"
            )
        if not 0.0 < self.preprocess.temporal_filter_alpha <= 1.0:
            raise ValueError("temporal_filter_alpha must be in (0, 1]")
        if not 0.0 < self.solver.output_alpha <= 1.0:
            raise ValueError("solver.output_alpha must be in (0, 1]")
        if self.vector_loss.type not in {"direction", "residual"}:
            raise ValueError("vector_loss.type must be 'direction' or 'residual'")
        if self.vector_loss.huber_delta <= 0.0:
            raise ValueError("vector_loss.huber_delta must be > 0")
        if self.vector_loss.scaling <= 0.0:
            raise ValueError("vector_loss.scaling must be > 0")
        if len(self.vector_loss.scale_landmarks) != 2:
            raise ValueError("vector_loss.scale_landmarks must have length 2")
        if len(self.vector_loss.scale_bodies) != 2:
            raise ValueError("vector_loss.scale_bodies must have length 2")
        if len(self.vector_loss.scale_body_types) != 2:
            raise ValueError("vector_loss.scale_body_types must have length 2")
        if any(link_type not in {"body", "site"} for link_type in self.vector_loss.scale_body_types):
            raise ValueError("vector_loss.scale_body_types must only contain 'body' or 'site'")
        for constraint in self.angle_constraints:
            if constraint.scale <= 0.0:
                raise ValueError("angle constraint scale must be > 0")
        if any(link_type not in {"body", "site"} for link_type in self.origin_link_types):
            raise ValueError("origin_link_types must only contain 'body' or 'site'")
        if any(link_type not in {"body", "site"} for link_type in self.task_link_types):
            raise ValueError("task_link_types must only contain 'body' or 'site'")
        if self.hand.side not in HAND_SIDES:
            raise ValueError("hand.side must only contain 'left' or 'right'")
        if not Path(self.hand.mjcf_path).exists():
            raise FileNotFoundError(f"MJCF file not found: {self.hand.mjcf_path}")
