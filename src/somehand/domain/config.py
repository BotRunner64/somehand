"""Domain configuration models for retargeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from somehand.external_assets import build_missing_asset_message

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
class ControllerConfig:
    backend: str = "viewer"
    model_family: str = ""
    control_rate_hz: int = 100
    sim_rate_hz: int = 500
    transport: str = "can"
    can_interface: str = "can0"
    modbus_port: str = "None"
    sdk_root: str = ""
    default_speed: list[int] = field(default_factory=list)
    default_torque: list[int] = field(default_factory=list)


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
class VectorConstraint:
    human: list[int] = field(default_factory=list)
    robot: list[str] = field(default_factory=list)
    robot_types: list[str] = field(default_factory=lambda: ["body", "body"])
    weight: float = 1.0
    loss_type: str = ""
    loss_scale: float = 0.0


@dataclass
class FrameConstraint:
    name: str = ""
    human_origin: int = 0
    human_primary: int = 0
    human_secondary: int = 0
    robot_origin: str = ""
    robot_primary: str = ""
    robot_secondary: str = ""
    robot_types: list[str] = field(default_factory=lambda: ["body", "body", "body"])
    primary_weight: float = 1.0
    secondary_weight: float = 1.0


@dataclass
class DistanceConstraint:
    human: list[int] = field(default_factory=list)
    robot: list[str] = field(default_factory=list)
    robot_types: list[str] = field(default_factory=lambda: ["site", "site"])
    weight: float = 1.0
    scale: float = 1.0
    threshold: float = 0.04
    activation_type: str = "gaussian"


@dataclass
class RetargetingConfig:
    hand: HandConfig = field(default_factory=HandConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    vector_constraints: list[VectorConstraint] = field(default_factory=list)
    frame_constraints: list[FrameConstraint] = field(default_factory=list)
    distance_constraints: list[DistanceConstraint] = field(default_factory=list)
    vector_loss: VectorLossConfig = field(default_factory=VectorLossConfig)
    angle_constraints: list[AngleConstraint] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    @property
    def human_vector_pairs(self) -> list[list[int]]:
        return [list(constraint.human) for constraint in self.vector_constraints]

    @property
    def origin_link_names(self) -> list[str]:
        return [constraint.robot[0] for constraint in self.vector_constraints]

    @property
    def task_link_names(self) -> list[str]:
        return [constraint.robot[1] for constraint in self.vector_constraints]

    @property
    def origin_link_types(self) -> list[str]:
        return [constraint.robot_types[0] for constraint in self.vector_constraints]

    @property
    def task_link_types(self) -> list[str]:
        return [constraint.robot_types[1] for constraint in self.vector_constraints]

    @property
    def vector_weights(self) -> list[float]:
        return [constraint.weight for constraint in self.vector_constraints]

    def validate(self) -> None:
        if not self.hand.side:
            raise ValueError("hand.side must be explicitly set to 'left' or 'right'")
        self.hand.side = normalize_hand_side(self.hand.side)
        for constraint in self.vector_constraints:
            if len(constraint.human) != 2:
                raise ValueError("vector constraint human must have length 2")
            if len(constraint.robot) != 2:
                raise ValueError("vector constraint robot must have length 2")
            if len(constraint.robot_types) != 2:
                raise ValueError("vector constraint robot_types must have length 2")
            if any(link_type not in {"body", "site"} for link_type in constraint.robot_types):
                raise ValueError("vector constraint robot_types must only contain 'body' or 'site'")
            if constraint.weight < 0.0:
                raise ValueError("vector constraint weight must be >= 0")
            if constraint.loss_type and constraint.loss_type not in {"direction", "residual"}:
                raise ValueError("vector constraint loss_type must be 'direction', 'residual', or empty")
        for constraint in self.frame_constraints:
            if len(constraint.robot_types) != 3:
                raise ValueError("frame constraint robot_types must have length 3")
            if any(link_type not in {"body", "site"} for link_type in constraint.robot_types):
                raise ValueError("frame constraint robot_types must only contain 'body' or 'site'")
            if constraint.primary_weight < 0.0:
                raise ValueError("frame constraint primary_weight must be >= 0")
            if constraint.secondary_weight < 0.0:
                raise ValueError("frame constraint secondary_weight must be >= 0")
        for constraint in self.distance_constraints:
            if len(constraint.human) != 2:
                raise ValueError("distance constraint human must have length 2")
            if len(constraint.robot) != 2:
                raise ValueError("distance constraint robot must have length 2")
            if len(constraint.robot_types) != 2:
                raise ValueError("distance constraint robot_types must have length 2")
            if any(link_type not in {"body", "site"} for link_type in constraint.robot_types):
                raise ValueError("distance constraint robot_types must only contain 'body' or 'site'")
            if constraint.weight < 0.0:
                raise ValueError("distance constraint weight must be >= 0")
            if constraint.activation_type not in {"gaussian", "linear"}:
                raise ValueError(
                    f"distance constraint activation_type must be 'gaussian' or 'linear', "
                    f"got '{constraint.activation_type}'"
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
        if self.hand.side not in HAND_SIDES:
            raise ValueError("hand.side must only contain 'left' or 'right'")
        if not Path(self.hand.mjcf_path).exists():
            raise FileNotFoundError(
                build_missing_asset_message(
                    self.hand.mjcf_path,
                    label="MJCF file",
                )
            )
        if self.controller.backend not in {"viewer", "sim", "real"}:
            raise ValueError("controller.backend must be one of: viewer, sim, real")
        if self.controller.transport not in {"can", "modbus"}:
            raise ValueError("controller.transport must be one of: can, modbus")
        if self.controller.control_rate_hz <= 0:
            raise ValueError("controller.control_rate_hz must be > 0")
        if self.controller.sim_rate_hz <= 0:
            raise ValueError("controller.sim_rate_hz must be > 0")


@dataclass
class BiHandViewerConfig:
    panel_width: int = 640
    panel_height: int = 720
    window_name: str = "Bi-Hand Retargeting"
    left_pos: tuple[float, float, float] = (0.22, 0.04, 0.02)
    right_pos: tuple[float, float, float] = (-0.22, 0.04, 0.02)
    camera_lookat: tuple[float, float, float] = (0.0, 0.04, 0.02)
    left_quat: tuple[float, float, float, float] = (0.69288325, 0.01522078, -0.05862347, 0.71850151)
    right_quat: tuple[float, float, float, float] = (0.71846417, 0.05829359, -0.01490552, 0.69295665)


@dataclass
class BiHandRetargetingConfig:
    left_config_path: str = ""
    right_config_path: str = ""
    viewer: BiHandViewerConfig = field(default_factory=BiHandViewerConfig)

    def validate(self) -> None:
        if not self.left_config_path:
            raise ValueError("left_config_path must be set")
        if not self.right_config_path:
            raise ValueError("right_config_path must be set")
        if not Path(self.left_config_path).exists():
            raise FileNotFoundError(f"Left-hand config not found: {self.left_config_path}")
        if not Path(self.right_config_path).exists():
            raise FileNotFoundError(f"Right-hand config not found: {self.right_config_path}")
        if self.viewer.panel_width <= 0:
            raise ValueError("viewer.panel_width must be > 0")
        if self.viewer.panel_height <= 0:
            raise ValueError("viewer.panel_height must be > 0")
        if len(self.viewer.left_pos) != 3:
            raise ValueError("viewer.left_pos must have length 3")
        if len(self.viewer.right_pos) != 3:
            raise ValueError("viewer.right_pos must have length 3")
        if len(self.viewer.camera_lookat) != 3:
            raise ValueError("viewer.camera_lookat must have length 3")
        if len(self.viewer.left_quat) != 4:
            raise ValueError("viewer.left_quat must have length 4")
        if len(self.viewer.right_quat) != 4:
            raise ValueError("viewer.right_quat must have length 4")
