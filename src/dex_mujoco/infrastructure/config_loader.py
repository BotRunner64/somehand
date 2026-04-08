"""YAML parsing and filesystem resolution for retargeting configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from dex_mujoco.domain.config import (
    AngleConstraint,
    HandConfig,
    PinchConfig,
    PositionConfig,
    PositionConstraint,
    PreprocessConfig,
    RetargetingConfig,
    SolverConfig,
    VectorLossConfig,
)


def load_retargeting_config(config_path: str) -> RetargetingConfig:
    config_path_obj = Path(config_path)
    with config_path_obj.open() as file_obj:
        data = yaml.safe_load(file_obj)

    config = RetargetingConfig()

    hand_data = data.get("hand", {})
    if isinstance(hand_data, str):
        hand_path = config_path_obj.parent / hand_data
        with hand_path.open() as file_obj:
            hand_data = yaml.safe_load(file_obj)

    mjcf_path = Path(hand_data.get("mjcf_path", ""))
    if not mjcf_path.is_absolute():
        mjcf_path = (config_path_obj.parent / mjcf_path).resolve()

    config.hand = HandConfig(
        name=hand_data.get("name", ""),
        mjcf_path=str(mjcf_path),
        urdf_source=hand_data.get("urdf_source", ""),
    )

    retargeting_data = data.get("retargeting", {})
    config.human_vector_pairs = retargeting_data.get("human_vector_pairs", [])
    config.origin_link_names = retargeting_data.get("origin_link_names", [])
    config.task_link_names = retargeting_data.get("task_link_names", [])
    config.origin_link_types = retargeting_data.get(
        "origin_link_types",
        ["body"] * len(config.origin_link_names),
    )
    config.task_link_types = retargeting_data.get(
        "task_link_types",
        ["body"] * len(config.task_link_names),
    )
    config.vector_weights = retargeting_data.get(
        "vector_weights",
        [1.0] * len(config.human_vector_pairs),
    )
    vector_loss_data = retargeting_data.get("vector_loss", {})
    config.vector_loss = VectorLossConfig(
        type=vector_loss_data.get("type", "direction"),
        huber_delta=vector_loss_data.get("huber_delta", 0.02),
        scaling=vector_loss_data.get("scaling", 1.0),
        scale_landmarks=vector_loss_data.get("scale_landmarks", [0, 9]),
        scale_bodies=vector_loss_data.get("scale_bodies", ["world", "middle_proximal"]),
        scale_body_types=vector_loss_data.get("scale_body_types", ["body", "body"]),
    )

    config.angle_constraints = [
        AngleConstraint(
            landmarks=item["landmarks"],
            joint=item["joint"],
            weight=item.get("weight", 1.0),
            scale=item.get("scale", 1.0),
            invert=item.get("invert", False),
        )
        for item in retargeting_data.get("angle_constraints", [])
    ]

    position_data = retargeting_data.get("position_constraints", {})
    if position_data and position_data.get("enabled", False):
        config.position = PositionConfig(
            enabled=True,
            weight=position_data.get("weight", 8.0),
            scale_landmarks=position_data.get("scale_landmarks", [0, 9]),
            scale_bodies=position_data.get("scale_bodies", ["world", "middle_proximal"]),
            scale_body_types=position_data.get("scale_body_types", ["body", "body"]),
            constraints=[
                PositionConstraint(
                    landmark=item["landmark"],
                    body=item["body"],
                    body_type=item.get("body_type", "body"),
                    weight=item.get("weight", 1.0),
                )
                for item in position_data.get("constraints", [])
            ],
        )

    pinch_data = retargeting_data.get("pinch", {})
    if pinch_data:
        config.pinch = PinchConfig(
            enabled=pinch_data.get("enabled", False),
            d1=pinch_data.get("d1", 0.03),
            d2=pinch_data.get("d2", 0.06),
            weight=pinch_data.get("weight", 5.0),
            thumb_weight_boost=pinch_data.get("thumb_weight_boost", 1.5),
            fingertip_sites=pinch_data.get("fingertip_sites", []),
        )

    preprocess_data = data.get("retargeting", {}).get("preprocess", {})
    config.preprocess = PreprocessConfig(
        **{
            key: value
            for key, value in preprocess_data.items()
            if key in PreprocessConfig.__dataclass_fields__
        }
    )

    solver_data = data.get("retargeting", {}).get("solver", {})
    config.solver = SolverConfig(
        **{
            key: value
            for key, value in solver_data.items()
            if key in SolverConfig.__dataclass_fields__
        }
    )

    config.validate()
    return config
