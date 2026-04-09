"""YAML parsing and filesystem resolution for retargeting configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from dex_mujoco.domain.config import (
    AngleConstraint,
    BiHandRetargetingConfig,
    BiHandViewerConfig,
    HandConfig,
    PinchConfig,
    PositionConfig,
    PositionConstraint,
    PreprocessConfig,
    RetargetingConfig,
    SolverConfig,
    VectorLossConfig,
)
from dex_mujoco.domain.hand_side import normalize_hand_side


def _deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _load_yaml_with_extends(config_path_obj: Path) -> dict:
    with config_path_obj.open() as file_obj:
        data = yaml.safe_load(file_obj) or {}

    extends_path = data.pop("extends", None)
    if extends_path is None:
        return data

    base_path = Path(extends_path)
    if not base_path.is_absolute():
        base_path = (config_path_obj.parent / base_path).resolve()
    base_data = _load_yaml_with_extends(base_path)
    merged = _deep_merge(base_data, data)
    if not isinstance(merged, dict):
        raise ValueError(f"Config root must be a mapping: {config_path_obj}")
    return merged


def load_retargeting_config(config_path: str) -> RetargetingConfig:
    config_path_obj = Path(config_path)
    data = _load_yaml_with_extends(config_path_obj)

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
        side=normalize_hand_side(hand_data.get("side", "")),
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


def _resolve_relative_path(config_path_obj: Path, value: str) -> str:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = (config_path_obj.parent / resolved).resolve()
    return str(resolved)


def _extract_nested_config_path(config_path_obj: Path, payload: object, *, side: str) -> str:
    if isinstance(payload, str):
        return _resolve_relative_path(config_path_obj, payload)
    if not isinstance(payload, dict):
        raise ValueError(f"{side} config entry must be a path or mapping")
    nested_path = payload.get("config_path", payload.get("config"))
    if not nested_path:
        raise ValueError(f"{side} config entry must define 'config' or 'config_path'")
    return _resolve_relative_path(config_path_obj, str(nested_path))


def load_bihand_config(config_path: str) -> BiHandRetargetingConfig:
    config_path_obj = Path(config_path)
    data = _load_yaml_with_extends(config_path_obj)

    viewer_data = data.get("viewer", {})
    config = BiHandRetargetingConfig(
        left_config_path=_extract_nested_config_path(config_path_obj, data.get("left", {}), side="left"),
        right_config_path=_extract_nested_config_path(config_path_obj, data.get("right", {}), side="right"),
        viewer=BiHandViewerConfig(
            panel_width=int(viewer_data.get("panel_width", 640)),
            panel_height=int(viewer_data.get("panel_height", 720)),
            window_name=str(viewer_data.get("window_name", "Bi-Hand Retargeting")),
            left_pos=tuple(float(value) for value in viewer_data.get("left_pos", (0.22, 0.04, 0.02))),
            right_pos=tuple(float(value) for value in viewer_data.get("right_pos", (-0.22, 0.04, 0.02))),
            camera_lookat=tuple(float(value) for value in viewer_data.get("camera_lookat", (0.0, 0.04, 0.02))),
            left_quat=tuple(float(value) for value in viewer_data.get("left_quat", (0.69288325, 0.01522078, -0.05862347, 0.71850151))),
            right_quat=tuple(float(value) for value in viewer_data.get("right_quat", (0.71846417, 0.05829359, -0.01490552, 0.69295665))),
        ),
    )
    config.validate()
    return config
