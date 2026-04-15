"""YAML parsing and filesystem resolution for retargeting configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from somehand.domain.config import (
    AngleConstraint,
    BiHandRetargetingConfig,
    BiHandViewerConfig,
    ControllerConfig,
    DistanceConstraint,
    FrameConstraint,
    HandConfig,
    PreprocessConfig,
    RetargetingConfig,
    SolverConfig,
    VectorConstraint,
    VectorLossConfig,
)
from somehand.domain.hand_side import normalize_hand_side
from somehand.infrastructure.universal_config import apply_universal_preset


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
    controller_data = data.get("controller", {})
    config.controller = ControllerConfig(
        backend=str(controller_data.get("backend", "viewer")),
        model_family=str(controller_data.get("model_family", "")),
        control_rate_hz=int(controller_data.get("control_rate_hz", 100)),
        sim_rate_hz=int(controller_data.get("sim_rate_hz", 500)),
        transport=str(controller_data.get("transport", "can")),
        can_interface=str(controller_data.get("can_interface", "can0")),
        modbus_port=str(controller_data.get("modbus_port", "None")),
        sdk_root=str(controller_data.get("sdk_root", "")),
        default_speed=[int(value) for value in controller_data.get("default_speed", [])],
        default_torque=[int(value) for value in controller_data.get("default_torque", [])],
    )

    retargeting_data = data.get("retargeting", {})
    config.preset = str(retargeting_data.get("preset", ""))
    legacy_vector_keys = {
        "human_vector_pairs",
        "origin_link_names",
        "task_link_names",
        "origin_link_types",
        "task_link_types",
        "vector_weights",
    }
    legacy_keys_present = sorted(key for key in legacy_vector_keys if key in retargeting_data)
    if legacy_keys_present:
        raise ValueError(
            "retargeting legacy vector schema is no longer supported; "
            f"use vector_constraints instead of {', '.join(legacy_keys_present)}"
        )
    config.vector_constraints = [
        VectorConstraint(
            human=[int(value) for value in item["human"]],
            robot=[str(value) for value in item["robot"]],
            robot_types=[str(value) for value in item.get("robot_types", ["body", "body"])],
            weight=float(item.get("weight", 1.0)),
            loss_type=str(item.get("loss_type", "")),
            loss_scale=float(item.get("loss_scale", 0.0)),
            optional=bool(item.get("optional", False)),
        )
        for item in retargeting_data.get("vector_constraints", [])
    ]
    config.distance_constraints = [
        DistanceConstraint(
            human=[int(value) for value in item["human"]],
            robot=[str(value) for value in item["robot"]],
            robot_types=[str(value) for value in item.get("robot_types", ["site", "site"])],
            weight=float(item.get("weight", 1.0)),
            scale=float(item.get("scale", 1.0)),
            threshold=float(item.get("threshold", 0.04)),
            activation_type=str(item.get("activation_type", "gaussian")),
            scale_mode=str(item.get("scale_mode", "raw")),
            optional=bool(item.get("optional", False)),
        )
        for item in retargeting_data.get("distance_constraints", [])
    ]
    config.frame_constraints = [
        FrameConstraint(
            name=str(item.get("name", "")),
            human_origin=int(item["human_origin"]),
            human_primary=int(item["human_primary"]),
            human_secondary=int(item["human_secondary"]),
            robot_origin=str(item["robot_origin"]),
            robot_primary=str(item["robot_primary"]),
            robot_secondary=str(item["robot_secondary"]),
            robot_types=[str(value) for value in item.get("robot_types", ["body", "body", "body"])],
            primary_weight=float(item.get("primary_weight", 1.0)),
            secondary_weight=float(item.get("secondary_weight", 1.0)),
            optional=bool(item.get("optional", False)),
        )
        for item in retargeting_data.get("frame_constraints", [])
    ]
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
            optional=bool(item.get("optional", False)),
        )
        for item in retargeting_data.get("angle_constraints", [])
    ]
    if config.preset == "universal":
        if any(
            retargeting_data.get(key)
            for key in ("vector_constraints", "distance_constraints", "frame_constraints", "angle_constraints")
        ):
            raise ValueError("retargeting.preset cannot be combined with explicit constraints")
        apply_universal_preset(config)
    if "position_constraints" in retargeting_data:
        raise ValueError("retargeting.position_constraints is no longer supported")
    if "pinch" in retargeting_data:
        raise ValueError("retargeting.pinch is no longer supported")

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
