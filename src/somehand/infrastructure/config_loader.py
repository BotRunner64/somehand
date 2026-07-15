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
)
from somehand.domain.hand_side import normalize_hand_side
from somehand.external_assets import resolve_asset_path
from somehand.paths import CONFIG_ROOT
from somehand.runtime.config_validation import validate_runtime_bihand_config, validate_runtime_retargeting_config


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


def _constraint_defaults(retargeting_data: dict, name: str) -> dict:
    defaults = retargeting_data.get("constraint_defaults", {})
    if not isinstance(defaults, dict):
        return {}
    section = defaults.get(name, {})
    if not isinstance(section, dict):
        return {}
    return section


def _human_pair_key(values: list[int]) -> str:
    return f"{values[0]},{values[1]}"


def _vector_constraint_weight(item: dict, robot_types: list[str], defaults: dict) -> float:
    if "weight" in item:
        return float(item["weight"])
    if len(robot_types) == 2 and robot_types[1] == "site":
        return float(defaults.get("terminal_weight", defaults.get("weight", 1.0)))
    return float(defaults.get("weight", 1.0))


def _distance_constraint_weight(item: dict, human: list[int], defaults: dict) -> float:
    if "weight" in item:
        return float(item["weight"])
    weights_by_human = defaults.get("weights_by_human", {})
    if isinstance(weights_by_human, dict):
        weight = weights_by_human.get(_human_pair_key(human))
        if weight is not None:
            return float(weight)
    return float(defaults.get("weight", 1.0))


def _build_vector_constraint(item: dict, defaults: dict) -> VectorConstraint:
    robot_types = [str(value) for value in item.get("robot_types", ["body", "body"])]
    return VectorConstraint(
        human=[int(value) for value in item["human"]],
        robot=[str(value) for value in item["robot"]],
        robot_types=robot_types,
        weight=_vector_constraint_weight(item, robot_types, defaults),
        optional=bool(item.get("optional", False)),
    )


def _build_distance_constraint(item: dict, defaults: dict) -> DistanceConstraint:
    human = [int(value) for value in item["human"]]
    return DistanceConstraint(
        human=human,
        robot=[str(value) for value in item["robot"]],
        robot_types=[str(value) for value in item.get("robot_types", ["site", "site"])],
        weight=_distance_constraint_weight(item, human, defaults),
        scale=float(item.get("scale", defaults.get("scale", 1.0))),
        threshold=float(item.get("threshold", defaults.get("threshold", 0.04))),
        activation_type=str(item.get("activation_type", defaults.get("activation_type", "gaussian"))),
        scale_mode=str(item.get("scale_mode", defaults.get("scale_mode", "raw"))),
        optional=bool(item.get("optional", False)),
    )


def _build_frame_constraint(item: dict, defaults: dict) -> FrameConstraint:
    return FrameConstraint(
        name=str(item.get("name", "")),
        human_origin=int(item["human_origin"]),
        human_primary=int(item["human_primary"]),
        human_secondary=int(item["human_secondary"]),
        robot_origin=str(item["robot_origin"]),
        robot_primary=str(item["robot_primary"]),
        robot_secondary=str(item["robot_secondary"]),
        robot_types=[str(value) for value in item.get("robot_types", ["body", "body", "body"])],
        primary_weight=float(item.get("primary_weight", defaults.get("primary_weight", 1.0))),
        secondary_weight=float(item.get("secondary_weight", defaults.get("secondary_weight", 1.0))),
        optional=bool(item.get("optional", False)),
    )


def _resolve_mjcf_path(config_path_obj: Path, value: str) -> Path:
    mjcf_path = Path(value)
    if mjcf_path.is_absolute():
        return mjcf_path

    resolved = (config_path_obj.parent / mjcf_path).resolve()
    try:
        config_path_obj.resolve().relative_to(CONFIG_ROOT.resolve())
    except ValueError:
        return resolved

    relative_parts = [part for part in mjcf_path.parts if part not in {".", ".."}]
    if "assets" not in relative_parts:
        return resolved
    assets_index = relative_parts.index("assets")
    return resolve_asset_path(Path(*relative_parts[assets_index:])).resolve()


def load_retargeting_config(config_path: str) -> RetargetingConfig:
    config_path_obj = Path(config_path)
    data = _load_yaml_with_extends(config_path_obj)

    config = RetargetingConfig()

    hand_data = data.get("hand", {})
    if isinstance(hand_data, str):
        hand_path = config_path_obj.parent / hand_data
        with hand_path.open() as file_obj:
            hand_data = yaml.safe_load(file_obj)

    mjcf_path = _resolve_mjcf_path(config_path_obj, str(hand_data.get("mjcf_path", "")))

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
    if config.preset:
        raise ValueError("retargeting.preset is no longer supported; define explicit constraints in the hand config")
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
    for item in retargeting_data.get("vector_constraints", []):
        removed_keys = sorted(key for key in ("loss_type", "loss_scale") if key in item)
        if removed_keys:
            raise ValueError(
                "scaled keyvector residual loss is no longer supported; "
                f"remove vector constraint keys: {', '.join(removed_keys)}"
            )
    vector_defaults = _constraint_defaults(retargeting_data, "vector")
    config.vector_constraints = [
        _build_vector_constraint(item, vector_defaults)
        for item in retargeting_data.get("vector_constraints", [])
    ]
    distance_defaults = _constraint_defaults(retargeting_data, "distance")
    config.distance_constraints = [
        _build_distance_constraint(item, distance_defaults)
        for item in retargeting_data.get("distance_constraints", [])
    ]
    frame_defaults = _constraint_defaults(retargeting_data, "frame")
    config.frame_constraints = [
        _build_frame_constraint(item, frame_defaults)
        for item in retargeting_data.get("frame_constraints", [])
    ]
    if "vector_loss" in retargeting_data:
        raise ValueError("retargeting.vector_loss is no longer supported")

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
    validate_runtime_retargeting_config(config)
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
    validate_runtime_bihand_config(config)
    return config
