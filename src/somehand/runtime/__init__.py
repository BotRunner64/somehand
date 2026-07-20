"""Canonical runtime namespace for external adapters and validators."""

from __future__ import annotations

from importlib import import_module

from .config_validation import validate_runtime_bihand_config, validate_runtime_retargeting_config

_MANUS_EXPORTS = {
    "BiHandManusRos2InputSource",
    "ManusRos2InputSource",
    "create_bihand_manus_ros2_source",
    "create_manus_ros2_source",
    "manus_message_to_hand_frame",
}

_INFRA_EXPORTS = {
    "AsyncBiHandLandmarkOutputSink",
    "AsyncLandmarkOutputSink",
    "BiHCMocapInputSource",
    "BiHandMediaPipeInputSource",
    "BiHandOutputWindowSink",
    "BiHandPicoInputSource",
    "BiHandVideoOutputSink",
    "HCMocapInputSource",
    "HandModel",
    "LinkerHandModelAdapter",
    "LinkerHandSdkController",
    "MediaPipeInputSource",
    "ModelNameResolver",
    "MujocoSimController",
    "OpenCvPreviewWindow",
    "RecordedBiHandDataSource",
    "RecordedHandDataSource",
    "RecordingBiHandTrackingSource",
    "RecordingHandTrackingSource",
    "RobotHandOutputSink",
    "RobotHandTargetOutputSink",
    "RobotHandVideoOutputSink",
    "TerminalRecordingController",
    "TrajectoryRecorder",
    "VectorRetargeter",
    "create_bihand_hc_mocap_udp_source",
    "create_bihand_pico_source",
    "create_bihand_recording_source",
    "create_hc_mocap_udp_source",
    "create_pico_source",
    "create_recording_source",
    "infer_linkerhand_model_family",
    "load_bihand_config",
    "load_bihand_recording_artifact",
    "load_hand_recording_artifact",
    "load_retargeting_config",
    "save_bihand_recording_artifact",
    "save_hand_recording_artifact",
    "save_trajectory_artifact",
}

__all__ = sorted(
    _INFRA_EXPORTS
    | _MANUS_EXPORTS
    | {
        "validate_runtime_bihand_config",
        "validate_runtime_retargeting_config",
    }
)


def __getattr__(name: str):
    if name == "validate_runtime_bihand_config":
        return validate_runtime_bihand_config
    if name == "validate_runtime_retargeting_config":
        return validate_runtime_retargeting_config
    if name in _MANUS_EXPORTS:
        manus_source = import_module("somehand.runtime.manus_source")
        return getattr(manus_source, name)
    if name in _INFRA_EXPORTS:
        infrastructure = import_module("somehand.infrastructure")
        return getattr(infrastructure, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
