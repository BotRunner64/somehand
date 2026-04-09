"""Infrastructure adapters for external systems."""

from .artifacts import (
    load_bihand_recording_artifact,
    load_hand_recording_artifact,
    save_bihand_recording_artifact,
    save_hand_recording_artifact,
    save_trajectory_artifact,
)
from .config_loader import load_retargeting_config
from .hand_model import HandModel
from .model_name_resolver import ModelNameResolver
from .preview import OpenCvPreviewWindow
from .sinks import (
    AsyncLandmarkOutputSink,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    RobotHandOutputSink,
    RobotHandVideoOutputSink,
    TrajectoryRecorder,
)
from .sources import (
    BiHCMocapInputSource,
    BiHandMediaPipeInputSource,
    BiHandPicoInputSource,
    HCMocapInputSource,
    MediaPipeInputSource,
    RecordedBiHandDataSource,
    RecordedHandDataSource,
    RecordingBiHandTrackingSource,
    RecordingHandTrackingSource,
    create_bihand_hc_mocap_udp_source,
    create_bihand_pico_source,
    create_bihand_recording_source,
    create_hc_mocap_udp_source,
    create_pico_source,
    create_recording_source,
)
from .terminal_controls import TerminalRecordingController
from .vector_solver import VectorRetargeter

__all__ = [
    "AsyncLandmarkOutputSink",
    "BiHCMocapInputSource",
    "BiHandMediaPipeInputSource",
    "BiHandOutputWindowSink",
    "BiHandPicoInputSource",
    "BiHandVideoOutputSink",
    "HCMocapInputSource",
    "HandModel",
    "MediaPipeInputSource",
    "ModelNameResolver",
    "OpenCvPreviewWindow",
    "RecordedBiHandDataSource",
    "RecordedHandDataSource",
    "RecordingBiHandTrackingSource",
    "RecordingHandTrackingSource",
    "RobotHandOutputSink",
    "RobotHandVideoOutputSink",
    "TrajectoryRecorder",
    "TerminalRecordingController",
    "VectorRetargeter",
    "create_bihand_hc_mocap_udp_source",
    "create_bihand_pico_source",
    "create_bihand_recording_source",
    "create_hc_mocap_udp_source",
    "create_pico_source",
    "create_recording_source",
    "load_bihand_recording_artifact",
    "load_hand_recording_artifact",
    "load_retargeting_config",
    "save_bihand_recording_artifact",
    "save_hand_recording_artifact",
    "save_trajectory_artifact",
]
