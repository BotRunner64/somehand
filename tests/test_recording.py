import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.domain.models import HandFrame, SourceFrame
from dex_mujoco.infrastructure.artifacts import load_hand_recording_artifact, save_hand_recording_artifact
from dex_mujoco.infrastructure.sources import RecordingHandTrackingSource, create_recording_source
from dex_mujoco.infrastructure.terminal_controls import TerminalRecordingController


def _frame(handedness: str = "Right", *, with_local: bool = False) -> HandFrame:
    landmarks = np.arange(63, dtype=np.float64).reshape(21, 3)
    landmarks_2d = np.arange(42, dtype=np.float64).reshape(21, 2)
    landmarks_3d_local = landmarks * 0.1 if with_local else None
    metadata = {"preprocess_frame_override": "camera_aligned"} if with_local else {}
    return HandFrame(
        landmarks_3d=landmarks,
        landmarks_2d=landmarks_2d,
        handedness=handedness,
        landmarks_3d_local=landmarks_3d_local,
        metadata=metadata,
    )


class _FakeSource:
    def __init__(self, frames):
        self.source_desc = "fake://recording"
        self._frames = list(frames)
        self._index = 0

    @property
    def fps(self) -> int:
        return 25

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> SourceFrame:
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> bool:
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self):
        return {}


def test_recording_wrapper_captures_detected_frames_only():
    wrapped = RecordingHandTrackingSource(
        _FakeSource(
            [
                SourceFrame(detection=_frame("Right")),
                SourceFrame(detection=None),
                SourceFrame(detection=_frame("Left", with_local=True)),
            ]
        )
    )

    while wrapped.is_available():
        wrapped.get_frame()

    assert len(wrapped.recorded_frames) == 2
    assert wrapped.recorded_frames[0].handedness == "Right"
    assert wrapped.recorded_frames[1].handedness == "Left"
    assert wrapped.recorded_frames[1].preprocess_frame_override == "camera_aligned"


def test_recording_wrapper_can_gate_recording_with_explicit_start_stop():
    wrapped = RecordingHandTrackingSource(
        _FakeSource(
            [
                SourceFrame(detection=_frame("Right")),
                SourceFrame(detection=_frame("Left")),
                SourceFrame(detection=_frame("Right", with_local=True)),
            ]
        ),
        recording_enabled=False,
    )

    wrapped.get_frame()
    wrapped.start_recording()
    wrapped.get_frame()
    wrapped.stop_recording()
    wrapped.get_frame()

    assert len(wrapped.recorded_frames) == 1
    assert wrapped.recorded_frames[0].handedness == "Left"


def test_terminal_recording_controller_responds_to_start_and_stop_keys():
    wrapped = RecordingHandTrackingSource(_FakeSource([]), recording_enabled=False)
    controller = TerminalRecordingController(wrapped)

    controller.handle_keypress("r")
    assert wrapped.is_recording is True
    assert controller.stop_requested is False

    controller.handle_keypress("s")
    assert wrapped.is_recording is False
    assert controller.stop_requested is True


def test_terminal_recording_controller_stop_requested_is_callable_for_session():
    wrapped = RecordingHandTrackingSource(_FakeSource([]), recording_enabled=False)
    controller = TerminalRecordingController(wrapped)
    stop_condition = lambda: controller.stop_requested

    assert stop_condition() is False
    controller.handle_keypress("s")
    assert stop_condition() is True


def test_hand_recording_artifact_roundtrip(tmp_path):
    recording_path = tmp_path / "session.pkl"
    frames = [_frame("Right"), _frame("Left", with_local=True)]

    save_hand_recording_artifact(
        str(recording_path),
        frames,
        source_fps=25,
        source_desc="camera://0",
        input_type="webcam",
        num_frames=3,
        handedness="Right",
        num_detected=2,
    )

    payload = load_hand_recording_artifact(str(recording_path))

    assert payload["fps"] == 25
    assert payload["input_source"] == "camera://0"
    assert payload["input_type"] == "webcam"
    assert payload["num_frames"] == 3
    assert payload["num_detected"] == 2
    assert len(payload["frames"]) == 2
    assert np.array_equal(payload["frames"][1].landmarks_3d_local, frames[1].landmarks_3d_local)


def test_recording_source_replays_saved_frames(tmp_path):
    recording_path = tmp_path / "session.pkl"
    frames = [_frame("Right"), _frame("Left")]

    save_hand_recording_artifact(
        str(recording_path),
        frames,
        source_fps=50,
        source_desc="udp://0.0.0.0:1118",
        input_type="hc_mocap",
        num_frames=2,
    )

    source = create_recording_source(recording_path=str(recording_path))
    seen = []

    while source.is_available():
        seen.append(source.get_frame().detection)

    assert source.fps == 50
    assert source.recording_metadata["input_type"] == "hc_mocap"
    assert len(seen) == 2
    assert seen[0] is not frames[0]
    assert np.array_equal(seen[0].landmarks_3d, frames[0].landmarks_3d)
    assert source.reset() is True
    assert source.is_available() is True
