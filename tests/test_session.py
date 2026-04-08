import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.application.session import RetargetingSession
from dex_mujoco.domain.models import HandFrame, RetargetingStepResult, SourceFrame


class _FakeEngine:
    def process(self, frame: HandFrame) -> RetargetingStepResult:
        return RetargetingStepResult(
            qpos=np.array([1.0, 2.0], dtype=np.float64),
            target_directions=None,
            processed_landmarks=frame.retarget_landmarks.copy(),
            handedness=frame.handedness,
        )


class _FakeSource:
    def __init__(self, frames):
        self.source_desc = "fake://source"
        self._frames = list(frames)
        self._index = 0
        self.closed = False

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> SourceFrame:
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    def stats_snapshot(self):
        return {}


class _LiveFakeSource(_FakeSource):
    def __init__(self, frames, *, updates: int = 8, period_s: float = 0.01):
        super().__init__(frames)
        self._latest_index = 0
        self._latest_frame = None
        self._running = True
        self._updates = updates
        self._period_s = period_s
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def latest_hand_frame_snapshot(self):
        frame = self._latest_frame
        if frame is None:
            return None
        return self._latest_index, frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        super().close()

    def _update_loop(self) -> None:
        for index in range(1, self._updates + 1):
            if not self._running:
                break
            self._latest_index = index
            self._latest_frame = _detection("Right")
            time.sleep(self._period_s)


class _FakeSink:
    def __init__(self):
        self.results = []
        self.closed = False

    @property
    def is_running(self) -> bool:
        return True

    def on_result(self, result: RetargetingStepResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


class _FakeFrameSink:
    def __init__(self):
        self.frames = []
        self.closed = False

    @property
    def is_running(self) -> bool:
        return True

    def on_frame(self, frame: HandFrame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class _FakePreview:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def show(self, source, frame: SourceFrame) -> bool:
        self.calls += 1
        return True

    def close(self) -> None:
        self.closed = True


def _detection(handedness: str = "Right") -> HandFrame:
    landmarks = np.zeros((21, 3), dtype=np.float64)
    return HandFrame(
        landmarks_3d=landmarks,
        landmarks_2d=None,
        handedness=handedness,
    )


def test_session_runs_source_engine_and_sinks():
    source = _FakeSource(
        [
            SourceFrame(detection=_detection("Right")),
            SourceFrame(detection=None),
            SourceFrame(detection=_detection("Left")),
        ]
    )
    sink = _FakeSink()
    preview = _FakePreview()
    session = RetargetingSession(_FakeEngine(), sinks=[sink], preview_window=preview)

    summary = session.run(source, input_type="test")

    assert summary.source_desc == "fake://source"
    assert summary.input_type == "test"
    assert summary.num_frames == 3
    assert summary.num_detected == 2
    assert len(sink.results) == 2
    assert sink.results[1].handedness == "Left"
    assert source.closed is True
    assert sink.closed is True
    assert preview.closed is True
    assert preview.calls == 3


def test_session_feeds_frame_sinks_inline_without_snapshot_support():
    source = _FakeSource([SourceFrame(detection=_detection("Right"))])
    frame_sink = _FakeFrameSink()

    session = RetargetingSession(_FakeEngine(), frame_sinks=[frame_sink])
    summary = session.run(source, input_type="test")

    assert summary.num_detected == 1
    assert len(frame_sink.frames) == 1
    assert frame_sink.closed is True


def test_session_decouples_frame_sinks_with_live_snapshot_source():
    class _SlowEngine(_FakeEngine):
        def process(self, frame: HandFrame) -> RetargetingStepResult:
            time.sleep(0.12)
            return super().process(frame)

    source = _LiveFakeSource([SourceFrame(detection=_detection("Right"))], updates=10, period_s=0.01)
    result_sink = _FakeSink()
    frame_sink = _FakeFrameSink()

    session = RetargetingSession(_SlowEngine(), sinks=[result_sink], frame_sinks=[frame_sink])
    summary = session.run(source, input_type="test")

    assert summary.num_detected == 1
    assert len(result_sink.results) == 1
    assert len(frame_sink.frames) >= 2
    assert frame_sink.closed is True


def test_session_stops_when_stop_condition_is_triggered():
    source = _FakeSource(
        [
            SourceFrame(detection=_detection("Right")),
            SourceFrame(detection=_detection("Left")),
        ]
    )
    sink = _FakeSink()
    session = RetargetingSession(_FakeEngine(), sinks=[sink])

    summary = session.run(
        source,
        input_type="test",
        stop_condition=lambda: len(sink.results) >= 1,
    )

    assert summary.num_frames == 1
    assert summary.num_detected == 1
    assert len(sink.results) == 1


def test_session_ignores_interrupts_raised_while_closing_resources():
    class _InterruptingSource(_FakeSource):
        def close(self) -> None:
            raise KeyboardInterrupt

    class _InterruptingSink(_FakeSink):
        def close(self) -> None:
            raise KeyboardInterrupt

    source = _InterruptingSource([SourceFrame(detection=_detection("Right"))])
    sink = _InterruptingSink()
    session = RetargetingSession(_FakeEngine(), sinks=[sink])

    summary = session.run(source, input_type="test")

    assert summary.num_frames == 1
    assert summary.num_detected == 1
