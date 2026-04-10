"""Session orchestration for retargeting + controller backends."""

from __future__ import annotations

import signal
import time
from collections.abc import Sequence
from threading import Event, Thread
from typing import Callable

from dex_mujoco.domain import (
    ControllerBackend,
    HandCommand,
    HandFrameSink,
    HandTrackingSource,
    OutputSink,
    PreviewWindow,
    RetargetingStepResult,
    SessionSummary,
)

from .engine import RetargetingEngine


class ControlledRetargetingSession:
    """Runs input -> retargeting -> controller -> sink loop."""

    def __init__(
        self,
        engine: RetargetingEngine,
        controller: ControllerBackend,
        *,
        sinks: Sequence[OutputSink] = (),
        frame_sinks: Sequence[HandFrameSink] = (),
        preview_window: PreviewWindow | None = None,
    ):
        self.engine = engine
        self.controller = controller
        self.sinks = list(sinks)
        self.frame_sinks = list(frame_sinks)
        self.preview_window = preview_window

    @property
    def is_running(self) -> bool:
        return self.controller.is_running and all(sink.is_running for sink in [*self.sinks, *self.frame_sinks])

    def run(
        self,
        source: HandTrackingSource,
        *,
        input_type: str,
        realtime: bool = False,
        loop: bool = False,
        stats_every: int = 0,
        stop_condition: Callable[[], bool] | None = None,
    ) -> SessionSummary:
        frame_count = 0
        detected_count = 0
        frame_period = 1.0 / max(source.fps, 1)
        frame_sink_stop = Event()
        frame_sink_thread = self._start_frame_sink_thread(source, stop_event=frame_sink_stop)
        self.controller.start()

        try:
            while True:
                if stop_condition is not None and stop_condition():
                    break
                if not source.is_available():
                    if loop and source.reset():
                        continue
                    break
                tic = time.monotonic()
                try:
                    frame = source.get_frame()
                except StopIteration:
                    break
                frame_count += 1

                if frame.detection is not None:
                    if frame_sink_thread is None:
                        for sink in self.frame_sinks:
                            sink.on_frame(frame.detection)
                    detected_count += 1
                    target = self.engine.process(frame.detection)
                    self.controller.set_command(
                        HandCommand(
                            target_qpos_rad=target.qpos.copy(),
                            hand_model=self.engine.config.hand.name,
                            hand_side=frame.detection.hand_side,
                            timestamp=time.monotonic(),
                            sequence_id=detected_count,
                        )
                    )
                    state = self.controller.get_state()
                    result = RetargetingStepResult(
                        qpos=target.qpos.copy() if state.measured_qpos_rad is None else state.measured_qpos_rad.copy(),
                        target_qpos=target.qpos.copy(),
                        target_directions=target.target_directions,
                        processed_landmarks=target.processed_landmarks,
                        hand_side=target.hand_side,
                        backend=state.backend,
                    )
                    for sink in self.sinks:
                        sink.on_result(result)

                if self.preview_window is not None and not self.preview_window.show(source, frame):
                    break

                if stats_every > 0 and frame_count % stats_every == 0:
                    stats = source.stats_snapshot()
                    if stats:
                        print(
                            "UDP stats:"
                            f" recv={stats.get('packets_received', 0)}"
                            f" valid={stats.get('packets_valid', 0)}"
                            f" bad_size={stats.get('packets_bad_size', 0)}"
                            f" bad_decode={stats.get('packets_bad_decode', 0)}"
                            f" floats={stats.get('last_float_count', 0)}/{stats.get('expected_float_count', 0)}"
                            f" bytes={stats.get('last_packet_bytes', 0)}"
                            f" sender={stats.get('last_sender')}"
                        )

                if stop_condition is not None and stop_condition():
                    break
                if not self.is_running:
                    break
                if realtime:
                    elapsed = time.monotonic() - tic
                    sleep_s = frame_period - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            frame_sink_stop.set()
            if frame_sink_thread is not None:
                frame_sink_thread.join(timeout=1.0)
            _close_resource(source)
            _close_resource(self.preview_window)
            for sink in reversed(self.frame_sinks):
                _close_resource(sink)
            for sink in reversed(self.sinks):
                _close_resource(sink)
            _close_resource(self.controller)
            signal.signal(signal.SIGINT, previous_sigint_handler)

        return SessionSummary(
            num_frames=frame_count,
            num_detected=detected_count,
            source_desc=source.source_desc,
            input_type=input_type,
        )

    def _start_frame_sink_thread(self, source: HandTrackingSource, *, stop_event: Event) -> Thread | None:
        if not self.frame_sinks:
            return None
        snapshot_fn = getattr(source, "latest_hand_frame_snapshot", None)
        if not callable(snapshot_fn):
            return None

        def _worker() -> None:
            last_frame_index = -1
            sleep_s = 1.0 / max(source.fps, 1)
            while not stop_event.is_set():
                snapshot = snapshot_fn()
                if snapshot is not None:
                    frame_index, frame = snapshot
                    if frame_index != last_frame_index:
                        last_frame_index = frame_index
                        for sink in self.frame_sinks:
                            if sink.is_running:
                                sink.on_frame(frame)
                if not self.is_running:
                    break
                time.sleep(sleep_s)

        thread = Thread(target=_worker, name="dex-mujoco-controller-frame-sink", daemon=True)
        thread.start()
        return thread


def _close_resource(resource: object) -> None:
    close_fn = getattr(resource, "close", None)
    if not callable(close_fn):
        return
    try:
        close_fn()
    except BaseException as exc:
        print(f"Warning: failed to close {type(resource).__name__}: {exc}")
