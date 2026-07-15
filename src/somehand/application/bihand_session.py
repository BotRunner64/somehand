"""Session-level orchestration for bi-hand sources, engine, preview, and sinks."""

from __future__ import annotations

import signal
import time
from collections.abc import Sequence
from threading import Event, Thread
from typing import Callable

from somehand.domain import (
    BiHandFrameSink,
    BiHandOutputSink,
    BiHandSessionSummary,
    BiHandTrackingSource,
    PreviewWindow,
)

from .bihand_engine import BiHandRetargetingEngine
from .session import _close_resource, _start_frame_sink_thread as _start_frame_sink_thread_impl


class BiHandRetargetingSession:
    """Runs the main bi-hand input -> engine -> sink loop."""

    def __init__(
        self,
        engine: BiHandRetargetingEngine,
        *,
        sinks: Sequence[BiHandOutputSink] = (),
        frame_sinks: Sequence[BiHandFrameSink] = (),
        preview_window: PreviewWindow | None = None,
    ):
        self.engine = engine
        self.sinks = list(sinks)
        self.frame_sinks = list(frame_sinks)
        self.preview_window = preview_window

    @property
    def is_running(self) -> bool:
        return all(sink.is_running for sink in [*self.sinks, *self.frame_sinks])

    def run(
        self,
        source: BiHandTrackingSource,
        *,
        input_type: str,
        realtime: bool = False,
        loop: bool = False,
        stats_every: int = 0,
        stop_condition: Callable[[], bool] | None = None,
    ) -> BiHandSessionSummary:
        frame_count = 0
        detected_count = 0
        detected_left = 0
        detected_right = 0
        detected_both = 0
        frame_period = 1.0 / max(source.fps, 1)
        frame_sink_stop = Event()
        frame_sink_thread = self._start_frame_sink_thread(source, stop_event=frame_sink_stop)

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
                detection = frame.detection

                if detection is not None and detection.has_detection:
                    if frame_sink_thread is None:
                        for sink in self.frame_sinks:
                            sink.on_frame(detection)

                    left_detected = detection.left is not None
                    right_detected = detection.right is not None
                    detected_count += 1
                    detected_left += int(left_detected)
                    detected_right += int(right_detected)
                    detected_both += int(left_detected and right_detected)

                    result = self.engine.process(detection)
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
            if self.preview_window is not None:
                _close_resource(self.preview_window)
            for sink in reversed(self.frame_sinks):
                _close_resource(sink)
            for sink in reversed(self.sinks):
                _close_resource(sink)
            signal.signal(signal.SIGINT, previous_sigint_handler)

        return BiHandSessionSummary(
            num_frames=frame_count,
            num_detected=detected_count,
            num_detected_left=detected_left,
            num_detected_right=detected_right,
            num_detected_both=detected_both,
            source_desc=source.source_desc,
            input_type=input_type,
        )

    def _start_frame_sink_thread(
        self,
        source: BiHandTrackingSource,
        *,
        stop_event: Event,
    ) -> Thread | None:
        return _start_frame_sink_thread_impl(
            source,
            self.frame_sinks,
            lambda: self.is_running,
            stop_event=stop_event,
            snapshot_attr_name="latest_bihand_frame_snapshot",
            thread_name="somehand-bihand-frame-sink",
        )
