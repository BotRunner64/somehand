"""Terminal keyboard controls for interactive recording sessions."""

from __future__ import annotations

import io
import os
import select
import sys
import termios
import tty
from threading import Event, Thread

from .sources import RecordingHandTrackingSource


class TerminalRecordingController:
    """Starts/stops recording from terminal key presses."""

    def __init__(
        self,
        recording_source: RecordingHandTrackingSource,
        *,
        start_key: str = "r",
        stop_key: str = "s",
        input_stream=None,
    ):
        self._recording_source = recording_source
        self._start_key = start_key.lower()
        self._stop_key = stop_key.lower()
        self._input_stream = sys.stdin if input_stream is None else input_stream
        self._close_event = Event()
        self._stop_requested = Event()
        self._thread: Thread | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def start(self) -> bool:
        isatty = getattr(self._input_stream, "isatty", None)
        if not callable(isatty) or not isatty():
            print("Interactive recording controls unavailable on this stdin; recording starts immediately.")
            self._recording_source.start_recording()
            return False

        try:
            fileno = self._input_stream.fileno()
        except (AttributeError, io.UnsupportedOperation, OSError):
            print("Interactive recording controls unavailable on this stdin; recording starts immediately.")
            self._recording_source.start_recording()
            return False

        self._thread = Thread(target=self._run, args=(fileno,), name="dex-mujoco-terminal-controls", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        self._close_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def handle_keypress(self, key: str) -> None:
        normalized = key.lower()
        if normalized == self._start_key:
            if self._recording_source.is_recording:
                print("Recording is already active.")
                return
            self._recording_source.start_recording()
            print("Recording started.")
            return

        if normalized == self._stop_key:
            if self._recording_source.is_recording:
                self._recording_source.stop_recording()
                print("Recording stopped. Saving and exiting...")
            else:
                print("Stop requested. Exiting...")
            self._stop_requested.set()

    def _run(self, fileno: int) -> None:
        try:
            original_attrs = termios.tcgetattr(fileno)
        except termios.error:
            print("Interactive recording controls unavailable on this terminal; recording starts immediately.")
            self._recording_source.start_recording()
            return

        try:
            tty.setcbreak(fileno)
            while not self._close_event.is_set() and not self._stop_requested.is_set():
                readable, _, _ = select.select([fileno], [], [], 0.1)
                if not readable:
                    continue

                data = os.read(fileno, 1)
                if not data:
                    continue

                key = data.decode(errors="ignore")
                if key:
                    self.handle_keypress(key)
        finally:
            termios.tcsetattr(fileno, termios.TCSADRAIN, original_attrs)
