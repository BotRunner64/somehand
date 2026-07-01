"""Async process-backed viewer wrappers."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import time
from multiprocessing.context import BaseContext
from pathlib import Path

import numpy as np

from somehand.infrastructure.hand_model import HandModel

from .viewer_landmarks import BiHandLandmarkVisualizer, LandmarkVisualizer
from .viewer_hand import HandVisualizer
from .vector_visualization import VectorPair

VIEWER_LOOP_PERIOD_S = 1.0 / 120.0


def _resolve_mjpython_executable() -> str | None:
    candidates = []

    mjpython_bin = os.environ.get("MJPYTHON_BIN")
    if mjpython_bin:
        candidates.append(Path(mjpython_bin))

    candidates.append(Path(sys.executable).with_name("mjpython"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mjpython")

    path_executable = shutil.which("mjpython")
    if path_executable:
        candidates.append(Path(path_executable))

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return resolved
    return None


def _viewer_spawn_context() -> BaseContext:
    ctx = mp.get_context("spawn")
    if sys.platform == "darwin":
        mjpython = _resolve_mjpython_executable()
        if mjpython is not None:
            ctx.set_executable(mjpython)
    return ctx


def landmark_viewer_worker(
    frame_queue: mp.queues.Queue,
    window_title: str | None,
    vector_pairs: list[VectorPair],
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    visualizer = LandmarkVisualizer(window_title=window_title, vector_pairs=vector_pairs)
    latest_landmarks = np.zeros((21, 3), dtype=np.float64)

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                latest_landmarks = np.asarray(item, dtype=np.float64)
                drained = True

            if drained:
                visualizer.update(latest_landmarks)
            else:
                visualizer.update(latest_landmarks)
                time.sleep(VIEWER_LOOP_PERIOD_S)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncProcessHandle:
    """Manages a one-item mp.Queue connected to a worker process."""

    def __init__(self, process: mp.Process, queue: "mp.queues.Queue") -> None:
        self._process = process
        self._queue = queue

    @property
    def is_running(self) -> bool:
        return self._process.is_alive()

    def send(self, payload: object) -> None:
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return
        self.send(None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


class AsyncLandmarkVisualizer:
    """Landmark viewer running in a separate process for stability."""

    def __init__(self, *, window_title: str | None = None, vector_pairs: list[VectorPair] | None = None):
        ctx = _viewer_spawn_context()
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=landmark_viewer_worker,
            args=(self._queue, window_title, [] if vector_pairs is None else [tuple(pair) for pair in vector_pairs]),
            name="somehand-landmark-viewer",
        )
        self._process.start()
        self._handle = AsyncProcessHandle(self._process, self._queue)

    @property
    def is_running(self) -> bool:
        return self._handle.is_running

    def update(self, landmarks: np.ndarray) -> None:
        self._handle.send(np.asarray(landmarks, dtype=np.float64))

    def close(self) -> None:
        self._handle.close()


def robot_hand_viewer_worker(
    mjcf_path: str,
    qpos_queue: mp.queues.Queue,
    overlay_label: str | None,
    window_title: str | None,
    viewer_mode: str,
    hand_side: str | None,
    robot_vector_specs: list[tuple[int, str, str, str, str]],
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    hand_model = HandModel(mjcf_path)
    visualizer = HandVisualizer(
        hand_model,
        overlay_label=overlay_label,
        window_title=window_title,
        viewer_mode=viewer_mode,
        hand_side=hand_side,
        robot_vector_specs=robot_vector_specs,
    )
    latest_qpos = hand_model.get_qpos()
    latest_target_directions = None

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = qpos_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                if isinstance(item, dict):
                    latest_qpos = np.asarray(item["qpos"], dtype=np.float64)
                    latest_target_directions = _optional_array(item.get("target_directions"))
                else:
                    latest_qpos = np.asarray(item, dtype=np.float64)
                    latest_target_directions = None
                drained = True

            if drained:
                visualizer.update(latest_qpos, target_directions=latest_target_directions)
            else:
                visualizer.update(latest_qpos, target_directions=latest_target_directions)
                time.sleep(VIEWER_LOOP_PERIOD_S)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncRobotHandVisualizer:
    """Robot-hand viewer running in a separate process for stability."""

    def __init__(
        self,
        mjcf_path: str,
        *,
        overlay_label: str | None = None,
        window_title: str | None = None,
        viewer_mode: str = "normal",
        hand_side: str | None = None,
        robot_vector_specs: list[tuple[int, str, str, str, str]] | None = None,
    ):
        ctx = _viewer_spawn_context()
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=robot_hand_viewer_worker,
            args=(
                mjcf_path,
                self._queue,
                overlay_label,
                window_title,
                viewer_mode,
                hand_side,
                [] if robot_vector_specs is None else list(robot_vector_specs),
            ),
            name="somehand-robot-hand-viewer",
        )
        self._process.start()
        self._handle = AsyncProcessHandle(self._process, self._queue)

    @property
    def is_running(self) -> bool:
        return self._handle.is_running

    def update(self, qpos: np.ndarray) -> None:
        self._handle.send(np.asarray(qpos, dtype=np.float64))

    def update_with_vectors(self, qpos: np.ndarray, target_directions: np.ndarray | None) -> None:
        self._handle.send(
            {
                "qpos": np.asarray(qpos, dtype=np.float64),
                "target_directions": _optional_array(target_directions),
            }
        )

    def close(self) -> None:
        self._handle.close()


def bihand_landmark_viewer_worker(
    frame_queue: mp.queues.Queue,
    left_vector_pairs: list[VectorPair],
    right_vector_pairs: list[VectorPair],
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    visualizer = BiHandLandmarkVisualizer(
        left_vector_pairs=left_vector_pairs,
        right_vector_pairs=right_vector_pairs,
    )
    latest_landmarks = np.full((2, 21, 3), np.nan, dtype=np.float64)

    try:
        while visualizer.is_running:
            drained = False
            while True:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    return
                latest_landmarks = np.asarray(item, dtype=np.float64)
                drained = True

            if drained:
                visualizer.update(latest_landmarks)
            else:
                visualizer.update(latest_landmarks)
                time.sleep(VIEWER_LOOP_PERIOD_S)
    except KeyboardInterrupt:
        return
    finally:
        visualizer.close()


class AsyncBiHandLandmarkVisualizer:
    """Bi-hand landmark viewer running in a separate process for stability."""

    def __init__(
        self,
        *,
        left_vector_pairs: list[VectorPair] | None = None,
        right_vector_pairs: list[VectorPair] | None = None,
    ):
        ctx = _viewer_spawn_context()
        self._queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=bihand_landmark_viewer_worker,
            args=(
                self._queue,
                [] if left_vector_pairs is None else [tuple(pair) for pair in left_vector_pairs],
                [] if right_vector_pairs is None else [tuple(pair) for pair in right_vector_pairs],
            ),
            name="somehand-bihand-landmark-viewer",
        )
        self._process.start()
        self._handle = AsyncProcessHandle(self._process, self._queue)

    @property
    def is_running(self) -> bool:
        return self._handle.is_running

    def update(self, landmarks: np.ndarray) -> None:
        self._handle.send(np.asarray(landmarks, dtype=np.float64))

    def close(self) -> None:
        self._handle.close()


def _optional_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


__all__ = [
    "VIEWER_LOOP_PERIOD_S",
    "AsyncProcessHandle",
    "AsyncLandmarkVisualizer",
    "AsyncRobotHandVisualizer",
    "AsyncBiHandLandmarkVisualizer",
    "bihand_landmark_viewer_worker",
    "landmark_viewer_worker",
    "robot_hand_viewer_worker",
]
