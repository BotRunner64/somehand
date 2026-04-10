"""MuJoCo simulation controller backend."""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

from dex_mujoco.domain import HandCommand, HandState
from dex_mujoco.infrastructure.hand_model import HandModel


class MujocoSimController:
    """Runs a fixed-rate MuJoCo simulation against target joint positions."""

    def __init__(
        self,
        mjcf_path: str,
        *,
        control_rate_hz: int = 100,
        sim_rate_hz: int = 500,
    ):
        self._hand_model = HandModel(mjcf_path)
        self._actuator_qpos_indices = self._hand_model.get_actuator_qpos_indices()
        self._control_rate_hz = int(control_rate_hz)
        self._sim_rate_hz = int(sim_rate_hz)
        self._control_interval_steps = max(int(round(self._sim_rate_hz / max(self._control_rate_hz, 1))), 1)
        self._lock = threading.Lock()
        self._target_qpos = self._hand_model.get_qpos()
        self._cached_state = HandState(
            measured_qpos_rad=self._hand_model.get_qpos(),
            measured_qvel=np.zeros(self._hand_model.nv, dtype=np.float64),
            applied_ctrl=np.zeros(self._hand_model.nu, dtype=np.float64),
            sim_time=float(self._hand_model.data.time),
            faults=None,
            contacts=[],
            backend="sim",
        )
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="dex-mujoco-sim-controller", daemon=True)
        self._thread.start()

    def set_command(self, command: HandCommand) -> None:
        with self._lock:
            self._target_qpos = np.asarray(command.target_qpos_rad, dtype=np.float64).copy()

    def get_state(self) -> HandState:
        with self._lock:
            state = self._cached_state
            return HandState(
                measured_qpos_rad=None if state.measured_qpos_rad is None else state.measured_qpos_rad.copy(),
                measured_qvel=None if state.measured_qvel is None else state.measured_qvel.copy(),
                applied_ctrl=None if state.applied_ctrl is None else state.applied_ctrl.copy(),
                sim_time=state.sim_time,
                faults=None if state.faults is None else list(state.faults),
                contacts=None if state.contacts is None else list(state.contacts),
                backend=state.backend,
            )

    def close(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        step = 0
        sleep_s = 1.0 / max(self._sim_rate_hz, 1)
        while self._running:
            tic = time.monotonic()
            with self._lock:
                target_qpos = self._target_qpos.copy()
            if step % self._control_interval_steps == 0:
                ctrl = target_qpos[self._actuator_qpos_indices].copy()
                low = self._hand_model.model.actuator_ctrlrange[:, 0]
                high = self._hand_model.model.actuator_ctrlrange[:, 1]
                np.clip(ctrl, low, high, out=ctrl)
                self._hand_model.data.ctrl[:] = ctrl
            mujoco.mj_step(self._hand_model.model, self._hand_model.data)
            with self._lock:
                self._cached_state = HandState(
                    measured_qpos_rad=self._hand_model.data.qpos.copy(),
                    measured_qvel=self._hand_model.data.qvel.copy(),
                    applied_ctrl=self._hand_model.data.ctrl.copy(),
                    sim_time=float(self._hand_model.data.time),
                    faults=None,
                    contacts=[],
                    backend="sim",
                )
            step += 1
            elapsed = time.monotonic() - tic
            if sleep_s > elapsed:
                time.sleep(sleep_s - elapsed)
