"""LinkerHand SDK controller backend."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from dex_mujoco.domain import HandCommand, HandState
from dex_mujoco.infrastructure.controllers.adapters import LinkerHandModelAdapter
from dex_mujoco.paths import DEFAULT_LINKERHAND_SDK_PATH


@lru_cache(maxsize=4)
def _load_linkerhand_api_class(sdk_root: str):
    root = Path(sdk_root or DEFAULT_LINKERHAND_SDK_PATH).resolve()
    module_path = root / "LinkerHand" / "linker_hand_api.py"
    if not module_path.exists():
        raise FileNotFoundError(f"LinkerHand API module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("dex_mujoco_linkerhand_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LinkerHand API module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module.LinkerHandApi


class LinkerHandSdkController:
    """Sends retargeted pose targets to LinkerHand SDK and polls state back."""

    def __init__(
        self,
        adapter: LinkerHandModelAdapter,
        *,
        transport: str = "can",
        can_interface: str = "can0",
        modbus_port: str = "None",
        default_speed: list[int] | None = None,
        default_torque: list[int] | None = None,
        sdk_root: str = "",
    ):
        self._adapter = adapter
        self._transport = transport
        self._can_interface = can_interface
        self._modbus_port = modbus_port if transport == "modbus" else "None"
        self._default_speed = list(default_speed or adapter.default_speed)
        self._default_torque = list(default_torque or adapter.default_torque)
        self._sdk_root = sdk_root
        self._api = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        api_cls = _load_linkerhand_api_class(self._sdk_root)
        self._api = api_cls(
            hand_type=self._adapter.hand_side,
            hand_joint=self._adapter.family,
            modbus=self._modbus_port,
            can=self._can_interface,
        )
        if self._default_speed:
            self._api.set_speed(self._default_speed)
        if self._default_torque:
            self._api.set_torque(self._default_torque)
        self._running = True

    def set_command(self, command: HandCommand) -> None:
        if self._api is None:
            raise RuntimeError("LinkerHandSdkController has not been started")
        pose = self._adapter.qpos_to_sdk_range(command.target_qpos_rad)
        self._api.finger_move(pose=pose)

    def get_state(self) -> HandState:
        if self._api is None:
            raise RuntimeError("LinkerHandSdkController has not been started")
        pose = self._api.get_state()
        qpos = self._adapter.sdk_range_to_qpos(pose)
        faults = None
        try:
            faults = list(self._api.get_fault())
        except BaseException:
            faults = None
        return HandState(
            measured_qpos_rad=qpos,
            measured_qvel=None,
            applied_ctrl=np.asarray(pose, dtype=np.float64),
            sim_time=None,
            faults=faults,
            contacts=None,
            backend="real",
        )

    def close(self) -> None:
        if self._api is not None:
            close_can = getattr(self._api, "close_can", None)
            if callable(close_can):
                close_can()
        self._running = False
