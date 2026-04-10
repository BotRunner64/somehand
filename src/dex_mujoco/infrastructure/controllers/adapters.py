"""Adapters between MuJoCo joint space and LinkerHand SDK spaces."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from dex_mujoco.infrastructure.hand_model import HandModel
from dex_mujoco.paths import DEFAULT_LINKERHAND_SDK_PATH

_FAMILY_PATTERN = re.compile(r"(o6|l6|l7|l10|l20|l21|l25|g20)", re.IGNORECASE)
_SUPPORTED_FAMILIES = {"O6", "L6", "L7", "L10", "L20", "L21", "L25", "G20"}


def infer_linkerhand_model_family(hand_name: str) -> str:
    match = _FAMILY_PATTERN.search(hand_name)
    if not match:
        raise ValueError(f"Cannot infer LinkerHand model family from hand name: {hand_name}")
    family = match.group(1).upper()
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported LinkerHand model family: {family}")
    return family


@lru_cache(maxsize=4)
def _load_mapping_module(sdk_root: str):
    root = Path(sdk_root or DEFAULT_LINKERHAND_SDK_PATH).resolve()
    mapping_path = root / "LinkerHand" / "utils" / "mapping.py"
    if not mapping_path.exists():
        raise FileNotFoundError(f"LinkerHand mapping module not found: {mapping_path}")
    spec = importlib.util.spec_from_file_location("dex_mujoco_linkerhand_mapping", mapping_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LinkerHand mapping module from: {mapping_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _default_speed_for_family(family: str) -> list[int]:
    if family in {"O6", "L6"}:
        return [180] * 6
    if family == "L7":
        return [180] * 7
    return [180] * 5


def _default_torque_for_family(family: str) -> list[int]:
    if family in {"O6", "L6"}:
        return [180] * 6
    if family == "L7":
        return [180] * 7
    return [200] * 5


@dataclass(slots=True)
class LinkerHandModelAdapter:
    """Maps between MuJoCo qpos and LinkerHand SDK command/state vectors."""

    hand_model: HandModel
    family: str
    hand_side: str
    sdk_root: str = ""
    _joint_index: dict[str, int] = field(init=False, repr=False)
    _mapping_module: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.family = self.family.upper()
        self.hand_side = self.hand_side.lower()
        self._joint_index = self.hand_model.get_joint_name_to_qpos_index()
        self._mapping_module = _load_mapping_module(self.sdk_root)

    @property
    def default_speed(self) -> list[int]:
        return _default_speed_for_family(self.family)

    @property
    def default_torque(self) -> list[int]:
        return _default_torque_for_family(self.family)

    def qpos_to_sdk_range(self, qpos: np.ndarray) -> list[int]:
        sdk_arc = self.qpos_to_sdk_arc(qpos)
        if self.hand_side == "left":
            return [int(round(value)) for value in self._mapping_module.arc_to_range_left(sdk_arc, self.family)]
        return [int(round(value)) for value in self._mapping_module.arc_to_range_right(sdk_arc, self.family)]

    def sdk_range_to_qpos(self, pose: list[int] | np.ndarray) -> np.ndarray:
        values = [int(round(float(value))) for value in pose]
        if self.hand_side == "left":
            sdk_arc = self._mapping_module.range_to_arc_left(values, self.family)
        else:
            sdk_arc = self._mapping_module.range_to_arc_right(values, self.family)
        return self.sdk_arc_to_qpos(np.asarray(sdk_arc, dtype=np.float64))

    def qpos_to_sdk_arc(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape[0] != self.hand_model.nq:
            raise ValueError(f"Expected qpos of shape ({self.hand_model.nq},), got {values.shape}")
        if self.family == "L10":
            return np.asarray(
                [
                    self._joint("thumb_cmc_pitch", values),
                    self._joint("thumb_cmc_roll", values),
                    self._joint("index_mcp_pitch", values),
                    self._joint("middle_mcp_pitch", values),
                    self._joint("ring_mcp_pitch", values),
                    self._joint("pinky_mcp_pitch", values),
                    self._joint("index_mcp_roll", values),
                    self._joint("ring_mcp_roll", values),
                    self._joint("pinky_mcp_roll", values),
                    self._joint("thumb_cmc_yaw", values),
                ],
                dtype=np.float64,
            )
        if self.family in {"L20", "G20"}:
            return np.asarray(
                [
                    self._joint("thumb_cmc_pitch", values),
                    self._joint("index_mcp_pitch", values),
                    self._joint("middle_mcp_pitch", values),
                    self._joint("ring_mcp_pitch", values),
                    self._joint("pinky_mcp_pitch", values),
                    self._joint("thumb_cmc_roll", values),
                    self._joint("index_mcp_roll", values),
                    self._joint("middle_mcp_roll", values),
                    self._joint("ring_mcp_roll", values),
                    self._joint("pinky_mcp_roll", values),
                    self._joint("thumb_cmc_yaw", values),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self._joint("thumb_dip", values),
                    self._joint("index_dip", values),
                    self._joint("middle_dip", values),
                    self._joint("ring_dip", values),
                    self._joint("pinky_dip", values),
                ],
                dtype=np.float64,
            )
        if self.family == "L21":
            return np.asarray(
                [
                    self._joint("thumb_cmc_pitch", values),
                    self._joint("index_mcp_pitch", values),
                    self._joint("middle_mcp_pitch", values),
                    self._joint("ring_mcp_pitch", values),
                    self._joint("pinky_mcp_pitch", values),
                    self._joint("thumb_cmc_yaw", values),
                    self._joint("index_mcp_roll", values),
                    self._joint("middle_mcp_roll", values),
                    self._joint("ring_mcp_roll", values),
                    self._joint("pinky_mcp_roll", values),
                    self._joint("thumb_cmc_roll", values),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self._joint("thumb_mcp", values),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self._joint("thumb_ip", values),
                    self._joint("index_pip", values),
                    self._joint("middle_pip", values),
                    self._joint("ring_pip", values),
                    self._joint("pinky_pip", values),
                ],
                dtype=np.float64,
            )
        if self.family == "L25":
            return np.asarray(
                [
                    self._joint("thumb_cmc_pitch", values),
                    self._joint("index_mcp_pitch", values),
                    self._joint("middle_mcp_pitch", values),
                    self._joint("ring_mcp_pitch", values),
                    self._joint("pinky_mcp_pitch", values),
                    self._joint("thumb_cmc_yaw", values),
                    self._joint("index_mcp_roll", values),
                    self._joint("middle_mcp_roll", values),
                    self._joint("ring_mcp_roll", values),
                    self._joint("pinky_mcp_roll", values),
                    self._joint("thumb_cmc_roll", values),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self._joint("thumb_mcp", values),
                    self._joint("index_pip", values),
                    self._joint("middle_pip", values),
                    self._joint("ring_pip", values),
                    self._joint("pinky_pip", values),
                    self._joint("thumb_ip", values),
                    self._joint("index_dip", values),
                    self._joint("middle_dip", values),
                    self._joint("ring_dip", values),
                    self._joint("pinky_dip", values),
                ],
                dtype=np.float64,
            )
        raise ValueError(f"SDK adapter not implemented for family: {self.family}")

    def sdk_arc_to_qpos(self, sdk_arc: np.ndarray) -> np.ndarray:
        arc = np.asarray(sdk_arc, dtype=np.float64)
        qpos = np.zeros(self.hand_model.nq, dtype=np.float64)
        if self.family == "L10":
            self._set_joint(qpos, "thumb_cmc_pitch", arc[0])
            self._set_joint(qpos, "thumb_cmc_roll", arc[1])
            self._set_joint(qpos, "thumb_cmc_yaw", arc[9])
            self._set_joint(qpos, "thumb_mcp", arc[0])
            self._set_joint(qpos, "thumb_ip", arc[0])
            self._set_joint(qpos, "index_mcp_pitch", arc[2])
            self._set_joint(qpos, "middle_mcp_pitch", arc[3])
            self._set_joint(qpos, "ring_mcp_pitch", arc[4])
            self._set_joint(qpos, "pinky_mcp_pitch", arc[5])
            self._set_joint(qpos, "index_mcp_roll", arc[6])
            self._set_joint(qpos, "ring_mcp_roll", arc[7])
            self._set_joint(qpos, "pinky_mcp_roll", arc[8])
            for finger_name, source_index in [("index", 2), ("middle", 3), ("ring", 4), ("pinky", 5)]:
                self._set_joint(qpos, f"{finger_name}_pip", arc[source_index])
                self._set_joint(qpos, f"{finger_name}_dip", arc[source_index])
            return qpos
        if self.family in {"L20", "G20"}:
            self._set_joint(qpos, "thumb_cmc_pitch", arc[0])
            self._set_joint(qpos, "index_mcp_pitch", arc[1])
            self._set_joint(qpos, "middle_mcp_pitch", arc[2])
            self._set_joint(qpos, "ring_mcp_pitch", arc[3])
            self._set_joint(qpos, "pinky_mcp_pitch", arc[4])
            self._set_joint(qpos, "thumb_cmc_roll", arc[5])
            self._set_joint(qpos, "index_mcp_roll", arc[6])
            self._set_joint(qpos, "middle_mcp_roll", arc[7])
            self._set_joint(qpos, "ring_mcp_roll", arc[8])
            self._set_joint(qpos, "pinky_mcp_roll", arc[9])
            self._set_joint(qpos, "thumb_cmc_yaw", arc[10])
            self._set_joint(qpos, "thumb_mcp", arc[15])
            self._set_joint(qpos, "thumb_dip", arc[15])
            for finger_name, source_index in [("index", 16), ("middle", 17), ("ring", 18), ("pinky", 19)]:
                pitch_value = arc[{"index": 1, "middle": 2, "ring": 3, "pinky": 4}[finger_name]]
                tip_value = arc[source_index]
                self._set_joint(qpos, f"{finger_name}_pip", tip_value if tip_value > 0 else pitch_value)
                self._set_joint(qpos, f"{finger_name}_dip", tip_value if tip_value > 0 else pitch_value)
            return qpos
        if self.family == "L21":
            self._set_joint(qpos, "thumb_cmc_pitch", arc[0])
            self._set_joint(qpos, "thumb_cmc_yaw", arc[5])
            self._set_joint(qpos, "thumb_cmc_roll", arc[10])
            self._set_joint(qpos, "thumb_mcp", arc[15])
            self._set_joint(qpos, "thumb_ip", arc[20])
            for finger_name, pitch_index, roll_index, tip_index in [
                ("index", 1, 6, 21),
                ("middle", 2, 7, 22),
                ("ring", 3, 8, 23),
                ("pinky", 4, 9, 24),
            ]:
                self._set_joint(qpos, f"{finger_name}_mcp_pitch", arc[pitch_index])
                self._set_joint(qpos, f"{finger_name}_mcp_roll", arc[roll_index])
                self._set_joint(qpos, f"{finger_name}_pip", arc[tip_index])
            return qpos
        if self.family == "L25":
            self._set_joint(qpos, "thumb_cmc_pitch", arc[0])
            self._set_joint(qpos, "thumb_cmc_yaw", arc[5])
            self._set_joint(qpos, "thumb_cmc_roll", arc[10])
            self._set_joint(qpos, "thumb_mcp", arc[15])
            self._set_joint(qpos, "thumb_ip", arc[20])
            for finger_name, pitch_index, roll_index, mid_index, tip_index in [
                ("index", 1, 6, 16, 21),
                ("middle", 2, 7, 17, 22),
                ("ring", 3, 8, 18, 23),
                ("pinky", 4, 9, 19, 24),
            ]:
                self._set_joint(qpos, f"{finger_name}_mcp_pitch", arc[pitch_index])
                self._set_joint(qpos, f"{finger_name}_mcp_roll", arc[roll_index])
                self._set_joint(qpos, f"{finger_name}_pip", arc[mid_index])
                self._set_joint(qpos, f"{finger_name}_dip", arc[tip_index])
            return qpos
        raise ValueError(f"SDK adapter not implemented for family: {self.family}")

    def _joint(self, name: str, qpos: np.ndarray) -> float:
        index = self._joint_index.get(name)
        if index is None:
            return 0.0
        return float(qpos[index])

    def _set_joint(self, qpos: np.ndarray, name: str, value: float) -> None:
        index = self._joint_index.get(name)
        if index is None:
            return
        qpos[index] = float(value)
