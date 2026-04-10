"""Control-layer models and backend protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .hand_side import normalize_hand_side


@dataclass(slots=True)
class HandCommand:
    """Target hand command produced by retargeting."""

    target_qpos_rad: np.ndarray
    hand_model: str
    hand_side: str
    timestamp: float
    sequence_id: int

    def __post_init__(self) -> None:
        self.hand_side = normalize_hand_side(self.hand_side)


@dataclass(slots=True)
class HandState:
    """Observed controller/backend state."""

    measured_qpos_rad: np.ndarray | None
    measured_qvel: np.ndarray | None
    applied_ctrl: np.ndarray | None
    sim_time: float | None
    faults: list[int] | None
    contacts: list[str] | None
    backend: str


class ControllerBackend(Protocol):
    @property
    def is_running(self) -> bool: ...

    def start(self) -> None: ...

    def set_command(self, command: HandCommand) -> None: ...

    def get_state(self) -> HandState: ...

    def close(self) -> None: ...
