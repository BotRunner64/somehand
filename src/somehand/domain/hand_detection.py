"""Hand landmark detection model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand_side import display_hand_side, normalize_hand_side


@dataclass
class HandDetection:
    landmarks_3d: np.ndarray  # (21, 3)
    landmarks_2d: np.ndarray  # (21, 2)
    hand_side: str  # "left" or "right"

    def __post_init__(self) -> None:
        self.hand_side = normalize_hand_side(self.hand_side)

    @property
    def handedness(self) -> str:
        return display_hand_side(self.hand_side)
