"""Small reusable primitives for vector retargeting."""

from __future__ import annotations

import numpy as np


def huber_loss(distance: float, delta: float) -> float:
    if distance <= delta:
        return 0.5 * distance * distance
    return delta * (distance - 0.5 * delta)


def huber_grad(distance: float, delta: float) -> float:
    if distance <= delta:
        return distance
    return delta


class TemporalFilter:
    """Exponential moving average filter for smooth landmark tracking."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def filter(self, value: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value.copy()
            return value
        self._prev = self.alpha * value + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self) -> None:
        self._prev = None
