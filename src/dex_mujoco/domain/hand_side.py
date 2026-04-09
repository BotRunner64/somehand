"""Utilities for explicit left/right hand-side semantics."""

from __future__ import annotations

HandSide = str
HAND_SIDES: tuple[HandSide, HandSide] = ("left", "right")


def normalize_hand_side(value: str) -> HandSide:
    normalized = value.strip().lower()
    if normalized not in HAND_SIDES:
        raise ValueError(f"hand side must be one of {HAND_SIDES}, got {value!r}")
    return normalized


def display_hand_side(hand_side: str) -> str:
    return normalize_hand_side(hand_side).capitalize()
