"""Shared frame conversions and preview annotation helpers."""

from __future__ import annotations

import numpy as np

from somehand.core import BiHandFrame, HandFrame
from somehand.domain.hand_detection import HandDetection

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def to_hand_frame(detection: HandDetection) -> HandFrame:
    return HandFrame(
        landmarks_3d=detection.landmarks_3d,
        landmarks_2d=detection.landmarks_2d,
        hand_side=detection.hand_side,
    )


def copy_hand_frame(frame: HandFrame) -> HandFrame:
    return HandFrame(
        landmarks_3d=np.array(frame.landmarks_3d, copy=True),
        landmarks_2d=None if frame.landmarks_2d is None else np.array(frame.landmarks_2d, copy=True),
        hand_side=frame.hand_side,
    )


def to_bihand_frame(*, left: HandDetection | None = None, right: HandDetection | None = None) -> BiHandFrame:
    return BiHandFrame(
        left=None if left is None else to_hand_frame(left),
        right=None if right is None else to_hand_frame(right),
    )


def copy_bihand_frame(frame: BiHandFrame) -> BiHandFrame:
    return BiHandFrame(
        left=None if frame.left is None else copy_hand_frame(frame.left),
        right=None if frame.right is None else copy_hand_frame(frame.right),
    )


def annotate_preview(frame: np.ndarray, detection: HandFrame) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    if detection.landmarks_2d is None:
        return annotated
    for x, y in detection.landmarks_2d:
        cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = tuple(detection.landmarks_2d[start_idx].astype(int))
        p2 = tuple(detection.landmarks_2d[end_idx].astype(int))
        cv2.line(annotated, p1, p2, (0, 200, 0), 1)
    return annotated


def _annotate_single_hand(frame: np.ndarray, detection: HandFrame, *, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    if detection.landmarks_2d is None:
        return annotated
    for x, y in detection.landmarks_2d:
        cv2.circle(annotated, (int(x), int(y)), 3, color, -1)
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = tuple(detection.landmarks_2d[start_idx].astype(int))
        p2 = tuple(detection.landmarks_2d[end_idx].astype(int))
        cv2.line(annotated, p1, p2, color, 1)
    return annotated


def annotate_bihand_preview(frame: np.ndarray, detection: BiHandFrame) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    if detection.left is not None:
        annotated = _annotate_single_hand(annotated, detection.left, color=(255, 140, 0))
        if detection.left.landmarks_2d is not None:
            wrist = detection.left.landmarks_2d[0].astype(int)
            cv2.putText(annotated, "Left", tuple(wrist), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
    if detection.right is not None:
        annotated = _annotate_single_hand(annotated, detection.right, color=(0, 220, 0))
        if detection.right.landmarks_2d is not None:
            wrist = detection.right.landmarks_2d[0].astype(int)
            cv2.putText(annotated, "Right", tuple(wrist), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return annotated
