"""MediaPipe hand tracking wrapper using Tasks API."""

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from .domain.hand_side import display_hand_side, normalize_hand_side
from .paths import DEFAULT_HAND_LANDMARKER_MODEL

# Default model path relative to project root
_DEFAULT_MODEL_PATH = DEFAULT_HAND_LANDMARKER_MODEL


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


class HandDetector:
    """Wraps MediaPipe HandLandmarker (Tasks API) for hand landmark detection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        target_hand: Optional[str] = None,
        swap_handedness: bool = False,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        if model_path is None:
            model_path = str(_DEFAULT_MODEL_PATH)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self.target_hand = None if target_hand is None else normalize_hand_side(target_hand)
        self.swap_handedness = swap_handedness

        if self.target_hand not in (None, "left", "right"):
            raise ValueError("target_hand must be None, 'left', or 'right'")

    def _normalize_handedness(self, handedness: str) -> str:
        """Convert MediaPipe handedness to actual left/right semantics."""
        normalized = normalize_hand_side(handedness)
        if not self.swap_handedness:
            return normalized
        if normalized == "left":
            return "right"
        return "left"

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandDetection]:
        """Detect hand landmarks from a BGR frame.

        Args:
            frame_bgr: BGR image from OpenCV.

        Returns:
            HandDetection or None if no hand detected.
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30fps
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        normalized_handedness = [
            self._normalize_handedness(handedness_list[0].category_name)
            for handedness_list in result.handedness
        ]

        hand_idx = None
        hand_side = None
        if self.target_hand is None:
            hand_idx = 0
            hand_side = normalized_handedness[0]
        else:
            for i, actual_hand_side in enumerate(normalized_handedness):
                if actual_hand_side == self.target_hand:
                    hand_idx = i
                    hand_side = actual_hand_side
                    break

            if hand_idx is None and len(result.hand_landmarks) == 1:
                hand_idx = 0
                hand_side = self.target_hand

        if hand_idx is None:
            return None

        hand_landmarks = result.hand_landmarks[hand_idx]
        hand_world_landmarks = result.hand_world_landmarks[hand_idx]

        h, w = frame_bgr.shape[:2]
        landmarks_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_world_landmarks]
        )
        landmarks_2d = np.array(
            [[lm.x, lm.y] for lm in hand_landmarks]
        ) * np.array([w, h])

        return HandDetection(
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            hand_side=hand_side,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, detection: HandDetection) -> np.ndarray:
        """Draw hand landmarks on a frame."""
        annotated = frame_bgr.copy()
        for x, y in detection.landmarks_2d:
            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw connections between adjacent landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # index
            (0, 9), (9, 10), (10, 11), (11, 12),   # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # little
            (5, 9), (9, 13), (13, 17),               # palm
        ]
        for i, j in connections:
            p1 = tuple(detection.landmarks_2d[i].astype(int))
            p2 = tuple(detection.landmarks_2d[j].astype(int))
            cv2.line(annotated, p1, p2, (0, 200, 0), 1)

        return annotated

    def close(self):
        self.landmarker.close()

    @staticmethod
    def create_source(source: Union[int, str]) -> Iterator[np.ndarray]:
        """Create a frame source from webcam (int) or video file (str).

        Args:
            source: Webcam device index (int) or video file path (str).

        Yields:
            BGR frames from the source.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
