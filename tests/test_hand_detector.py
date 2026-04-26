import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import somehand.hand_detector as hand_detector
from somehand.hand_detector import HandDetector


def test_detect_all_returns_empty_list_when_mediapipe_finds_no_hands(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class _FakeImage:
        def __init__(self, *, image_format, data):
            self.image_format = image_format
            self.data = data

    class _FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            assert image.data is frame
            assert timestamp_ms == 33
            return SimpleNamespace(hand_landmarks=[])

    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(Image=_FakeImage, ImageFormat=SimpleNamespace(SRGB="srgb")),
    )
    monkeypatch.setattr(hand_detector.cv2, "cvtColor", lambda source, code: source)

    detector = object.__new__(HandDetector)
    detector.landmarker = _FakeLandmarker()
    detector._timestamp_ms = 0

    assert detector.detect_all(frame) == []
