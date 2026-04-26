import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import somehand.runtime.source_adapters as source_adapters


def test_bihand_mediapipe_source_treats_no_detections_as_empty(monkeypatch):
    preview_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class _FakeHandDetector:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def detect_all(self, frame):
            assert frame is preview_frame
            return None

        def close(self):
            self.closed = True

        @staticmethod
        def create_source(source):
            assert source == 0
            return iter([preview_frame])

    monkeypatch.setattr(source_adapters, "HandDetector", _FakeHandDetector)

    source = source_adapters.BiHandMediaPipeInputSource(
        0,
        swap_handedness=False,
        source_desc="camera://0",
    )

    frame = source.get_frame()

    assert frame.detection is None
    assert frame.preview_frame is preview_frame
    assert source.latest_bihand_frame_snapshot() is None
