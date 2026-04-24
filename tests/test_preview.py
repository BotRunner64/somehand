import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import somehand.infrastructure.preview as preview_module
from somehand.domain.models import SourceFrame
from somehand.infrastructure.preview import OpenCvPreviewWindow


def test_opencv_preview_disables_after_imshow_error(monkeypatch, capsys):
    calls = {"imshow": 0, "wait_key": 0}

    def _failing_imshow(window_name, frame):
        calls["imshow"] += 1
        raise cv2.error("imshow failed")

    def _wait_key(delay):
        calls["wait_key"] += 1
        return -1

    monkeypatch.setattr(preview_module.cv2, "imshow", _failing_imshow)
    monkeypatch.setattr(preview_module.cv2, "waitKey", _wait_key)

    window = OpenCvPreviewWindow()
    frame = SourceFrame(
        detection=None,
        preview_frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert window.show(object(), frame) is True
    assert window.show(object(), frame) is True
    assert calls == {"imshow": 1, "wait_key": 0}
    assert "Warning: OpenCV preview disabled" in capsys.readouterr().out
