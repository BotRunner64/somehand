"""OpenCV preview window adapter."""

from __future__ import annotations

import cv2

from somehand.domain import SourceFrame


class OpenCvPreviewWindow:
    def __init__(self, window_name: str = "Hand Detection"):
        self.window_name = window_name
        self._disabled = False

    def show(self, source: object, frame: SourceFrame) -> bool:
        if self._disabled:
            return True
        if frame.preview_frame is None:
            return True

        preview = frame.preview_frame
        annotate_preview = getattr(source, "annotate_preview", None)
        if frame.detection is not None and callable(annotate_preview):
            preview = annotate_preview(preview, frame.detection)

        try:
            cv2.imshow(self.window_name, preview)
            return (cv2.waitKey(1) & 0xFF) != ord("q")
        except cv2.error as exc:
            self._disabled = True
            print(f"Warning: OpenCV preview disabled: {exc}")
            return True

    def close(self) -> None:
        if self._disabled:
            return
        cv2.destroyAllWindows()
