import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.domain.models import BiHandFrame, HandFrame
from somehand.runtime.source_sampling import FixedRateBiHandTrackingSource, FixedRateHandTrackingSource


def _hand_frame(value: float, hand_side: str = "right") -> HandFrame:
    landmarks_3d = np.full((21, 3), value, dtype=np.float64)
    landmarks_2d = np.full((21, 2), value, dtype=np.float64)
    return HandFrame(
        landmarks_3d=landmarks_3d,
        landmarks_2d=landmarks_2d,
        hand_side=hand_side,
    )


class _SnapshotHandSource:
    def __init__(self, frame: HandFrame):
        self.source_desc = "fake://hand-snapshot"
        self._snapshot = (1, frame)

    @property
    def fps(self) -> int:
        return 120

    def is_available(self) -> bool:
        return True

    def get_frame(self):
        raise AssertionError("snapshot-based source should not fall back to get_frame")

    def latest_hand_frame_snapshot(self):
        return self._snapshot

    def set_snapshot(self, index: int, frame: HandFrame) -> None:
        self._snapshot = (index, frame)

    def reset(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self):
        return {}


class _SnapshotBiHandSource:
    def __init__(self, frame: BiHandFrame):
        self.source_desc = "fake://bihand-snapshot"
        self._snapshot = (1, frame)

    @property
    def fps(self) -> int:
        return 90

    def is_available(self) -> bool:
        return True

    def get_frame(self):
        raise AssertionError("snapshot-based source should not fall back to get_frame")

    def latest_bihand_frame_snapshot(self):
        return self._snapshot

    def set_snapshot(self, index: int, frame: BiHandFrame) -> None:
        self._snapshot = (index, frame)

    def reset(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self):
        return {}


def test_fixed_rate_hand_source_uses_requested_sample_fps_and_latest_snapshot():
    source = _SnapshotHandSource(_hand_frame(1.0))
    wrapped = FixedRateHandTrackingSource(source, sample_fps=50)

    first = wrapped.get_frame().detection
    source.set_snapshot(2, _hand_frame(8.0))
    second = wrapped.get_frame().detection

    assert first is not None
    assert second is not None
    assert wrapped.fps == 50
    assert wrapped.stats_snapshot()["sample_fps"] == 50
    np.testing.assert_allclose(first.landmarks_3d, 1.0)
    np.testing.assert_allclose(second.landmarks_3d, 8.0)


def test_fixed_rate_hand_source_reuses_last_sample_when_no_new_frame_arrives():
    source = _SnapshotHandSource(_hand_frame(3.0))
    wrapped = FixedRateHandTrackingSource(source, sample_fps=60)

    first = wrapped.get_frame().detection
    second = wrapped.get_frame().detection

    assert first is not None
    assert second is not None
    np.testing.assert_allclose(second.landmarks_3d, first.landmarks_3d)


def test_fixed_rate_bihand_source_uses_requested_sample_fps():
    source = _SnapshotBiHandSource(BiHandFrame(left=_hand_frame(2.0, "left"), right=_hand_frame(4.0, "right")))
    wrapped = FixedRateBiHandTrackingSource(source, sample_fps=72)

    frame = wrapped.get_frame().detection

    assert frame is not None
    assert wrapped.fps == 72
    assert wrapped.stats_snapshot()["sample_fps"] == 72
    assert frame.left is not None
    assert frame.right is not None
    np.testing.assert_allclose(frame.left.landmarks_3d, 2.0)
    np.testing.assert_allclose(frame.right.landmarks_3d, 4.0)



class _StoppingHandSource:
    source_desc = "fake://stopping-hand"

    @property
    def fps(self) -> int:
        return 120

    def is_available(self) -> bool:
        return True

    def get_frame(self):
        raise StopIteration

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def stats_snapshot(self):
        return {}


def test_fixed_rate_hand_source_propagates_stop_iteration():
    source = _StoppingHandSource()

    wrapped = FixedRateHandTrackingSource(
        source,
        sample_fps=30,
    )

    with pytest.raises(StopIteration):
        wrapped.get_frame()
