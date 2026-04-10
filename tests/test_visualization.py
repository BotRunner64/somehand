import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco import visualization


class _FakeHandle:
    def __init__(self, *, on_close=None):
        self.closed = False
        self._on_close = on_close

    def close(self) -> None:
        self.closed = True
        if self._on_close is not None:
            self._on_close()

    def is_running(self) -> bool:
        return not self.closed


def test_managed_passive_viewer_waits_for_render_thread_on_close(monkeypatch):
    release = threading.Event()
    thread_seen = threading.Event()
    state = {}
    handle = _FakeHandle(on_close=release.set)

    def _fake_launch_internal(*args, **kwargs):
        kwargs["handle_return"].put_nowait(handle)
        state["thread"] = threading.current_thread()
        thread_seen.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(visualization.mujoco.viewer, "_launch_internal", _fake_launch_internal)

    viewer = visualization._ManagedPassiveViewer(object(), object())

    assert thread_seen.wait(timeout=1.0) is True
    worker = state["thread"]
    assert worker.is_alive() is True
    viewer.close(timeout=1.0)

    assert handle.closed is True
    assert worker.is_alive() is False


def test_compute_bounding_sphere_accounts_for_geom_radii():
    points = visualization.np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=visualization.np.float64,
    )
    center, radius = visualization._compute_bounding_sphere(
        points,
        radii=visualization.np.array([0.05, 0.1], dtype=visualization.np.float64),
    )

    visualization.np.testing.assert_allclose(center, [0.125, 0.0, 0.0])
    assert radius == pytest.approx(0.175)


def test_camera_distance_for_radius_scales_with_scene_size():
    near = visualization._camera_distance_for_radius(
        0.05,
        fovy_degrees=45.0,
        aspect_ratio=4.0 / 3.0,
    )
    far = visualization._camera_distance_for_radius(
        0.15,
        fovy_degrees=45.0,
        aspect_ratio=4.0 / 3.0,
    )

    assert near >= visualization._MIN_CAMERA_DISTANCE
    assert far > near


def test_bihand_landmark_camera_defaults_match_bihand_view():
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["distance"] == visualization._DEFAULT_BIHAND_CAMERA["distance"]
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["azimuth"] == visualization._DEFAULT_BIHAND_CAMERA["azimuth"]
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["elevation"] == visualization._DEFAULT_BIHAND_CAMERA["elevation"]
