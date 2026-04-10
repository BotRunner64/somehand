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


def test_managed_passive_viewer_passes_window_title_via_loader(monkeypatch):
    captured = {}
    handle = _FakeHandle()
    model = object()
    data = object()

    def _fake_launch_with_title(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        kwargs["handle_return"].put_nowait(handle)

    monkeypatch.setattr(visualization, "_launch_passive_internal_with_window_title", _fake_launch_with_title)

    viewer = visualization._ManagedPassiveViewer(model, data, window_title="Retargeting")
    viewer.close(timeout=1.0)

    assert captured["args"] == (model, data)
    assert captured["kwargs"]["window_title"] == "Retargeting"


def test_set_viewer_window_title_updates_sim_filename():
    class _FakeSim:
        def __init__(self):
            self.filename = ""

    class _FakeViewer:
        def __init__(self):
            self._sim = _FakeSim()

        def _get_sim(self):
            return self._sim

    viewer = _FakeViewer()
    visualization._set_viewer_window_title(viewer, "Retargeting")

    assert viewer._sim.filename == "Retargeting"


def test_hand_visualizer_recompiles_model_when_window_title_is_set(monkeypatch):
    created = {}

    class _FakeViewer:
        def __init__(self, model, data, **kwargs):
            created["viewer_model"] = model
            created["viewer_data"] = data
            created["viewer_kwargs"] = kwargs
            self.cam = object()

        def lock(self):
            class _Lock:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc_val, exc_tb):
                    return False

            return _Lock()

        def sync(self, state_only=False):
            created.setdefault("sync_calls", []).append(state_only)

        def is_running(self):
            return True

    fake_model = object()
    fake_data = object()

    monkeypatch.setattr(visualization, "_compile_model_with_name", lambda path, name: (fake_model, fake_data))
    monkeypatch.setattr(visualization, "_ManagedPassiveViewer", _FakeViewer)
    monkeypatch.setattr(visualization, "_set_viewer_window_title", lambda viewer, title: None)
    monkeypatch.setattr(visualization, "_set_viewer_overlay_label", lambda viewer, label: None)
    monkeypatch.setattr(visualization, "configure_free_camera", lambda *args, **kwargs: None)

    hand_model = type("HandModelStub", (), {"mjcf_path": "model.xml", "model": object(), "data": object()})()
    visualizer = visualization.HandVisualizer(hand_model, window_title="Sim State")

    assert visualizer.model is fake_model
    assert visualizer.data is fake_data
    assert created["viewer_kwargs"]["window_title"] == "Sim State"


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
