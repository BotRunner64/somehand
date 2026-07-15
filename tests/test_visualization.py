import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand import visualization
import somehand.cli.runtime as cli_runtime
import somehand.runtime.viewer_hand as viewer_hand
import somehand.runtime.viewer_passive as viewer_passive
import somehand.runtime.viewer_async as viewer_async
import somehand.runtime.viewer_landmarks as viewer_landmarks
from somehand.runtime.vector_visualization import (
    FRAME_NORMAL_RGBA,
    FRAME_PRIMARY_RGBA,
    FRAME_SECONDARY_RGBA,
    append_landmark_frame_geoms,
    append_landmark_vector_geoms,
    target_direction_ends,
)


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

    monkeypatch.setattr(viewer_passive.sys, "platform", "linux")
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

    monkeypatch.setattr(viewer_passive.sys, "platform", "linux")
    monkeypatch.setattr(viewer_passive, "launch_passive_internal_with_window_title", _fake_launch_with_title)

    viewer = viewer_passive.ManagedPassiveViewer(model, data, window_title="Retargeting")
    viewer.close(timeout=1.0)

    assert captured["args"] == (model, data)
    assert captured["kwargs"]["window_title"] == "Retargeting"


def test_viewer_spawn_context_uses_mjpython_launcher_on_macos(monkeypatch):
    class _FakeContext:
        def __init__(self):
            self.executable = None

        def set_executable(self, executable):
            self.executable = executable

    fake_context = _FakeContext()
    calls = {}

    def _fake_get_context(method):
        calls["method"] = method
        return fake_context

    monkeypatch.setattr(viewer_async.sys, "platform", "darwin")
    monkeypatch.setattr(viewer_async, "_resolve_mjpython_executable", lambda: "/env/bin/mjpython")
    monkeypatch.setattr(viewer_async.mp, "get_context", _fake_get_context)

    context = viewer_async._viewer_spawn_context()

    assert context is fake_context
    assert calls["method"] == "spawn"
    assert fake_context.executable == "/env/bin/mjpython"


def test_resolve_mjpython_prefers_explicit_env(monkeypatch, tmp_path):
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python_executable = env_bin / "python"
    python_executable.write_text("")
    auto_mjpython = env_bin / "mjpython"
    auto_mjpython.write_text("")
    auto_mjpython.chmod(0o755)

    explicit_mjpython = tmp_path / "custom" / "mjpython"
    explicit_mjpython.parent.mkdir()
    explicit_mjpython.write_text("")
    explicit_mjpython.chmod(0o755)

    monkeypatch.setattr(viewer_async.sys, "executable", str(python_executable))
    monkeypatch.setenv("MJPYTHON_BIN", str(explicit_mjpython))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "env"))
    monkeypatch.setenv("PATH", str(env_bin))

    assert viewer_async._resolve_mjpython_executable() == str(explicit_mjpython)


def test_robot_hand_viewer_worker_plain_qpos_clears_cached_target_directions(monkeypatch):
    updates = []

    class _FakeHandModel:
        def __init__(self, mjcf_path):
            self.mjcf_path = mjcf_path

        def get_qpos(self):
            return np.array([0.0], dtype=np.float64)

    class _FakeVisualizer:
        def __init__(self, *args, **kwargs):
            return None

        @property
        def is_running(self):
            return len(updates) < 2

        def update(self, qpos, target_directions=None, **kwargs):
            updates.append(
                (
                    np.asarray(qpos, dtype=np.float64).copy(),
                    None if target_directions is None else np.asarray(target_directions, dtype=np.float64).copy(),
                )
            )

        def close(self):
            return None

    class _FakeQueue:
        def __init__(self):
            self.items = [
                {"qpos": np.array([1.0]), "target_directions": np.array([[1.0, 0.0, 0.0]])},
                viewer_async.queue.Empty,
                np.array([2.0]),
                viewer_async.queue.Empty,
            ]

        def get_nowait(self):
            item = self.items.pop(0)
            if item is viewer_async.queue.Empty:
                raise viewer_async.queue.Empty
            return item

    monkeypatch.setattr(viewer_async.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(viewer_async, "HandModel", _FakeHandModel)
    monkeypatch.setattr(viewer_async, "HandVisualizer", _FakeVisualizer)

    viewer_async.robot_hand_viewer_worker(
        "model.xml",
        _FakeQueue(),
        None,
        None,
        "normal",
        None,
        [],
    )

    np.testing.assert_allclose(updates[0][0], [1.0])
    np.testing.assert_allclose(updates[0][1], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(updates[1][0], [2.0])
    assert updates[1][1] is None


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

    monkeypatch.setattr(viewer_hand, "compile_model_with_name", lambda path, name: (fake_model, fake_data))
    monkeypatch.setattr(viewer_hand, "ManagedPassiveViewer", _FakeViewer)
    monkeypatch.setattr(viewer_hand, "set_viewer_window_title", lambda viewer, title: None)
    monkeypatch.setattr(viewer_hand, "set_viewer_overlay_label", lambda viewer, label: None)
    monkeypatch.setattr(viewer_hand, "configure_free_camera", lambda *args, **kwargs: None)

    hand_model = type("HandModelStub", (), {"mjcf_path": "model.xml", "model": object(), "data": object()})()
    visualizer = viewer_hand.HandVisualizer(hand_model, window_title="Sim State")

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


def test_landmark_camera_defaults_match_single_hand_view():
    assert visualization._DEFAULT_LANDMARK_CAMERA["distance"] == visualization._DEFAULT_HAND_CAMERA["distance"]
    assert visualization._DEFAULT_LANDMARK_CAMERA["azimuth"] == visualization._DEFAULT_HAND_CAMERA["azimuth"]
    assert visualization._DEFAULT_LANDMARK_CAMERA["elevation"] == visualization._DEFAULT_HAND_CAMERA["elevation"]
    assert visualization._DEFAULT_LANDMARK_CAMERA["lookat"] == visualization._DEFAULT_HAND_CAMERA["lookat"]


def test_bihand_landmark_camera_defaults_match_bihand_view():
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["distance"] == visualization._DEFAULT_BIHAND_CAMERA["distance"]
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["azimuth"] == visualization._DEFAULT_BIHAND_CAMERA["azimuth"]
    assert visualization._DEFAULT_BIHAND_LANDMARK_CAMERA["elevation"] == visualization._DEFAULT_BIHAND_CAMERA["elevation"]


def test_bihand_landmark_visualizer_waits_for_both_hands_before_locking_camera(monkeypatch):
    class _FakeViewer:
        def __init__(self):
            self.cam = object()
            self.sync_calls = []

        def lock(self):
            class _Lock:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc_val, exc_tb):
                    return False

            return _Lock()

        def sync(self, state_only=False):
            self.sync_calls.append(state_only)

    model = viewer_landmarks.mujoco.MjModel.from_xml_string(viewer_landmarks.LANDMARK_VIEWER_XML)
    visualizer = object.__new__(viewer_landmarks.BiHandLandmarkVisualizer)
    visualizer.model = model
    visualizer.data = viewer_landmarks.mujoco.MjData(model)
    visualizer.viewer = _FakeViewer()
    visualizer._camera_initialized = False

    framed_points = []
    monkeypatch.setattr(
        viewer_landmarks,
        "try_frame_camera_to_points",
        lambda *args, **kwargs: framed_points.append(np.array(kwargs["points"], copy=True)) or True,
    )
    monkeypatch.setattr(
        visualizer,
        "_update_landmark_overlay",
        lambda hands: None,
    )

    hands = np.full((2, 21, 3), np.nan, dtype=np.float64)
    hands[0] = np.linspace(0.0, 1.0, 63, dtype=np.float64).reshape(21, 3)

    visualizer.update(hands)

    assert framed_points == []
    assert visualizer._camera_initialized is False


def test_bihand_landmark_visualizer_frames_camera_when_both_hands_are_visible(monkeypatch):
    class _FakeViewer:
        def __init__(self):
            self.cam = object()

        def lock(self):
            class _Lock:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc_val, exc_tb):
                    return False

            return _Lock()

        def sync(self, state_only=False):
            return None

    model = viewer_landmarks.mujoco.MjModel.from_xml_string(viewer_landmarks.LANDMARK_VIEWER_XML)
    visualizer = object.__new__(viewer_landmarks.BiHandLandmarkVisualizer)
    visualizer.model = model
    visualizer.data = viewer_landmarks.mujoco.MjData(model)
    visualizer.viewer = _FakeViewer()
    visualizer._camera_initialized = False

    framed_points = []
    monkeypatch.setattr(
        viewer_landmarks,
        "try_frame_camera_to_points",
        lambda *args, **kwargs: framed_points.append(np.array(kwargs["points"], copy=True)) or True,
    )
    monkeypatch.setattr(visualizer, "_update_landmark_overlay", lambda hands: None)

    hands = np.zeros((2, 21, 3), dtype=np.float64)
    hands[0, :, 0] = 0.2
    hands[1, :, 0] = -0.2

    visualizer.update(hands)

    assert len(framed_points) == 1
    assert framed_points[0].shape == (42, 3)
    assert visualizer._camera_initialized is True



def test_append_single_landmark_geoms_appends_after_existing_scene_geoms():
    model = visualization.mujoco.MjModel.from_xml_string(visualization._LANDMARK_VIEWER_XML)
    scene = visualization.mujoco.MjvScene(model, maxgeom=128)
    scene.ngeom = 2

    visualization._append_single_landmark_geoms(scene, np.zeros((21, 3), dtype=np.float64))

    assert scene.ngeom == 2 + 21 + len(visualization._HAND_CONNECTIONS)


def test_append_bihand_landmark_geoms_skips_nan_hand_points():
    model = visualization.mujoco.MjModel.from_xml_string(visualization._LANDMARK_VIEWER_XML)
    scene = visualization.mujoco.MjvScene(model, maxgeom=128)
    hands = np.full((2, 21, 3), np.nan, dtype=np.float64)
    hands[0] = 0.0

    visualization._append_bihand_landmark_geoms(scene, hands)

    assert scene.ngeom == 21 + len(visualization._HAND_CONNECTIONS)


def test_append_landmark_vector_geoms_adds_segment_and_tip_geoms():
    model = visualization.mujoco.MjModel.from_xml_string(visualization._LANDMARK_VIEWER_XML)
    scene = visualization.mujoco.MjvScene(model, maxgeom=16)
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1] = [0.05, 0.0, 0.0]
    landmarks[5] = [0.0, 0.06, 0.0]

    append_landmark_vector_geoms(scene, landmarks, [(0, 1), (0, 5)])

    assert scene.ngeom == 4


def test_landmark_frame_geoms_draw_orthonormal_axes():
    model = visualization.mujoco.MjModel.from_xml_string(visualization._LANDMARK_VIEWER_XML)
    scene = visualization.mujoco.MjvScene(model, maxgeom=16)
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1] = [0.0, 0.0, 0.0]
    landmarks[2] = [0.05, 0.0, 0.0]
    landmarks[5] = [0.05, 0.05, 0.0]

    append_landmark_frame_geoms(scene, landmarks, [(1, 2, 5)])

    assert scene.ngeom == 6
    np.testing.assert_allclose(scene.geoms[0].rgba, FRAME_PRIMARY_RGBA)
    np.testing.assert_allclose(scene.geoms[2].rgba, FRAME_SECONDARY_RGBA)
    np.testing.assert_allclose(scene.geoms[4].rgba, FRAME_NORMAL_RGBA)


def test_landmark_frame_geoms_skip_nan_without_recoloring_existing_geoms():
    model = visualization.mujoco.MjModel.from_xml_string(visualization._LANDMARK_VIEWER_XML)
    scene = visualization.mujoco.MjvScene(model, maxgeom=16)
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1] = [0.05, 0.0, 0.0]

    append_landmark_vector_geoms(scene, landmarks, [(0, 1)])
    original_rgba = np.array(scene.geoms[0].rgba, copy=True)
    landmarks[5] = [np.nan, 0.0, 0.0]

    append_landmark_frame_geoms(scene, landmarks, [(0, 1, 5)])

    assert scene.ngeom == 2
    np.testing.assert_allclose(scene.geoms[0].rgba, original_rgba)


def test_landmark_visualizer_diagnostic_mode_dims_base_links():
    model = viewer_landmarks.mujoco.MjModel.from_xml_string(viewer_landmarks.LANDMARK_VIEWER_XML)
    scene = viewer_landmarks.mujoco.MjvScene(model, maxgeom=128)
    fake_viewer = type("Viewer", (), {"user_scn": scene})()
    visualizer = object.__new__(viewer_landmarks.LandmarkVisualizer)
    visualizer.viewer = fake_viewer
    visualizer._vector_pairs = []
    visualizer._distance_pairs = []
    visualizer._frame_triples = [(1, 2, 5)]
    visualizer._angle_triples = []
    visualizer._diagnostic = True
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1] = [0.0, 0.0, 0.0]
    landmarks[2] = [0.05, 0.0, 0.0]
    landmarks[5] = [0.0, 0.05, 0.0]

    visualizer._update_landmark_overlay(landmarks)

    assert scene.geoms[0].rgba[3] == pytest.approx(viewer_landmarks.DIAGNOSTIC_BASE_POINT_ALPHA)
    assert scene.geoms[21].rgba[3] == pytest.approx(viewer_landmarks.DIAGNOSTIC_BASE_BONE_ALPHA)


def test_target_direction_ends_use_current_robot_vector_lengths():
    starts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    current_ends = np.array([[1.0, 0.0, 0.2]], dtype=np.float64)
    target_directions = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)

    ends = target_direction_ends(starts, current_ends, target_directions)

    np.testing.assert_allclose(ends, [[1.0, 0.2, 0.0]])


def test_target_direction_ends_are_length_capped():
    starts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    current_ends = np.array([[0.0, 0.0, 0.2]], dtype=np.float64)
    target_directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    ends = target_direction_ends(starts, current_ends, target_directions, max_length=0.035)

    np.testing.assert_allclose(ends, [[0.035, 0.0, 0.0]])


def test_robot_vector_specs_filter_world_origins_preserving_original_indices():
    config = type(
        "Config",
        (),
        {
            "vector_constraints": [
                type("Constraint", (), {"robot": ["world", "palm"], "robot_types": ["body", "body"]})(),
                type("Constraint", (), {"robot": ["palm", "tip"], "robot_types": ["body", "site"]})(),
            ]
        },
    )()

    specs = cli_runtime._robot_vector_specs(config)

    assert specs == [(1, "palm", "body", "tip", "site")]


def test_select_target_vectors_uses_original_constraint_indices():
    starts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    current_ends = starts + np.array([0.0, 0.0, 0.1], dtype=np.float64)
    target_directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    target_indices = np.array([1, 3], dtype=np.int32)

    selected_starts, selected_current_ends, selected_targets = viewer_hand.select_target_vectors(
        starts,
        current_ends,
        target_directions,
        target_indices,
    )

    np.testing.assert_allclose(selected_starts, starts)
    np.testing.assert_allclose(selected_current_ends, current_ends)
    np.testing.assert_allclose(selected_targets, target_directions[[1, 3]])


def _diagnostic_test_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="palm" pos="0 0 0">
          <geom type="sphere" size="0.01"/>
          <body name="tip" pos="0.05 0 0">
            <joint name="finger_hinge" type="hinge" range="-1 1" limited="true"/>
            <geom type="sphere" size="0.01"/>
            <site name="tip_site" pos="0.01 0 0"/>
          </body>
          <body name="slider_body" pos="0 0.05 0">
            <joint name="finger_slide" type="slide" range="0 0.02" limited="true"/>
            <geom type="sphere" size="0.01"/>
          </body>
          <body name="ball_body" pos="0 0 0.05">
            <joint name="ball_joint" type="ball"/>
            <geom type="sphere" size="0.01"/>
          </body>
          <body name="unlimited_body" pos="0 0 -0.05">
            <joint name="unlimited_hinge" type="hinge"/>
            <geom type="sphere" size="0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = viewer_hand.mujoco.MjModel.from_xml_string(xml)
    data = viewer_hand.mujoco.MjData(model)
    viewer_hand.mujoco.mj_forward(model, data)
    return model, data


def test_variable_markers_include_only_scalar_ranged_joints():
    model, _ = _diagnostic_test_model()

    markers = viewer_hand.resolve_variable_markers(model)
    marker_names = [
        viewer_hand.mujoco.mj_id2name(model, viewer_hand.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id, _qpos_id, _low, _high in markers
    ]

    assert marker_names == ["finger_hinge", "finger_slide"]


def test_fingertip_site_visibility_toggles_only_tip_sites():
    xml = """
    <mujoco>
      <worldbody>
        <body name="finger">
          <site name="finger_tip" pos="0 0 0" size="0.004" rgba="0 0 1 0.5"/>
          <site name="debug_site" pos="0 0 0.01" size="0.004" rgba="0 1 0 0.75"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = viewer_hand.mujoco.MjModel.from_xml_string(xml)
    tip_id = viewer_hand.mujoco.mj_name2id(model, viewer_hand.mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
    debug_id = viewer_hand.mujoco.mj_name2id(model, viewer_hand.mujoco.mjtObj.mjOBJ_SITE, "debug_site")

    viewer_hand.set_fingertip_site_visibility(model, visible=False)

    np.testing.assert_allclose(model.site_rgba[tip_id], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(model.site_rgba[debug_id], [0.0, 1.0, 0.0, 0.75])

    viewer_hand.set_fingertip_site_visibility(model, visible=True)

    np.testing.assert_allclose(model.site_rgba[tip_id], [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(model.site_rgba[debug_id], [0.0, 1.0, 0.0, 0.75])


def test_hand_visualizer_overlay_geoms_are_mode_gated():
    model, data = _diagnostic_test_model()
    scene = viewer_hand.mujoco.MjvScene(model, maxgeom=64)
    fake_viewer = type("Viewer", (), {"user_scn": scene})()
    visualizer = object.__new__(viewer_hand.HandVisualizer)
    visualizer.model = model
    visualizer.data = data
    visualizer.viewer = fake_viewer
    visualizer._vector_points = []
    visualizer._variable_markers = []

    visualizer._update_vector_overlay(np.ones((1, 3), dtype=np.float64))

    assert scene.ngeom == 0

    visualizer._vector_points = viewer_hand.resolve_robot_vector_points(
        model,
        [(0, "palm", "body", "tip_site", "site")],
        hand_side="right",
    )
    visualizer._variable_markers = viewer_hand.resolve_variable_markers(model)

    visualizer._update_vector_overlay(np.array([[0.0, 1.0, 0.0]], dtype=np.float64))

    assert scene.ngeom > 0


def test_hand_visualizer_draws_all_robot_constraint_types():
    model, data = _diagnostic_test_model()
    scene = viewer_hand.mujoco.MjvScene(model, maxgeom=128)
    fake_viewer = type("Viewer", (), {"user_scn": scene})()
    visualizer = object.__new__(viewer_hand.HandVisualizer)
    visualizer.model = model
    visualizer.data = data
    visualizer.viewer = fake_viewer
    visualizer._vector_points = viewer_hand.resolve_robot_vector_points(
        model,
        [(0, "palm", "body", "tip_site", "site")],
        hand_side="right",
    )
    visualizer._distance_points = viewer_hand.resolve_robot_distance_points(
        model,
        [(0, "palm", "body", "tip_site", "site")],
        hand_side="right",
    )
    visualizer._frame_points = viewer_hand.resolve_robot_frame_points(
        model,
        [(0, "palm", "body", "tip_site", "site", "slider_body", "body")],
        hand_side="right",
    )
    visualizer._angle_points = viewer_hand.resolve_robot_angle_points(
        model,
        [(0, "finger_hinge")],
        hand_side="right",
    )
    visualizer._variable_markers = []

    visualizer._update_vector_overlay(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        target_frame_primary_directions=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        target_frame_secondary_directions=np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        target_distances=np.array([0.03], dtype=np.float64),
        target_angles=np.array([0.5], dtype=np.float64),
    )

    assert scene.ngeom >= 16


def test_bihand_visualizer_rotates_target_direction_overlays_into_scene_frame():
    model, data = _diagnostic_test_model()
    scene = viewer_hand.mujoco.MjvScene(model, maxgeom=64)
    fake_viewer = type("Viewer", (), {"user_scn": scene})()
    visualizer = object.__new__(viewer_hand.BiHandVisualizer)
    visualizer.model = model
    visualizer.data = data
    visualizer.viewer = fake_viewer
    left_rotation = viewer_hand._quat_to_rotation_matrix(
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    )
    visualizer.scene = SimpleNamespace(
        left_rotation=left_rotation,
        right_rotation=np.eye(3),
        left_vector_points=viewer_hand.resolve_robot_vector_points(
            model,
            [(0, "palm", "body", "tip_site", "site")],
            hand_side="right",
        ),
        right_vector_points=[],
        left_distance_points=[],
        right_distance_points=[],
        left_frame_points=[],
        right_frame_points=[],
        left_angle_points=[],
        right_angle_points=[],
        left_variable_markers=[],
        right_variable_markers=[],
    )

    visualizer._update_vector_overlay(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        None,
    )

    assert scene.ngeom == 4
    np.testing.assert_allclose(scene.geoms[3].pos, [0.0, 0.035, 0.0], atol=1e-12)
