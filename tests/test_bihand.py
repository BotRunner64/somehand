import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dex_mujoco.interfaces.cli as cli_module
import dex_mujoco.infrastructure.sinks as sinks_module
from dex_mujoco.cli import build_parser
from dex_mujoco.domain import BiHandFrame, BiHandRetargetingConfig, BiHandSourceFrame, HandFrame, RetargetingConfig
from dex_mujoco.infrastructure.artifacts import load_bihand_recording_artifact, save_bihand_recording_artifact
from dex_mujoco.infrastructure.hand_model import HandModel
from dex_mujoco.infrastructure.sources import RecordingBiHandTrackingSource, create_bihand_recording_source
from dex_mujoco.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH
from dex_mujoco.visualization import BiHandScene


def _hand_frame(hand_side: str) -> HandFrame:
    landmarks_3d = np.arange(63, dtype=np.float64).reshape(21, 3)
    landmarks_2d = np.arange(42, dtype=np.float64).reshape(21, 2)
    return HandFrame(
        landmarks_3d=landmarks_3d,
        landmarks_2d=landmarks_2d,
        hand_side=hand_side,
    )


def _bihand_frame(*, left: bool = True, right: bool = True) -> BiHandFrame:
    return BiHandFrame(
        left=_hand_frame("left") if left else None,
        right=_hand_frame("right") if right else None,
    )


class _FakeBiHandSource:
    def __init__(self, frames):
        self.source_desc = "fake://bihand"
        self._frames = list(frames)
        self._index = 0

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> BiHandSourceFrame:
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> bool:
        self._index = 0
        return True

    def close(self) -> None:
        return None

    def stats_snapshot(self):
        return {}


def test_bihand_cli_uses_repo_defaults():
    parser = build_parser()
    args = parser.parse_args(["bihand", "hc-mocap"])

    assert args.command == "bihand"
    assert args.bihand_command == "hc-mocap"
    assert args.config == str(DEFAULT_BIHAND_CONFIG_PATH)
    assert args.reference_bvh == str(DEFAULT_HC_MOCAP_REFERENCE_BVH)
    assert args.udp_port == 1118


def test_bihand_config_loads_default_yaml():
    config = BiHandRetargetingConfig.load(str(DEFAULT_BIHAND_CONFIG_PATH))

    assert config.left_config_path.endswith("configs/retargeting/left/linkerhand_l20_left.yaml")
    assert config.right_config_path.endswith("configs/retargeting/right/linkerhand_l20_right.yaml")
    assert config.viewer.panel_width == 640
    assert config.viewer.left_pos == (0.22, 0.04, 0.02)
    assert config.viewer.right_pos == (-0.22, 0.04, 0.02)
    assert config.viewer.camera_lookat == (0.0, 0.04, 0.02)
    assert config.viewer.left_quat == (0.69288325, 0.01522078, -0.05862347, 0.71850151)
    assert config.viewer.right_quat == (0.71846417, 0.05829359, -0.01490552, 0.69295665)


def test_build_bihand_session_adds_replay_video_sink(monkeypatch):
    created = {}

    class _FakeVideoSink:
        def __init__(
            self,
            left_hand_model,
            right_hand_model,
            *,
            output_path,
            fps,
            panel_width,
            panel_height,
            left_pos,
            right_pos,
            camera_lookat,
            left_quat,
            right_quat,
        ):
            created["left_hand_model"] = left_hand_model
            created["right_hand_model"] = right_hand_model
            created["output_path"] = output_path
            created["fps"] = fps
            created["panel_width"] = panel_width
            created["panel_height"] = panel_height
            created["left_pos"] = left_pos
            created["right_pos"] = right_pos
            created["camera_lookat"] = camera_lookat
            created["left_quat"] = left_quat
            created["right_quat"] = right_quat

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "BiHandVideoOutputSink", _FakeVideoSink)

    engine = SimpleNamespace(
        left_engine=SimpleNamespace(hand_model=object()),
        right_engine=SimpleNamespace(hand_model=object()),
        config=SimpleNamespace(
            viewer=SimpleNamespace(
                panel_width=600,
                panel_height=400,
                window_name="test",
                left_pos=(-0.3, 0.05, 0.01),
                right_pos=(0.3, 0.05, 0.01),
                camera_lookat=(0.0, 0.05, 0.01),
                left_quat=(0.1, 0.2, 0.3, 0.4),
                right_quat=(0.5, 0.6, 0.7, 0.8),
            )
        ),
    )
    session = cli_module._build_bihand_session(
        engine,
        visualize=False,
        show_preview=False,
        video_output_path="recordings/bihand.mp4",
        video_output_fps=30,
    )

    assert len(session.sinks) == 1
    assert created == {
        "left_hand_model": engine.left_engine.hand_model,
        "right_hand_model": engine.right_engine.hand_model,
        "output_path": "recordings/bihand.mp4",
        "fps": 30,
        "panel_width": 600,
        "panel_height": 400,
        "left_pos": (-0.3, 0.05, 0.01),
        "right_pos": (0.3, 0.05, 0.01),
        "camera_lookat": (0.0, 0.05, 0.01),
        "left_quat": (0.1, 0.2, 0.3, 0.4),
        "right_quat": (0.5, 0.6, 0.7, 0.8),
    }


def test_bihand_output_window_sink_uses_mujoco_visualizer(monkeypatch):
    created = {}

    class _FakeVisualizer:
        def __init__(
            self,
            left_hand_model,
            right_hand_model,
            *,
            key_callback=None,
            left_pos=None,
            right_pos=None,
            camera_lookat=None,
            left_quat=None,
            right_quat=None,
        ):
            created["left_hand_model"] = left_hand_model
            created["right_hand_model"] = right_hand_model
            created["key_callback"] = key_callback
            created["left_pos"] = left_pos
            created["right_pos"] = right_pos
            created["camera_lookat"] = camera_lookat
            created["left_quat"] = left_quat
            created["right_quat"] = right_quat
            self.updated = []

        @property
        def is_running(self):
            return True

        def update(self, left_qpos, right_qpos):
            self.updated.append((left_qpos, right_qpos))

        def close(self):
            created["closed"] = True

    monkeypatch.setattr(sinks_module, "BiHandVisualizer", _FakeVisualizer)

    sink = sinks_module.BiHandOutputWindowSink(
        left_hand_model="left_model",
        right_hand_model="right_model",
        key_callback="handler",
        left_pos=(-0.25, 0.03, 0.01),
        right_pos=(0.25, 0.03, 0.01),
        camera_lookat=(0.0, 0.03, 0.01),
        left_quat=(0.11, 0.22, 0.33, 0.44),
        right_quat=(0.55, 0.66, 0.77, 0.88),
    )
    result = SimpleNamespace(left=SimpleNamespace(qpos=np.array([1.0])), right=SimpleNamespace(qpos=np.array([2.0])))
    sink.on_result(result)
    sink.close()

    assert created["left_hand_model"] == "left_model"
    assert created["right_hand_model"] == "right_model"
    assert created["key_callback"] == "handler"
    assert created["left_pos"] == (-0.25, 0.03, 0.01)
    assert created["right_pos"] == (0.25, 0.03, 0.01)
    assert created["camera_lookat"] == (0.0, 0.03, 0.01)
    assert created["left_quat"] == (0.11, 0.22, 0.33, 0.44)
    assert created["right_quat"] == (0.55, 0.66, 0.77, 0.88)
    assert created["closed"] is True


def test_bihand_recording_wrapper_captures_detected_frames_only():
    wrapped = RecordingBiHandTrackingSource(
        _FakeBiHandSource(
            [
                BiHandSourceFrame(detection=_bihand_frame(left=True, right=False)),
                BiHandSourceFrame(detection=None),
                BiHandSourceFrame(detection=_bihand_frame(left=False, right=True)),
            ]
        )
    )

    while wrapped.is_available():
        wrapped.get_frame()

    assert len(wrapped.recorded_frames) == 2
    assert wrapped.recorded_frames[0].left is not None
    assert wrapped.recorded_frames[0].right is None
    assert wrapped.recorded_frames[1].left is None
    assert wrapped.recorded_frames[1].right is not None


def test_bihand_recording_artifact_roundtrip(tmp_path):
    recording_path = tmp_path / "bihand.pkl"
    frames = [_bihand_frame(left=True, right=True), _bihand_frame(left=True, right=False)]

    save_bihand_recording_artifact(
        str(recording_path),
        frames,
        source_fps=60,
        source_desc="pico://both",
        input_type="pico",
        num_frames=3,
        num_detected=2,
    )

    payload = load_bihand_recording_artifact(str(recording_path))

    assert payload["fps"] == 60
    assert payload["input_source"] == "pico://both"
    assert payload["input_type"] == "pico"
    assert payload["num_frames"] == 3
    assert payload["num_detected"] == 2
    assert len(payload["frames"]) == 2
    assert payload["frames"][1].left is not None
    assert payload["frames"][1].right is None


def test_bihand_recording_source_replays_saved_frames(tmp_path):
    recording_path = tmp_path / "bihand.pkl"
    frames = [_bihand_frame(left=True, right=True), _bihand_frame(left=False, right=True)]

    save_bihand_recording_artifact(
        str(recording_path),
        frames,
        source_fps=50,
        source_desc="udp://0.0.0.0:1118",
        input_type="hc_mocap",
        num_frames=2,
        num_detected=2,
    )

    source = create_bihand_recording_source(recording_path=str(recording_path))
    seen = []

    while source.is_available():
        seen.append(source.get_frame().detection)

    assert source.fps == 50
    assert source.recording_metadata["input_type"] == "hc_mocap"
    assert len(seen) == 2
    assert seen[0] is not frames[0]
    assert seen[0].left is not None
    assert seen[0].right is not None


def test_bihand_scene_compiles_combined_mujoco_model():
    config = BiHandRetargetingConfig.load(str(DEFAULT_BIHAND_CONFIG_PATH))
    left_hand_model = HandModel(RetargetingConfig.load(config.left_config_path).hand.mjcf_path)
    right_hand_model = HandModel(RetargetingConfig.load(config.right_config_path).hand.mjcf_path)
    scene = BiHandScene(left_hand_model, right_hand_model)

    assert scene.model.nq == left_hand_model.nq + right_hand_model.nq
    assert scene.model.nu == left_hand_model.nu + right_hand_model.nu
    assert scene.left_qpos_indices.shape[0] == left_hand_model.nq
    assert scene.right_qpos_indices.shape[0] == right_hand_model.nq


def test_bihand_scene_applies_configured_root_quaternions():
    config = BiHandRetargetingConfig.load(str(DEFAULT_BIHAND_CONFIG_PATH))
    left_hand_model = HandModel(RetargetingConfig.load(config.left_config_path).hand.mjcf_path)
    right_hand_model = HandModel(RetargetingConfig.load(config.right_config_path).hand.mjcf_path)
    scene = BiHandScene(
        left_hand_model,
        right_hand_model,
        left_quat=config.viewer.left_quat,
        right_quat=config.viewer.right_quat,
    )

    assert np.allclose(scene.model.body_quat[1], np.array(config.viewer.left_quat))
    right_body_id = scene.model.nbody - (right_hand_model.model.nbody - 1)
    assert np.allclose(scene.model.body_quat[right_body_id], np.array(config.viewer.right_quat))


def test_bihand_render_helper_uses_front_palm_camera(monkeypatch):
    calls = []

    class _FakeRenderer:
        def update_scene(self, data, camera):
            return None

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            return None

    class _FakeScene:
        def __init__(self, left_hand_model, right_hand_model, *, left_pos, right_pos, left_quat, right_quat):
            self.model = SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480)))
            self.data = object()

        def update(self, left_qpos, right_qpos):
            return None

    monkeypatch.setattr(sinks_module, "BiHandScene", _FakeScene)
    monkeypatch.setattr(sinks_module, "_create_offscreen_renderer", lambda model, *, width, height: _FakeRenderer())
    monkeypatch.setattr(sinks_module.mujoco, "MjvCamera", lambda: SimpleNamespace())
    monkeypatch.setattr(sinks_module.mujoco, "mjv_defaultCamera", lambda camera: None)
    monkeypatch.setattr(sinks_module, "configure_free_camera", lambda camera, **kwargs: None)
    monkeypatch.setattr(sinks_module.cv2, "cvtColor", lambda frame, code: frame)

    def _fake_try_frame_hand_camera(camera, *, model, data, aspect_ratio=None, azimuth=None, elevation=None):
        calls.append((azimuth, elevation))
        return True

    monkeypatch.setattr(sinks_module, "_try_frame_hand_camera", _fake_try_frame_hand_camera)

    helper = sinks_module._BiHandRenderHelper(
        left_hand_model=object(),
        right_hand_model=object(),
        panel_width=640,
        panel_height=480,
        left_pos=(0.22, 0.04, 0.02),
        right_pos=(-0.22, 0.04, 0.02),
        camera_lookat=(0.0, 0.04, 0.02),
        left_quat=(0.1, 0.2, 0.3, 0.4),
        right_quat=(0.5, 0.6, 0.7, 0.8),
    )
    helper.render(SimpleNamespace(left=SimpleNamespace(qpos=np.array([1.0])), right=SimpleNamespace(qpos=np.array([2.0]))))

    assert calls == [(-90.0, -5.0)]
