import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import somehand.interfaces.cli as cli_module
import somehand.infrastructure.sinks as sinks_module
from somehand.cli import build_parser
from somehand.infrastructure.sinks import _fit_video_size
from somehand.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


def test_hc_mocap_uses_repo_defaults():
    parser = build_parser()
    args = parser.parse_args(["hc-mocap"])

    assert args.command == "hc-mocap"
    assert args.config == str(DEFAULT_CONFIG_PATH)
    assert args.reference_bvh == str(DEFAULT_HC_MOCAP_REFERENCE_BVH)
    assert args.udp_port == 1118
    assert args.hand == "right"


def test_video_command_requires_video_path():
    parser = build_parser()
    args = parser.parse_args(["video", "--video", "input.mp4", "--hand", "left"])

    assert args.command == "video"
    assert args.video == "input.mp4"
    assert args.hand == "left"


def test_video_command_accepts_both_hand_selector():
    parser = build_parser()
    args = parser.parse_args(["video", "--video", "input.mp4", "--hand", "both"])

    assert args.command == "video"
    assert args.video == "input.mp4"
    assert args.hand == "both"
    assert args.config == str(DEFAULT_BIHAND_CONFIG_PATH)


def test_replay_command_uses_realtime_replay_by_default():
    parser = build_parser()
    args = parser.parse_args(["replay", "--recording", "session.pkl", "--loop"])

    assert args.command == "replay"
    assert args.recording == "session.pkl"
    assert args.loop is True


def test_dump_video_command_requires_recording_and_output_paths():
    parser = build_parser()
    args = parser.parse_args(["dump-video", "--recording", "session.pkl", "--output", "recordings/replay.mp4"])

    assert args.command == "dump-video"
    assert args.recording == "session.pkl"
    assert args.output == "recordings/replay.mp4"


def test_run_replay_uses_landmark_retarget_and_sim_viewers_for_sim_backend(monkeypatch):
    calls = {}

    class _FakeSource:
        fps = 30
        source_desc = "recordings/session.pkl"
        recording_metadata = {
            "input_source": "recordings/session.pkl",
            "input_type": "pico",
            "num_detected": 12,
        }

    class _FakeSession:
        def run(self, source, **kwargs):
            calls["run"] = kwargs
            return SimpleNamespace(num_frames=12, num_detected=12, source_desc=source.source_desc, input_type="replay")

    def _fake_build_runtime_session(*args, **kwargs):
        calls["session_kwargs"] = kwargs
        return _FakeSession()

    monkeypatch.setattr(cli_module, "create_recording_source", lambda **kwargs: _FakeSource())
    monkeypatch.setattr(cli_module, "_wrap_source_for_recording", lambda source, **kwargs: source)
    monkeypatch.setattr(cli_module, "_build_engine", lambda args, **kwargs: SimpleNamespace(describe=lambda: {"model_name": "m", "dof": 1, "vector_pairs": 1}))
    monkeypatch.setattr(cli_module, "_build_runtime_session", _fake_build_runtime_session)
    monkeypatch.setattr(cli_module, "_print_startup", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_finalize_run", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        recording="recordings/session.pkl",
        record_output=None,
        backend="sim",
        hand="right",
        loop=False,
        config="unused.yaml",
    )

    cli_module._run_replay(args)

    assert calls["session_kwargs"]["include_landmark_viewer"] is True
    assert calls["session_kwargs"]["include_sim_state_viewer"] is True


def test_run_dump_video_uses_offline_video_only_session(monkeypatch):
    calls = {}

    class _FakeSource:
        fps = 30
        source_desc = "recordings/session.pkl"
        recording_metadata = {
            "input_source": "recordings/session.pkl",
            "input_type": "webcam",
            "num_detected": 12,
        }

    class _FakeSession:
        def run(self, source, **kwargs):
            calls["run"] = kwargs
            return SimpleNamespace(num_frames=12, num_detected=12, source_desc=source.source_desc, input_type="replay")

    def _fake_build_session(*args, **kwargs):
        calls["session_kwargs"] = kwargs
        return _FakeSession()

    monkeypatch.setattr(cli_module, "create_recording_source", lambda **kwargs: _FakeSource())
    monkeypatch.setattr(cli_module, "_build_engine", lambda args, **kwargs: SimpleNamespace(describe=lambda: {"model_name": "m", "dof": 1, "vector_pairs": 1}))
    monkeypatch.setattr(cli_module, "_build_session", _fake_build_session)
    monkeypatch.setattr(cli_module, "_print_startup", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_finalize_run", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        recording="recordings/session.pkl",
        output="recordings/replay.mp4",
        hand="right",
        config="unused.yaml",
    )

    cli_module._run_dump_video(args)

    assert calls["session_kwargs"] == {
        "visualize": False,
        "show_preview": False,
        "video_output_path": "recordings/replay.mp4",
        "video_output_fps": 30,
    }
    assert calls["run"] == {
        "input_type": "replay",
        "realtime": False,
    }


def test_custom_config_is_preserved_for_both_hand_selector():
    parser = build_parser()
    args = parser.parse_args(["video", "--video", "input.mp4", "--hand", "both", "--config", "custom_both.yaml"])

    assert args.config == "custom_both.yaml"


def test_webcam_both_dispatches_to_bihand(monkeypatch):
    called = []

    monkeypatch.setattr(cli_module, "_run_bihand_webcam", lambda args: called.append(("bihand", args.config, args.hand)))
    monkeypatch.setattr(cli_module, "_run_webcam", lambda args: called.append(("single", args.config, args.hand)))

    cli_module.main(["webcam", "--hand", "both"])

    assert called == [("bihand", str(DEFAULT_BIHAND_CONFIG_PATH), "both")]


def test_bihand_subcommand_is_rejected():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["bihand", "webcam"])


def test_build_session_adds_replay_video_sink(monkeypatch):
    created = {}

    class _FakeVideoSink:
        def __init__(self, hand_model, *, output_path, fps):
            created["hand_model"] = hand_model
            created["output_path"] = output_path
            created["fps"] = fps

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "RobotHandVideoOutputSink", _FakeVideoSink)

    engine = SimpleNamespace(hand_model=object())
    session = cli_module._build_session(
        engine,
        visualize=False,
        show_preview=False,
        video_output_path="recordings/replay.mp4",
        video_output_fps=30,
    )

    assert len(session.sinks) == 1
    assert created == {
        "hand_model": engine.hand_model,
        "output_path": "recordings/replay.mp4",
        "fps": 30,
    }


def test_build_session_adds_single_viewer_sink_for_viewer_backend(monkeypatch):
    created = []

    class _FakeLandmarkSink:
        def __init__(self, *, window_title=None):
            created.append(("landmark", window_title))

        @property
        def is_running(self):
            return True

        def close(self):
            return None

    class _FakeOutputSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("robot", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "AsyncLandmarkOutputSink", _FakeLandmarkSink)
    monkeypatch.setattr(cli_module, "RobotHandOutputSink", _FakeOutputSink)

    engine = SimpleNamespace(hand_model=object())
    session = cli_module._build_session(
        engine,
        backend="viewer",
        visualize=True,
        show_preview=False,
    )

    assert len(session.frame_sinks) == 1
    assert created == [
        ("landmark", "Input Landmarks"),
        ("robot", engine.hand_model, None, None, "Retargeting"),
    ]


def test_fit_video_size_scales_to_offscreen_limits():
    width, height = _fit_video_size(
        requested_width=1280,
        requested_height=720,
        max_width=640,
        max_height=480,
    )

    assert (width, height) == (640, 360)


def test_robot_hand_video_sink_auto_frames_only_once(monkeypatch, tmp_path):
    calls = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            return None

    class _FakeRenderer:
        def __init__(self, model, *, height, width):
            self.height = height
            self.width = width
            self.cameras = []

        def update_scene(self, data, camera):
            self.cameras.append(camera)

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            return None

    monkeypatch.setattr(sinks_module.cv2, "VideoWriter", _FakeWriter)
    monkeypatch.setattr(sinks_module.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(sinks_module.cv2, "cvtColor", lambda frame, code: frame)
    monkeypatch.setattr(
        sinks_module.mujoco,
        "MjData",
        lambda model: SimpleNamespace(qpos=np.zeros(3), geom_xpos=np.zeros((0, 3))),
    )
    monkeypatch.setattr(sinks_module.mujoco, "MjvCamera", lambda: SimpleNamespace())
    monkeypatch.setattr(sinks_module.mujoco, "mjv_defaultCamera", lambda camera: None)
    monkeypatch.setattr(sinks_module.mujoco, "mj_forward", lambda model, data: None)
    monkeypatch.setattr(sinks_module, "configure_default_hand_camera", lambda camera: None)

    def _fake_try_frame_hand_camera(camera, *, model, data, aspect_ratio=None):
        calls.append(aspect_ratio)
        return True

    monkeypatch.setattr(
        sinks_module,
        "_create_offscreen_renderer",
        lambda model, *, width, height: _FakeRenderer(model, width=width, height=height),
    )
    monkeypatch.setattr(sinks_module, "_try_frame_hand_camera", _fake_try_frame_hand_camera)

    model = SimpleNamespace(
        vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480)),
    )
    hand_model = SimpleNamespace(model=model)
    sink = sinks_module.RobotHandVideoOutputSink(
        hand_model,
        output_path=str(tmp_path / "replay.mp4"),
        fps=30,
    )

    result = SimpleNamespace(qpos=np.array([0.1, 0.2, 0.3], dtype=np.float64))
    sink.on_result(result)
    sink.on_result(result)
    sink.close()

    assert calls == [640 / 360]


def test_create_offscreen_renderer_prefers_egl_on_linux(monkeypatch):
    calls = []

    class _FakeRenderer:
        def __init__(self, model, *, height, width):
            self.model = model
            self.height = height
            self.width = width

    def _fake_reload_renderer_cls_for_backend(backend):
        calls.append(("reload", backend))
        assert backend == "egl"
        return _FakeRenderer

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(sinks_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sinks_module, "_reload_renderer_cls_for_backend", _fake_reload_renderer_cls_for_backend)

    renderer = sinks_module._create_offscreen_renderer(object(), width=320, height=240)

    assert isinstance(renderer, _FakeRenderer)
    assert calls == [("reload", "egl")]


def test_pico_command_uses_default_timeout():
    parser = build_parser()
    args = parser.parse_args(["pico", "--hand", "left"])

    assert args.command == "pico"
    assert args.pico_timeout == 60.0
    assert args.hand == "left"


def test_webcam_command_uses_current_common_args():
    parser = build_parser()
    args = parser.parse_args(["webcam"])

    assert set(vars(args)) == {
        "backend",
        "can_interface",
        "camera",
        "command",
        "config",
        "control_rate",
        "hand",
        "modbus_port",
        "model_family",
        "record_output",
        "sdk_root",
        "sim_rate",
        "swap_hands",
        "transport",
    }


def test_webcam_command_uses_default_camera():
    parser = build_parser()
    args = parser.parse_args(["webcam"])

    assert args.command == "webcam"
    assert args.camera == 0
