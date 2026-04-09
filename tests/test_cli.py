import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dex_mujoco.interfaces.cli as cli_module
import dex_mujoco.infrastructure.sinks as sinks_module
from dex_mujoco.cli import build_parser
from dex_mujoco.infrastructure.sinks import _fit_video_size
from dex_mujoco.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


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


def test_replay_command_uses_realtime_replay_by_default():
    parser = build_parser()
    args = parser.parse_args(["replay", "--recording", "session.pkl", "--loop"])

    assert args.command == "replay"
    assert args.recording == "session.pkl"
    assert args.loop is True


def test_replay_command_accepts_dump_video_path():
    parser = build_parser()
    args = parser.parse_args(["replay", "--recording", "session.pkl", "--dump-video", "recordings/replay.mp4"])

    assert args.command == "replay"
    assert args.dump_video == "recordings/replay.mp4"


def test_bihand_default_config_replaces_single_hand_default():
    args = SimpleNamespace(config=str(DEFAULT_CONFIG_PATH))

    cli_module._use_default_bihand_config_if_needed(args)

    assert args.config == str(DEFAULT_BIHAND_CONFIG_PATH)


def test_webcam_both_dispatches_to_bihand(monkeypatch):
    called = []

    monkeypatch.setattr(cli_module, "_run_bihand_webcam", lambda args: called.append(("bihand", args.config, args.hand)))
    monkeypatch.setattr(cli_module, "_run_webcam", lambda args: called.append(("single", args.config, args.hand)))

    cli_module.main(["webcam", "--hand", "both"])

    assert called == [("bihand", str(DEFAULT_BIHAND_CONFIG_PATH), "both")]


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


def test_build_session_falls_back_to_video_only_when_visualization_unavailable(monkeypatch, capsys):
    class _FakeVideoSink:
        def __init__(self, hand_model, *, output_path, fps):
            self.hand_model = hand_model
            self.output_path = output_path
            self.fps = fps

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "AsyncLandmarkOutputSink", lambda: (_ for _ in ()).throw(RuntimeError("no display")))
    monkeypatch.setattr(cli_module, "RobotHandVideoOutputSink", _FakeVideoSink)

    engine = SimpleNamespace(hand_model=object())
    session = cli_module._build_session(
        engine,
        visualize=True,
        show_preview=False,
        video_output_path="recordings/replay.mp4",
        video_output_fps=30,
        allow_visualization_fallback=True,
    )

    assert len(session.frame_sinks) == 0
    assert len(session.sinks) == 1
    assert isinstance(session.sinks[0], _FakeVideoSink)
    assert "visualization disabled during replay video dump" in capsys.readouterr().out


def test_build_session_skips_visualization_when_glfw_unavailable(monkeypatch, capsys):
    class _FakeVideoSink:
        def __init__(self, hand_model, *, output_path, fps):
            self.hand_model = hand_model
            self.output_path = output_path
            self.fps = fps

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "_interactive_visualization_available", lambda: (False, "GLFW is unavailable"))
    monkeypatch.setattr(cli_module, "RobotHandVideoOutputSink", _FakeVideoSink)

    engine = SimpleNamespace(hand_model=object())
    session = cli_module._build_session(
        engine,
        visualize=True,
        show_preview=False,
        video_output_path="recordings/replay.mp4",
        video_output_fps=30,
        allow_visualization_fallback=True,
    )

    assert len(session.frame_sinks) == 0
    assert len(session.sinks) == 1
    assert isinstance(session.sinks[0], _FakeVideoSink)
    assert "visualization disabled during replay video dump: GLFW is unavailable" in capsys.readouterr().out


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
        "camera",
        "command",
        "config",
        "hand",
        "record_output",
        "swap_hands",
    }


def test_webcam_command_uses_default_camera():
    parser = build_parser()
    args = parser.parse_args(["webcam"])

    assert args.command == "webcam"
    assert args.camera == 0
