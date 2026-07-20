"""Tests for viewer-only MANUS bi-hand ROS 2 integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import importlib

import somehand.cli.commands as cli_commands
import somehand.cli.runtime as cli_runtime

cli_main = importlib.import_module("somehand.cli.main")
from somehand.cli import build_parser
from somehand.runtime.manus_source import BiHandManusRos2InputSource


MANUS_SPECS = [
    ("Hand", "Invalid"),
    ("Thumb", "MCP"), ("Thumb", "PIP"),
    ("Thumb", "IP"), ("Thumb", "TIP"),
    ("Index", "MCP"), ("Index", "PIP"),
    ("Index", "IP"), ("Index", "DIP"), ("Index", "TIP"),
    ("Middle", "MCP"), ("Middle", "PIP"),
    ("Middle", "IP"), ("Middle", "DIP"), ("Middle", "TIP"),
    ("Ring", "MCP"), ("Ring", "PIP"),
    ("Ring", "IP"), ("Ring", "DIP"), ("Ring", "TIP"),
    ("Pinky", "MCP"), ("Pinky", "PIP"),
    ("Pinky", "IP"), ("Pinky", "DIP"), ("Pinky", "TIP"),
]


def _node(node_id: int, chain: str, joint: str):
    return SimpleNamespace(
        node_id=node_id,
        chain_type=chain,
        joint_type=joint,
        pose=SimpleNamespace(
            position=SimpleNamespace(
                x=float(node_id),
                y=float(node_id) + 0.1,
                z=float(node_id) + 0.2,
            )
        ),
    )


def _message(side: str):
    nodes = [
        _node(node_id, chain, joint)
        for node_id, (chain, joint) in enumerate(MANUS_SPECS)
    ]
    return SimpleNamespace(
        side=side,
        raw_nodes=nodes,
        raw_node_count=len(nodes),
    )


class _FakeRclpy:
    def ok(self) -> bool:
        return True

    def spin_once(self, node, *, timeout_sec: float) -> None:
        return None


def _bare_source() -> BiHandManusRos2InputSource:
    source = BiHandManusRos2InputSource.__new__(
        BiHandManusRos2InputSource
    )
    source.left_topic = "/left"
    source.right_topic = "/right"
    source.source_desc = "ros2://left=/left;right=/right"
    source.timeout = 1.0
    source._fps = 120
    source._rclpy = _FakeRclpy()
    source._node = object()
    source._latest_msgs = {"left": None, "right": None}
    source._available = True
    source._received_count = {"left": 0, "right": 0}
    source._converted_count = {"left": 0, "right": 0}
    source._side_mismatch_count = {"left": 0, "right": 0}
    source._timeout_count = 0
    return source


def test_bihand_manus_source_emits_fresh_pair() -> None:
    source = _bare_source()
    source._latest_msgs = {
        "left": _message("Left"),
        "right": _message("Right"),
    }
    result = source.get_frame()
    assert result.detection is not None
    assert result.detection.left.hand_side == "left"
    assert result.detection.right.hand_side == "right"
    assert result.detection.left.landmarks_3d.shape == (21, 3)
    assert result.detection.right.landmarks_3d.shape == (21, 3)
    assert source._latest_msgs == {"left": None, "right": None}


def test_bihand_manus_source_rejects_side_mismatch() -> None:
    source = _bare_source()
    source._store_message("left", _message("Right"))
    assert source._latest_msgs["left"] is None
    assert source._side_mismatch_count["left"] == 1


def test_bihand_manus_source_stops_on_one_side_timeout(capsys) -> None:
    source = _bare_source()
    source.timeout = 0.0
    source._latest_msgs["right"] = _message("Right")
    with pytest.raises(StopIteration):
        source.get_frame()
    captured = capsys.readouterr()
    assert "MANUS bi-hand input timeout" in captured.out
    assert "left" in captured.out
    assert source._timeout_count == 1


def _args() -> list[str]:
    return [
        "manus-bihand-ros2",
        "--left-topic", "/manus_glove_1",
        "--right-topic", "/manus_glove_0",
        "--left-manus-calibration", "/tmp/left.json",
        "--right-manus-calibration", "/tmp/right.json",
    ]


def test_manus_bihand_parser_is_viewer_only() -> None:
    args = build_parser().parse_args(_args())
    assert args.command == "manus-bihand-ros2"
    assert args.backend == "viewer"
    assert args.manus_timeout == 2.0
    assert args.signal_fps is None


def test_manus_bihand_parser_rejects_real_backend() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([*_args(), "--backend", "real"])


def test_manus_bihand_dispatches_to_handler(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        cli_commands,
        "_run_bihand_manus_ros2",
        lambda args: called.append(args.command),
    )
    cli_main.main(_args())
    assert called == ["manus-bihand-ros2"]


def test_build_bihand_engine_uses_both_profiles(monkeypatch) -> None:
    calls = {}

    class _FakeEngineType:
        @staticmethod
        def from_calibrated_manus_paths(**kwargs):
            calls.update(kwargs)
            return "engine"

    monkeypatch.setattr(
        cli_runtime,
        "BiHandRetargetingEngine",
        _FakeEngineType,
    )
    args = SimpleNamespace(
        config="bihand.yaml",
        backend="viewer",
        left_manus_calibration="left.json",
        right_manus_calibration="right.json",
    )
    result = cli_runtime.build_bihand_engine(
        args,
        input_type="manus_bihand_ros2",
    )
    assert result == "engine"
    assert calls == {
        "config_path": "bihand.yaml",
        "left_calibration_path": "left.json",
        "right_calibration_path": "right.json",
        "input_type": "manus_bihand_ros2",
    }


def test_run_bihand_manus_builds_source_and_session(monkeypatch) -> None:
    calls = {}

    class _FakeSource:
        source_desc = "ros2://left=/manus_glove_1;right=/manus_glove_0"
        fps = 30

    class _FakeSession:
        def run(self, source, **kwargs):
            calls["run_kwargs"] = kwargs
            return SimpleNamespace(
                num_frames=8,
                num_detected=8,
                num_detected_left=8,
                num_detected_right=8,
                num_detected_both=8,
                source_desc=source.source_desc,
                input_type="manus_bihand_ros2",
            )

    def fake_source(**kwargs):
        calls["source_kwargs"] = kwargs
        return _FakeSource()

    def fake_engine(args, **kwargs):
        calls["engine_kwargs"] = kwargs
        return SimpleNamespace(
            describe=lambda: {
                "left_model_name": "revo2_left",
                "right_model_name": "revo2_right",
                "left_dof": 11,
                "right_dof": 11,
            }
        )

    def fake_session(engine, **kwargs):
        calls["session_kwargs"] = kwargs
        return _FakeSession()

    monkeypatch.setattr(
        cli_commands,
        "create_bihand_manus_ros2_source",
        fake_source,
    )
    monkeypatch.setattr(
        cli_commands,
        "_wrap_live_bihand_source",
        lambda source, **kwargs: source,
    )
    monkeypatch.setattr(
        cli_commands,
        "_wrap_bihand_source_for_interactive_recording",
        lambda source, **kwargs: (source, None),
    )
    monkeypatch.setattr(cli_commands, "_build_bihand_engine", fake_engine)
    monkeypatch.setattr(cli_commands, "_build_bihand_session", fake_session)
    monkeypatch.setattr(
        cli_commands,
        "_print_bihand_startup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_commands,
        "_finalize_bihand_run",
        lambda *args, **kwargs: None,
    )

    args = SimpleNamespace(
        left_topic="/manus_glove_1",
        right_topic="/manus_glove_0",
        left_manus_calibration="left.json",
        right_manus_calibration="right.json",
        manus_timeout=1.0,
        signal_fps=30,
        record_output=None,
        backend="viewer",
        config="revo2_bihand.yaml",
    )
    cli_commands._run_bihand_manus_ros2(args)
    assert calls["source_kwargs"] == {
        "left_topic": "/manus_glove_1",
        "right_topic": "/manus_glove_0",
        "timeout": 1.0,
    }
    assert calls["engine_kwargs"] == {
        "input_type": "manus_bihand_ros2"
    }
    assert calls["session_kwargs"] == {
        "visualize": True,
        "show_preview": False,
        "key_callback": None,
    }
    assert calls["run_kwargs"]["input_type"] == "manus_bihand_ros2"
    assert calls["run_kwargs"]["stop_condition"] is None
