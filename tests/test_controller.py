import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.application import ControlledRetargetingSession
from somehand.domain import HandCommand
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.controllers.adapters import LinkerHandModelAdapter, infer_linkerhand_model_family
from somehand.infrastructure.controllers.mujoco_sim import MujocoSimController
import somehand.interfaces.cli as cli_module


class _IdentityMapping:
    @staticmethod
    def arc_to_range_left(values, family):
        return list(values)

    @staticmethod
    def arc_to_range_right(values, family):
        return list(values)

    @staticmethod
    def range_to_arc_left(values, family):
        return list(values)

    @staticmethod
    def range_to_arc_right(values, family):
        return list(values)


def test_controller_config_parses_optional_fields(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "linkerhand_l20_right"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "controller:",
                '  backend: "sim"',
                "  control_rate_hz: 120",
                "  sim_rate_hz: 600",
                '  transport: "can"',
                '  can_interface: "can1"',
                "retargeting:",
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
            ]
        )
    )

    config = load_retargeting_config(str(config_path))
    assert config.controller.backend == "sim"
    assert config.controller.control_rate_hz == 120
    assert config.controller.sim_rate_hz == 600
    assert config.controller.can_interface == "can1"


def test_infer_linkerhand_family_from_hand_name():
    assert infer_linkerhand_model_family("linkerhand_l20_right") == "L20"
    assert infer_linkerhand_model_family("linkerhand_l25_left") == "L25"


def test_l20_adapter_maps_key_joints(monkeypatch):
    monkeypatch.setattr("somehand.infrastructure.controllers.adapters._load_mapping_module", lambda sdk_root: _IdentityMapping)
    hand_model = HandModel("assets/mjcf/linkerhand_l20_right/model.xml")
    adapter = LinkerHandModelAdapter(hand_model, family="L20", hand_side="right")
    qpos = np.zeros(hand_model.nq, dtype=np.float64)
    index = hand_model.get_joint_name_to_qpos_index()
    qpos[index["thumb_cmc_pitch"]] = 0.31
    qpos[index["thumb_cmc_roll"]] = 0.22
    qpos[index["thumb_cmc_yaw"]] = 0.41
    qpos[index["index_mcp_pitch"]] = 0.52
    qpos[index["index_mcp_roll"]] = 0.12
    qpos[index["index_dip"]] = 0.77

    arc = adapter.qpos_to_sdk_arc(qpos)
    restored = adapter.sdk_arc_to_qpos(arc)

    assert arc[0] == 0.31
    assert arc[5] == 0.22
    assert arc[10] == 0.41
    assert arc[1] == 0.52
    assert arc[6] == 0.12
    assert arc[16] == 0.77
    assert restored[index["thumb_cmc_pitch"]] == 0.31
    assert restored[index["thumb_cmc_roll"]] == 0.22
    assert restored[index["thumb_cmc_yaw"]] == 0.41
    assert restored[index["index_mcp_pitch"]] == 0.52
    assert restored[index["index_mcp_roll"]] == 0.12
    assert restored[index["index_dip"]] == 0.77


def test_o6_adapter_maps_primary_sdk_axes(monkeypatch):
    monkeypatch.setattr("somehand.infrastructure.controllers.adapters._load_mapping_module", lambda sdk_root: _IdentityMapping)
    hand_model = HandModel("assets/mjcf/linkerhand_o6_right/model.xml")
    adapter = LinkerHandModelAdapter(hand_model, family="O6", hand_side="right")
    qpos = np.zeros(hand_model.nq, dtype=np.float64)
    index = hand_model.get_joint_name_to_qpos_index()
    qpos[index["rh_thumb_cmc_pitch"]] = 0.31
    qpos[index["rh_thumb_cmc_yaw"]] = 0.52
    qpos[index["rh_thumb_ip"]] = 0.66
    qpos[index["rh_index_mcp_pitch"]] = 1.00
    qpos[index["rh_middle_mcp_pitch"]] = 1.10
    qpos[index["rh_ring_mcp_pitch"]] = 1.20
    qpos[index["rh_pinky_mcp_pitch"]] = 1.30

    arc = adapter.qpos_to_sdk_arc(qpos)
    restored = adapter.sdk_arc_to_qpos(arc)

    assert np.allclose(arc, [0.31, 0.52, 1.00, 1.10, 1.20, 1.30])
    assert restored[index["rh_thumb_cmc_pitch"]] == pytest.approx(0.31)
    assert restored[index["rh_thumb_cmc_yaw"]] == pytest.approx(0.52)
    assert restored[index["rh_thumb_ip"]] == pytest.approx(0.31 * (1.08 / 0.58))
    assert restored[index["rh_index_mcp_pitch"]] == pytest.approx(1.00)
    assert restored[index["rh_index_dip"]] == pytest.approx(1.00 * (1.43 / 1.6))
    assert restored[index["rh_middle_mcp_pitch"]] == pytest.approx(1.10)
    assert restored[index["rh_middle_dip"]] == pytest.approx(1.10 * (1.43 / 1.6))


def test_mujoco_sim_controller_applies_ctrlrange_clipped_targets():
    hand_model = HandModel("assets/mjcf/linkerhand_l20_right/model.xml")
    controller = MujocoSimController(hand_model.mjcf_path, control_rate_hz=50, sim_rate_hz=200)
    target = hand_model.get_qpos()
    target[:] = 0.0
    target[0] = float(hand_model.model.actuator_ctrlrange[0, 1] + 0.2)
    controller.start()
    controller.set_command(
        HandCommand(
            target_qpos_rad=target,
            hand_model="linkerhand_l20_right",
            hand_side="right",
            timestamp=time.monotonic(),
            sequence_id=1,
        )
    )
    time.sleep(0.05)
    state = controller.get_state()
    ctrlrange = hand_model.model.actuator_ctrlrange

    assert state.backend == "sim"
    assert state.applied_ctrl is not None
    assert state.applied_ctrl.shape[0] == hand_model.nu
    assert np.all(state.applied_ctrl <= ctrlrange[:, 1] + 1e-9)
    assert np.all(state.applied_ctrl >= ctrlrange[:, 0] - 1e-9)

    controller.close()
    assert controller.is_running is False


def test_mujoco_sim_controller_adds_minimum_damping_for_undamped_models():
    controller = MujocoSimController("assets/mjcf/linkerhand_l20_right/model.xml", control_rate_hz=50, sim_rate_hz=200)

    damping = controller._hand_model.model.dof_damping[controller._actuator_dof_indices]

    assert damping.shape[0] == controller._hand_model.nu
    assert np.all(damping >= 0.75)

    controller.close()


def test_mujoco_sim_controller_softens_position_actuator_kp():
    controller = MujocoSimController("assets/mjcf/linkerhand_l20_right/model.xml", control_rate_hz=50, sim_rate_hz=200)

    kp = controller._hand_model.model.actuator_gainprm[:, 0]
    bias = controller._hand_model.model.actuator_biasprm[:, 1]

    assert np.all(kp <= 5.0)
    assert np.allclose(bias, -kp)

    controller.close()


def test_build_runtime_session_uses_controlled_session_for_sim(monkeypatch):
    captured = {}

    class _FakeController:
        def __init__(self, mjcf_path, *, control_rate_hz, sim_rate_hz):
            captured["mjcf_path"] = mjcf_path
            captured["control_rate_hz"] = control_rate_hz
            captured["sim_rate_hz"] = sim_rate_hz

        @property
        def is_running(self):
            return True

        def start(self):
            return None

        def set_command(self, command):
            return None

        def get_state(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "MujocoSimController", _FakeController)

    engine = SimpleNamespace(
        hand_model=object(),
        config=SimpleNamespace(
            hand=SimpleNamespace(mjcf_path="assets/mjcf/linkerhand_l20_right/model.xml", name="linkerhand_l20_right", side="right"),
            controller=SimpleNamespace(model_family="", default_speed=[], default_torque=[]),
        ),
    )
    args = SimpleNamespace(
        backend="sim",
        control_rate=120,
        sim_rate=600,
        sdk_root=None,
        model_family=None,
        transport="can",
        can_interface="can0",
        modbus_port="None",
    )

    session = cli_module._build_runtime_session(engine, args, visualize=False, show_preview=False)

    assert isinstance(session, ControlledRetargetingSession)
    assert captured == {
        "mjcf_path": "assets/mjcf/linkerhand_l20_right/model.xml",
        "control_rate_hz": 120,
        "sim_rate_hz": 600,
    }


def test_build_runtime_session_adds_target_and_sim_viewers_for_sim(monkeypatch):
    created = []

    class _FakeTargetSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("target", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeStateSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("state", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeController:
        @property
        def is_running(self):
            return True

        def start(self):
            return None

        def set_command(self, command):
            return None

        def get_state(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "RobotHandTargetOutputSink", _FakeTargetSink)
    monkeypatch.setattr(cli_module, "RobotHandOutputSink", _FakeStateSink)
    monkeypatch.setattr(cli_module, "MujocoSimController", lambda *args, **kwargs: _FakeController())

    engine = SimpleNamespace(
        hand_model=object(),
        config=SimpleNamespace(
            hand=SimpleNamespace(mjcf_path="assets/mjcf/linkerhand_l20_right/model.xml", name="linkerhand_l20_right", side="right"),
            controller=SimpleNamespace(model_family="", default_speed=[], default_torque=[]),
        ),
    )
    args = SimpleNamespace(
        backend="sim",
        control_rate=120,
        sim_rate=600,
        sdk_root=None,
        model_family=None,
        transport="can",
        can_interface="can0",
        modbus_port="None",
    )

    session = cli_module._build_runtime_session(engine, args, visualize=True, show_preview=False)

    assert isinstance(session, ControlledRetargetingSession)
    assert created == [
        ("target", engine.hand_model, None, None, "Retargeting"),
        ("state", engine.hand_model, None, None, "Sim State"),
    ]


def test_build_runtime_session_can_skip_target_viewer_for_sim(monkeypatch):
    created = []

    class _FakeTargetSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("target", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeStateSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("state", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeController:
        @property
        def is_running(self):
            return True

        def start(self):
            return None

        def set_command(self, command):
            return None

        def get_state(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        cli_module,
        "AsyncLandmarkOutputSink",
        lambda **kwargs: SimpleNamespace(is_running=True, close=lambda: None),
    )
    monkeypatch.setattr(cli_module, "RobotHandTargetOutputSink", _FakeTargetSink)
    monkeypatch.setattr(cli_module, "RobotHandOutputSink", _FakeStateSink)
    monkeypatch.setattr(cli_module, "MujocoSimController", lambda *args, **kwargs: _FakeController())

    engine = SimpleNamespace(
        hand_model=object(),
        config=SimpleNamespace(
            hand=SimpleNamespace(mjcf_path="assets/mjcf/linkerhand_l20_right/model.xml", name="linkerhand_l20_right", side="right"),
            controller=SimpleNamespace(model_family="", default_speed=[], default_torque=[]),
        ),
    )
    args = SimpleNamespace(
        backend="sim",
        control_rate=120,
        sim_rate=600,
        sdk_root=None,
        model_family=None,
        transport="can",
        can_interface="can0",
        modbus_port="None",
    )

    session = cli_module._build_runtime_session(
        engine,
        args,
        visualize=True,
        show_preview=False,
        include_sim_state_viewer=False,
    )

    assert isinstance(session, ControlledRetargetingSession)
    assert created == [
        ("target", engine.hand_model, None, None, "Retargeting"),
    ]


def test_build_runtime_session_can_skip_landmark_viewer_for_sim(monkeypatch):
    created = []
    landmark_created = []

    class _FakeTargetSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("target", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeStateSink:
        def __init__(self, hand_model, *, key_callback=None, overlay_label=None, window_title=None):
            created.append(("state", hand_model, key_callback, overlay_label, window_title))

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            return None

    class _FakeController:
        @property
        def is_running(self):
            return True

        def start(self):
            return None

        def set_command(self, command):
            return None

        def get_state(self):
            return None

        def close(self):
            return None

    class _FakeLandmarkSink:
        def __init__(self):
            landmark_created.append(True)

        @property
        def is_running(self):
            return True

        def on_frame(self, frame):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cli_module, "AsyncLandmarkOutputSink", _FakeLandmarkSink)
    monkeypatch.setattr(cli_module, "RobotHandTargetOutputSink", _FakeTargetSink)
    monkeypatch.setattr(cli_module, "RobotHandOutputSink", _FakeStateSink)
    monkeypatch.setattr(cli_module, "MujocoSimController", lambda *args, **kwargs: _FakeController())

    engine = SimpleNamespace(
        hand_model=object(),
        config=SimpleNamespace(
            hand=SimpleNamespace(mjcf_path="assets/mjcf/linkerhand_l20_right/model.xml", name="linkerhand_l20_right", side="right"),
            controller=SimpleNamespace(model_family="", default_speed=[], default_torque=[]),
        ),
    )
    args = SimpleNamespace(
        backend="sim",
        control_rate=120,
        sim_rate=600,
        sdk_root=None,
        model_family=None,
        transport="can",
        can_interface="can0",
        modbus_port="None",
    )

    session = cli_module._build_runtime_session(
        engine,
        args,
        visualize=True,
        show_preview=False,
        include_landmark_viewer=False,
    )

    assert isinstance(session, ControlledRetargetingSession)
    assert landmark_created == []
    assert created == [
        ("target", engine.hand_model, None, None, "Retargeting"),
        ("state", engine.hand_model, None, None, "Sim State"),
    ]


def test_controlled_session_cleans_up_when_controller_start_fails():
    class _FakeSource:
        def __init__(self):
            self.source_desc = "fake://source"
            self.closed = False

        @property
        def fps(self):
            return 30

        def is_available(self):
            return False

        def get_frame(self):
            raise StopIteration

        def reset(self):
            return False

        def close(self):
            self.closed = True

        def stats_snapshot(self):
            return {}

    class _FakeSink:
        def __init__(self):
            self.closed = False

        @property
        def is_running(self):
            return True

        def on_result(self, result):
            return None

        def close(self):
            self.closed = True

    class _FakePreview:
        def __init__(self):
            self.closed = False

        def show(self, source, frame):
            return True

        def close(self):
            self.closed = True

    class _FailingController:
        def __init__(self):
            self.closed = False

        @property
        def is_running(self):
            return False

        def start(self):
            raise ModuleNotFoundError("No module named 'can'")

        def set_command(self, command):
            return None

        def get_state(self):
            return None

        def close(self):
            self.closed = True

    source = _FakeSource()
    sink = _FakeSink()
    preview = _FakePreview()
    controller = _FailingController()
    session = ControlledRetargetingSession(
        SimpleNamespace(config=SimpleNamespace()),
        controller,
        sinks=[sink],
        preview_window=preview,
    )

    with pytest.raises(ModuleNotFoundError, match="can"):
        session.run(source, input_type="replay")

    assert source.closed is True
    assert sink.closed is True
    assert preview.closed is True
    assert controller.closed is True
