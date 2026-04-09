import sys
from pathlib import Path

import pytest
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.infrastructure.hand_model import HandModel
from dex_mujoco.infrastructure.model_name_resolver import ModelNameResolver
from dex_mujoco.infrastructure.vector_solver import VectorRetargeter


def test_config_validation_rejects_mismatched_vector_lengths(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "bad"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  human_vector_pairs:",
                "    - [0, 5]",
                "  origin_link_names: []",
                '  task_link_names: ["middle_proximal"]',
            ]
        )
    )

    with pytest.raises(ValueError, match="origin_link_names length"):
        RetargetingConfig.load(str(config_path))


def test_angle_constraint_parses_scale_and_invert(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "angle.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "ok"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  human_vector_pairs:",
                "    - [0, 4]",
                '  origin_link_names: ["world"]',
                '  task_link_names: ["thumb_distal_tip"]',
                '  origin_link_types: ["body"]',
                '  task_link_types: ["site"]',
                "  vector_weights:",
                "    - 1.0",
                "  angle_constraints:",
                "    - landmarks: [4, 0, 8]",
                '      joint: "thumb_cmc_yaw"',
                "      weight: 1.2",
                "      scale: 2.0",
                "      invert: true",
            ]
        )
    )

    config = RetargetingConfig.load(str(config_path))
    assert config.angle_constraints[0].scale == pytest.approx(2.0)
    assert config.angle_constraints[0].invert is True


def test_vector_loss_parses_residual_settings(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "vector_loss.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "ok"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  human_vector_pairs:",
                "    - [0, 4]",
                '  origin_link_names: ["world"]',
                '  task_link_names: ["thumb_distal_tip"]',
                '  origin_link_types: ["body"]',
                '  task_link_types: ["site"]',
                "  vector_weights:",
                "    - 1.0",
                "  vector_loss:",
                '    type: "residual"',
                "    huber_delta: 0.03",
                "    scaling: 1.2",
                "    scale_landmarks: [0, 9]",
                '    scale_bodies: ["world", "middle_proximal"]',
                '    scale_body_types: ["body", "body"]',
            ]
        )
    )

    config = RetargetingConfig.load(str(config_path))
    assert config.vector_loss.type == "residual"
    assert config.vector_loss.huber_delta == pytest.approx(0.03)
    assert config.vector_loss.scaling == pytest.approx(1.2)


def test_all_mjcf_assets_have_side_specific_configs():
    mjcf_roots = Path("assets/mjcf")
    config_roots = Path("configs/retargeting")
    asset_names = sorted(path.parent.name for path in mjcf_roots.glob("*/model.xml"))
    config_names = {path.stem for path in config_roots.glob("*/*.yaml")}
    missing = [name for name in asset_names if name not in config_names]
    assert missing == []


def test_side_specific_configs_load_successfully():
    config_paths = sorted(Path("configs/retargeting").glob("*/*.yaml"))
    assert config_paths
    for config_path in config_paths:
        if config_path.parent.name in {"base", "bihand"}:
            continue
        config = RetargetingConfig.load(str(config_path))
        assert config.hand.name == config_path.stem


def test_side_specific_configs_instantiate_vector_retargeter():
    config_paths = sorted(Path("configs/retargeting").glob("*/*.yaml"))
    assert config_paths
    for config_path in config_paths:
        if config_path.parent.name in {"base", "bihand"}:
            continue
        config = RetargetingConfig.load(str(config_path))
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        assert retargeter.config.hand.name == config.hand.name


def test_model_name_resolver_supports_single_letter_prefixes():
    model = mujoco.MjModel.from_xml_path("assets/mjcf/dexhand021_right/model.xml")
    resolver = ModelNameResolver(model, hand_side="right")

    assert resolver.resolve("f_link1_1", obj_type=mujoco.mjtObj.mjOBJ_BODY, role="Body") == "r_f_link1_1"
    assert resolver.resolve("f_joint1_1", obj_type=mujoco.mjtObj.mjOBJ_JOINT, role="Joint") == "r_f_joint1_1"


def test_model_name_resolver_supports_left_right_word_prefixes():
    model = mujoco.MjModel.from_xml_path("assets/mjcf/revo2_right/model.xml")
    resolver = ModelNameResolver(model, hand_side="right")

    assert (
        resolver.resolve("thumb_metacarpal_link", obj_type=mujoco.mjtObj.mjOBJ_BODY, role="Body")
        == "right_thumb_metacarpal_link"
    )
    assert (
        resolver.resolve("index_distal_link_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE, role="Site")
        == "right_index_distal_link_tip"
    )


def test_revo2_model_preserves_mimic_equalities():
    hand_model = HandModel("assets/mjcf/revo2_right/model.xml")

    assert len(hand_model.mimic_joints) == 5
    assert hand_model.model.neq == 5
    assert hand_model.model.nu == 6


def test_hand_model_set_qpos_applies_mimic_relationships():
    hand_model = HandModel("assets/mjcf/revo2_right/model.xml")
    qpos = hand_model.get_qpos()
    qpos[:] = 0.0
    qpos[1] = 0.4
    qpos[3] = 0.5
    hand_model.set_qpos(qpos)

    actual_qpos = hand_model.get_qpos()
    assert actual_qpos[2] == pytest.approx(0.4)
    assert actual_qpos[4] == pytest.approx(0.5 * 1.155)
