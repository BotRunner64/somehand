import sys
from pathlib import Path

import pytest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.model_name_resolver import ModelNameResolver
from somehand.infrastructure.vector_solver import VectorRetargeter


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _mesh_tip_for_body(model: mujoco.MjModel, body_id: int) -> np.ndarray | None:
    vertices_by_geom: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        if count <= 0:
            continue
        vertices = np.array(model.mesh_vert[start:start + count], copy=True)
        vertices = vertices @ _quat_to_matrix(model.geom_quat[geom_id]).T
        vertices += model.geom_pos[geom_id]
        vertices_by_geom.append(vertices)
    if not vertices_by_geom:
        return None
    vertices = np.vstack(vertices_by_geom)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = vertices @ axis
    if abs(float(projection.max())) < abs(float(projection.min())):
        axis = -axis
        projection = -projection
    tip_band = vertices[projection >= float(projection.max()) - 0.0015]
    centroid = tip_band.mean(axis=0)
    return tip_band[int(np.argmin(np.linalg.norm(tip_band - centroid, axis=1)))]


def test_config_validation_rejects_legacy_vector_schema(tmp_path):
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

    with pytest.raises(ValueError, match="legacy vector schema"):
        load_retargeting_config(str(config_path))


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
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
                "  angle_constraints:",
                "    - landmarks: [4, 0, 8]",
                '      joint: "thumb_cmc_yaw"',
                "      weight: 1.2",
                "      scale: 2.0",
                "      invert: true",
            ]
        )
    )

    config = load_retargeting_config(str(config_path))
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
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
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

    config = load_retargeting_config(str(config_path))
    assert config.vector_loss.type == "residual"
    assert config.vector_loss.huber_delta == pytest.approx(0.03)
    assert config.vector_loss.scaling == pytest.approx(1.2)


def test_frame_constraint_parses_thumb_cmc_axes(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "frame.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "ok"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
                "  frame_constraints:",
                '    - name: "thumb_cmc_frame"',
                "      human_origin: 1",
                "      human_primary: 2",
                "      human_secondary: 5",
                '      robot_origin: "thumb_metacarpals_base1"',
                '      robot_primary: "thumb_metacarpals"',
                '      robot_secondary: "index_proximal"',
                "      primary_weight: 1.4",
                "      secondary_weight: 0.9",
            ]
        )
    )

    config = load_retargeting_config(str(config_path))
    assert config.frame_constraints[0].name == "thumb_cmc_frame"
    assert config.frame_constraints[0].human_secondary == 5
    assert config.frame_constraints[0].primary_weight == pytest.approx(1.4)
    assert config.frame_constraints[0].secondary_weight == pytest.approx(0.9)


def test_removed_position_constraints_are_rejected(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "position.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "ok"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
                "  position_constraints:",
                "    enabled: true",
            ]
        )
    )

    with pytest.raises(ValueError, match="position_constraints"):
        load_retargeting_config(str(config_path))


def test_removed_pinch_is_rejected(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "pinch.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "ok"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                "  vector_constraints:",
                "    - human: [0, 4]",
                '      robot: ["world", "thumb_distal_tip"]',
                '      robot_types: ["body", "site"]',
                "      weight: 1.0",
                "  pinch:",
                "    enabled: true",
            ]
        )
    )

    with pytest.raises(ValueError, match="pinch"):
        load_retargeting_config(str(config_path))


def test_top_level_loader_exports_work():
    from somehand import load_bihand_config, load_retargeting_config

    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    bihand = load_bihand_config("configs/retargeting/bihand/linkerhand_l20_bihand.yaml")

    assert config.hand.name == "linkerhand_l20_right"
    assert config.preset == "universal"
    assert bihand.left_config_path.endswith("configs/retargeting/left/linkerhand_l20_left.yaml")


def test_config_classes_do_not_expose_loader_classmethods():
    from somehand import BiHandRetargetingConfig, RetargetingConfig

    assert not hasattr(RetargetingConfig, "load")
    assert not hasattr(BiHandRetargetingConfig, "load")


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
        config = load_retargeting_config(str(config_path))
        assert config.hand.name == config_path.stem


def test_side_specific_configs_instantiate_vector_retargeter():
    config_paths = sorted(Path("configs/retargeting").glob("*/*.yaml"))
    assert config_paths
    for config_path in config_paths:
        if config_path.parent.name in {"base", "bihand"}:
            continue
        config = load_retargeting_config(str(config_path))
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        assert retargeter.config.hand.name == config.hand.name


def test_universal_preset_loads_minimal_constraint_set():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_o6_right.yaml")

    assert config.preset == "universal"
    assert len(config.vector_constraints) == 16
    thumb_mid_tip = next(
        constraint for constraint in config.vector_constraints if constraint.robot == ["thumb_mid", "thumb_tip"]
    )
    assert thumb_mid_tip.loss_type == "residual"
    thumb_distal_tip = next(
        constraint for constraint in config.vector_constraints if constraint.robot == ["thumb_distal", "thumb_tip"]
    )
    assert thumb_distal_tip.loss_type == "residual"
    assert len(config.distance_constraints) == 4
    assert {tuple(constraint.robot) for constraint in config.distance_constraints} == {
        ("thumb_tip", "index_tip"),
        ("thumb_tip", "middle_tip"),
        ("thumb_tip", "ring_tip"),
        ("thumb_tip", "pinky_tip"),
    }
    assert len(config.frame_constraints) == 1
    assert config.frame_constraints[0].name == "thumb_cmc_frame"
    assert config.angle_constraints == []


def test_side_specific_configs_instantiate_universal_vector_retargeter():
    config_paths = sorted(Path("configs/retargeting").glob("*/*.yaml"))
    assert config_paths
    for config_path in config_paths:
        if config_path.parent.name in {"base", "bihand"}:
            continue
        config = load_retargeting_config(str(config_path))
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        assert config.preset == "universal"
        assert retargeter.config.hand.name == config.hand.name
        assert len(retargeter.config.distance_constraints) == 4
        assert len(retargeter.config.frame_constraints) == 1
        assert retargeter.config.frame_constraints[0].name == "thumb_cmc_frame"
        assert retargeter.config.angle_constraints == []


def test_all_fingertip_sites_align_with_mesh_surface_points():
    model_paths = sorted(Path("assets/mjcf").glob("*/model.xml"))
    for model_path in model_paths:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        for site_id in range(model.nsite):
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            if not site_name or not site_name.endswith("_tip"):
                continue
            body_id = int(model.site_bodyid[site_id])
            mesh_tip = _mesh_tip_for_body(model, body_id)
            if mesh_tip is None:
                continue
            error = float(np.linalg.norm(model.site_pos[site_id] - mesh_tip))
            assert error < 1e-3, f"{model_path}:{site_name} drifted {error:.4f}m from mesh tip"
            assert model.site_size[site_id, 0] == pytest.approx(0.004)
            assert np.allclose(model.site_rgba[site_id], np.array([1.0, 0.0, 0.0, 1.0]))


def test_all_distance_constraint_configs_cover_thumb_to_all_fingertips():
    expected_pairs = {(4, 8), (4, 12), (4, 16), (4, 20)}
    expected_closure_pairs = {(8, 5), (12, 9), (16, 13), (20, 17)}
    config_paths = sorted(Path("configs/retargeting").glob("*/*.yaml"))
    assert config_paths
    for config_path in config_paths:
        if config_path.parent.name == "bihand" or config_path.name.startswith("_"):
            continue
        config = load_retargeting_config(str(config_path))
        if not config.distance_constraints:
            continue
        actual_pairs = {tuple(constraint.human) for constraint in config.distance_constraints}
        assert expected_pairs.issubset(actual_pairs), f"{config_path} pinch distance pairs mismatch: {actual_pairs}"
        assert not expected_closure_pairs.issubset(actual_pairs), f"{config_path} should no longer include closure pairs: {actual_pairs}"


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


def test_model_name_resolver_supports_wujihand_finger_names():
    model = mujoco.MjModel.from_xml_path("assets/mjcf/wujihand_right/model.xml")
    resolver = ModelNameResolver(model, hand_side="right")

    assert (
        resolver.resolve("finger3_link2", obj_type=mujoco.mjtObj.mjOBJ_BODY, role="Body")
        == "right_finger3_link2"
    )
    assert (
        resolver.resolve("finger3_link4_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE, role="Site")
        == "right_finger3_link4_tip"
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
