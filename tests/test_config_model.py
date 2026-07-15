import sys
from pathlib import Path

import pytest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel, mimic_joint_derivative
from somehand.infrastructure.model_name_resolver import ModelNameResolver
from somehand.infrastructure.vector_solver import VectorRetargeter


def _side_specific_config_paths() -> list[Path]:
    return sorted(Path("configs/retargeting/left").glob("*_left.yaml")) + sorted(
        Path("configs/retargeting/right").glob("*_right.yaml")
    )


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


def test_removed_vector_loss_is_rejected(tmp_path):
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
                '    type: "direction"',
            ]
        )
    )

    with pytest.raises(ValueError, match="vector_loss"):
        load_retargeting_config(str(config_path))


def test_removed_vector_constraint_loss_override_is_rejected(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "vector_loss_override.yaml"
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
                '      loss_type: "residual"',
                "      loss_scale: 1.0",
            ]
        )
    )

    with pytest.raises(ValueError, match="scaled keyvector residual loss"):
        load_retargeting_config(str(config_path))


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
    from somehand.runtime import load_bihand_config, load_retargeting_config

    config = load_retargeting_config("configs/retargeting/right/linkerhand_l20_right.yaml")
    bihand = load_bihand_config("configs/retargeting/bihand/linkerhand_l20_bihand.yaml")

    assert config.hand.name == "linkerhand_l20_right"
    assert config.preset == ""
    assert bihand.left_config_path.endswith("configs/retargeting/left/linkerhand_l20_left.yaml")


def test_public_api_exports_library_entrypoints():
    from somehand.api import (
        DEFAULT_BIHAND_CONFIG_PATH,
        HandFrame,
        RetargetingEngine,
        load_retargeting_config,
        resolve_config_path,
    )

    assert DEFAULT_BIHAND_CONFIG_PATH.name == "linkerhand_l20_bihand.yaml"
    assert resolve_config_path("right/linkerhand_l20_right.yaml").name == "linkerhand_l20_right.yaml"
    assert HandFrame.__name__ == "HandFrame"
    assert RetargetingEngine.__name__ == "RetargetingEngine"
    assert callable(load_retargeting_config)


def test_config_classes_do_not_expose_loader_classmethods():
    from somehand.core import BiHandRetargetingConfig, RetargetingConfig

    assert not hasattr(RetargetingConfig, "load")
    assert not hasattr(BiHandRetargetingConfig, "load")


def test_top_level_package_is_metadata_only():
    import somehand

    assert hasattr(somehand, "__version__")
    assert not hasattr(somehand, "load_retargeting_config")


def test_pyproject_keeps_optional_cli_dependencies_out_of_core():
    text = Path("pyproject.toml").read_text()
    dependencies = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    cli_extra = text.split("cli = [", 1)[1].split("]", 1)[0]
    dev_extra = text.split("dev = [", 1)[1].split("]", 1)[0]

    assert "mediapipe" not in dependencies
    assert "opencv-python" not in dependencies
    assert "pico-bridge" not in dependencies
    assert "mediapipe" in cli_extra
    assert "opencv-python" in cli_extra
    assert "pico-bridge" in cli_extra
    assert "mediapipe" in dev_extra
    assert "opencv-python" in dev_extra
    assert "pico-bridge" in dev_extra


def test_all_mjcf_assets_have_side_specific_configs():
    mjcf_roots = Path("assets/mjcf")
    config_roots = Path("configs/retargeting")
    asset_names = sorted(path.parent.name for path in mjcf_roots.glob("*/model.xml"))
    config_names = {path.stem for path in config_roots.glob("*/*.yaml")}
    missing = [name for name in asset_names if name not in config_names]
    assert missing == []


def test_side_specific_configs_load_successfully():
    config_paths = _side_specific_config_paths()
    assert config_paths
    for config_path in config_paths:
        config = load_retargeting_config(str(config_path))
        assert config.hand.name == config_path.stem


def test_side_specific_configs_instantiate_vector_retargeter():
    config_paths = _side_specific_config_paths()
    assert config_paths
    for config_path in config_paths:
        config = load_retargeting_config(str(config_path))
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        assert retargeter.config.hand.name == config.hand.name


def test_retargeting_preset_is_rejected(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "preset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "bad"',
                '  side: "right"',
                f'  mjcf_path: "{mjcf_path}"',
                "retargeting:",
                '  preset: "universal"',
            ]
        )
    )

    with pytest.raises(ValueError, match="preset is no longer supported"):
        load_retargeting_config(str(config_path))


def test_hand_config_owns_vector_topology():
    config = load_retargeting_config("configs/retargeting/right/linkerhand_o6_right.yaml")

    assert config.preset == ""
    assert len(config.vector_constraints) == 10
    thumb_base_distal = next(
        constraint for constraint in config.vector_constraints if constraint.robot == ["thumb_metacarpals", "thumb_distal"]
    )
    assert thumb_base_distal.human == [1, 3]
    assert thumb_base_distal.weight == pytest.approx(1.0)
    thumb_distal_tip = next(
        constraint for constraint in config.vector_constraints if constraint.robot == ["thumb_distal", "thumb_distal_tip"]
    )
    assert thumb_distal_tip.human == [3, 4]
    assert thumb_distal_tip.weight == pytest.approx(0.9)
    assert {
        tuple(constraint.robot)
        for constraint in config.vector_constraints
        if constraint.robot[0].split("_", 1)[0] in {"index", "middle", "ring", "pinky"}
    } == {
        ("index_proximal", "index_distal"),
        ("index_distal", "index_distal_tip"),
        ("middle_proximal", "middle_distal"),
        ("middle_distal", "middle_distal_tip"),
        ("ring_proximal", "ring_distal"),
        ("ring_distal", "ring_distal_tip"),
        ("pinky_proximal", "pinky_distal"),
        ("pinky_distal", "pinky_distal_tip"),
    }
    assert len(config.distance_constraints) == 4
    assert {tuple(constraint.robot) for constraint in config.distance_constraints} == {
        ("thumb_tip", "index_tip"),
        ("thumb_tip", "middle_tip"),
        ("thumb_tip", "ring_tip"),
        ("thumb_tip", "pinky_tip"),
    }
    distance_by_human = {tuple(constraint.human): constraint for constraint in config.distance_constraints}
    assert distance_by_human[(4, 8)].weight == pytest.approx(2000.0)
    assert distance_by_human[(4, 12)].weight == pytest.approx(1500.0)
    assert distance_by_human[(4, 16)].weight == pytest.approx(1000.0)
    assert distance_by_human[(4, 20)].weight == pytest.approx(800.0)
    assert distance_by_human[(4, 8)].scale == pytest.approx(1.0)
    assert distance_by_human[(4, 8)].threshold == pytest.approx(0.04)
    assert distance_by_human[(4, 8)].activation_type == "linear"
    assert distance_by_human[(4, 8)].scale_mode == "hand_scaled"
    assert len(config.frame_constraints) == 1
    assert config.frame_constraints[0].name == "thumb_cmc_frame"
    assert config.frame_constraints[0].primary_weight == pytest.approx(2.0)
    assert config.frame_constraints[0].secondary_weight == pytest.approx(1.8)
    assert config.angle_constraints == []


def test_wujihand_four_fingers_use_three_visible_phalange_vectors():
    config = load_retargeting_config("configs/retargeting/right/wujihand_right.yaml")

    expected_pairs = {
        ((5, 6), ("finger2_link2", "finger2_link3")),
        ((6, 7), ("finger2_link3", "finger2_link4")),
        ((7, 8), ("finger2_link4", "finger2_link4_tip")),
        ((9, 10), ("finger3_link2", "finger3_link3")),
        ((10, 11), ("finger3_link3", "finger3_link4")),
        ((11, 12), ("finger3_link4", "finger3_link4_tip")),
        ((13, 14), ("finger4_link2", "finger4_link3")),
        ((14, 15), ("finger4_link3", "finger4_link4")),
        ((15, 16), ("finger4_link4", "finger4_link4_tip")),
        ((17, 18), ("finger5_link2", "finger5_link3")),
        ((18, 19), ("finger5_link3", "finger5_link4")),
        ((19, 20), ("finger5_link4", "finger5_link4_tip")),
    }

    actual_pairs = {
        (tuple(constraint.human), tuple(constraint.robot))
        for constraint in config.vector_constraints
        if constraint.human[0] >= 5
    }

    assert actual_pairs == expected_pairs


def test_dexhand021_four_fingers_use_three_visible_phalange_vectors():
    config = load_retargeting_config("configs/retargeting/right/dexhand021_right.yaml")

    expected_pairs = {
        ((5, 6), ("f_link2_2", "f_link2_3")),
        ((6, 7), ("f_link2_3", "f_link2_4")),
        ((7, 8), ("f_link2_4", "f_link2_4_tip")),
        ((9, 10), ("f_link3_2", "f_link3_3")),
        ((10, 11), ("f_link3_3", "f_link3_4")),
        ((11, 12), ("f_link3_4", "f_link3_4_tip")),
        ((13, 14), ("f_link4_2", "f_link4_3")),
        ((14, 15), ("f_link4_3", "f_link4_4")),
        ((15, 16), ("f_link4_4", "f_link4_4_tip")),
        ((17, 18), ("f_link5_2", "f_link5_3")),
        ((18, 19), ("f_link5_3", "f_link5_4")),
        ((19, 20), ("f_link5_4", "f_link5_4_tip")),
    }

    actual_pairs = {
        (tuple(constraint.human), tuple(constraint.robot))
        for constraint in config.vector_constraints
        if constraint.human[0] >= 5
    }

    assert actual_pairs == expected_pairs


def test_omnihand_vectors_follow_mjcf_finger_links():
    config = load_retargeting_config("configs/retargeting/right/omnihand_right.yaml")

    expected_pairs = {
        ((1, 2), ("thumb_abad_link", "thumb_mcp_link")),
        ((2, 3), ("thumb_mcp_link", "thumb_pip_link")),
        ((3, 4), ("thumb_pip_link", "thumb_dip_link")),
        ((3, 4), ("thumb_dip_link", "thumb_dip_link_tip")),
        ((5, 6), ("index_abad_link", "index_pip_link")),
        ((6, 7), ("index_pip_link", "index_dip_link")),
        ((7, 8), ("index_dip_link", "index_dip_link_tip")),
        ((9, 10), ("middle_pip_link", "middle_dip_link")),
        ((10, 12), ("middle_dip_link", "middle_dip_link_tip")),
        ((13, 14), ("ring_abad_link", "ring_pip_link")),
        ((14, 15), ("ring_pip_link", "ring_dip_link")),
        ((15, 16), ("ring_dip_link", "ring_dip_link_tip")),
        ((17, 18), ("pinky_abad_link", "pinky_pip_link")),
        ((18, 19), ("pinky_pip_link", "pinky_dip_link")),
        ((19, 20), ("pinky_dip_link", "pinky_dip_link_tip")),
    }

    actual_pairs = {
        (tuple(constraint.human), tuple(constraint.robot))
        for constraint in config.vector_constraints
    }

    assert actual_pairs == expected_pairs


def test_side_specific_configs_resolve_all_configured_vectors():
    config_paths = _side_specific_config_paths()
    assert config_paths
    for config_path in config_paths:
        config = load_retargeting_config(str(config_path))
        hand_model = HandModel(config.hand.mjcf_path)
        configured_vector_count = len(config.vector_constraints)
        retargeter = VectorRetargeter(hand_model, config)
        assert retargeter.config.hand.name == config.hand.name
        assert len(retargeter.config.vector_constraints) == configured_vector_count
        assert len(retargeter.config.vector_constraints) > 0


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


def test_linkerhand_l20pro_pinky_chain_and_mesh_are_laterally_aligned():
    model = mujoco.MjModel.from_xml_path("assets/mjcf/linkerhand_l20pro_right/model.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    points = []
    for body_name in ("pinky_metacarpals", "pinky_middle", "pinky_distal"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        points.append(data.xpos[body_id])
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinky_distal_tip")
    points.append(data.site_xpos[tip_id])
    points = np.vstack(points)
    assert float(np.ptp(points[:, 1])) < 1e-3

    mesh_centers = []
    for body_name in ("pinky_proximal", "pinky_middle", "pinky_distal"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_rotation = data.xmat[body_id].reshape(3, 3)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.array(model.mesh_vert[start:start + count], copy=True)
            world_vertices = data.xpos[body_id] + (model.geom_pos[geom_id] + vertices) @ body_rotation.T
            mesh_centers.append(world_vertices.mean(axis=0))
    mesh_centers = np.vstack(mesh_centers)
    assert float(np.ptp(mesh_centers[:, 1])) < 1e-3


def test_all_distance_constraint_configs_cover_thumb_to_all_fingertips():
    expected_pairs = {(4, 8), (4, 12), (4, 16), (4, 20)}
    expected_closure_pairs = {(8, 5), (12, 9), (16, 13), (20, 17)}
    config_paths = _side_specific_config_paths()
    assert config_paths
    for config_path in config_paths:
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


def _semantic_point(hand_model: HandModel, *, hand_side: str, name: str, obj_type) -> np.ndarray:
    resolver = ModelNameResolver(hand_model.model, hand_side=hand_side)
    resolved = resolver.resolve(name, obj_type=obj_type, role="Orientation regression")
    point_id = mujoco.mj_name2id(hand_model.model, obj_type, resolved)
    if obj_type == mujoco.mjtObj.mjOBJ_SITE:
        return hand_model.data.site_xpos[point_id].copy()
    return hand_model.data.xpos[point_id].copy()


def _hand_frame(mjcf_path: str, *, hand_side: str) -> np.ndarray:
    hand_model = HandModel(mjcf_path)
    middle_base = _semantic_point(hand_model, hand_side=hand_side, name="middle_base", obj_type=mujoco.mjtObj.mjOBJ_BODY)
    middle_tip = _semantic_point(hand_model, hand_side=hand_side, name="middle_tip", obj_type=mujoco.mjtObj.mjOBJ_SITE)
    index_base = _semantic_point(hand_model, hand_side=hand_side, name="index_base", obj_type=mujoco.mjtObj.mjOBJ_BODY)
    ring_base = _semantic_point(hand_model, hand_side=hand_side, name="ring_base", obj_type=mujoco.mjtObj.mjOBJ_BODY)

    y_axis = middle_tip - middle_base
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = index_base - ring_base
    x_axis = x_axis - y_axis * np.dot(x_axis, y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def test_new_hand_models_align_to_reference_landmark_frame():
    reference_frames = {
        "right": _hand_frame("assets/mjcf/linkerhand_l20_right/model.xml", hand_side="right"),
        "left": _hand_frame("assets/mjcf/linkerhand_l20_left/model.xml", hand_side="left"),
    }
    for hand_name in ("dex5", "inspire_dfq", "inspire_ftp", "omnihand", "rohand", "sharpa_wave"):
        for hand_side in ("right", "left"):
            candidate = _hand_frame(f"assets/mjcf/{hand_name}_{hand_side}/model.xml", hand_side=hand_side)
            reference = reference_frames[hand_side]
            axis_cosines = np.diag(reference.T @ candidate)
            assert np.all(axis_cosines > 0.995), f"{hand_name}_{hand_side} frame drifted: {axis_cosines}"


def test_new_hand_models_export_single_tip_site_per_finger():
    for hand_name in ("dex5", "inspire_dfq", "inspire_ftp", "omnihand", "rohand", "sharpa_wave"):
        for hand_side in ("right", "left"):
            hand_model = HandModel(f"assets/mjcf/{hand_name}_{hand_side}/model.xml")
            tip_sites = [name for name in hand_model.get_site_names() if name.endswith("_tip")]
            assert len(tip_sites) == 5, f"{hand_name}_{hand_side} exported {len(tip_sites)} tip sites: {tip_sites}"


def test_rohand_slider_joint_drives_passive_linkage_via_polynomial_equalities():
    hand_model = HandModel("assets/mjcf/rohand_right/model.xml")
    qpos = hand_model.get_qpos()
    name_to_idx = hand_model.get_joint_name_to_qpos_index()

    baseline = {
        name: float(qpos[name_to_idx[name]])
        for name in ("if_proximal_link", "if_distal_link", "if_connecting_link", "th_proximal_link")
    }

    qpos[name_to_idx["if_slider_link"]] = 0.019
    qpos[name_to_idx["th_slider_link"]] = 0.01
    hand_model.set_qpos(qpos)
    actual = hand_model.get_qpos()

    assert abs(float(actual[name_to_idx["if_proximal_link"]]) - baseline["if_proximal_link"]) > 0.5
    assert abs(float(actual[name_to_idx["if_distal_link"]]) - baseline["if_distal_link"]) > 0.5
    assert abs(float(actual[name_to_idx["if_connecting_link"]]) - baseline["if_connecting_link"]) > 0.5
    assert abs(float(actual[name_to_idx["th_proximal_link"]]) - baseline["th_proximal_link"]) > 0.2


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


def test_vector_retargeter_reduces_grad_for_polynomial_mimics():
    config = load_retargeting_config("configs/retargeting/right/rohand_right.yaml")
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)

    mimic = next(item for item in hand_model.mimic_joints if "polycoef" in item and "multiplier" not in item)
    source_qpos_id = int(mimic["source_qpos_id"])
    source_value = 0.01
    retargeter.data.qpos[source_qpos_id] = source_value

    grad = np.zeros(retargeter.model.nv, dtype=np.float64)
    grad[source_qpos_id] = 1.5
    grad[int(mimic["qpos_id"])] = 2.0

    reduced_grad = retargeter._reduce_grad(grad)
    reduced_index = retargeter._independent_qpos_indices.index(source_qpos_id)

    expected = 1.5 + mimic_joint_derivative(mimic, source_value) * 2.0
    assert reduced_grad[reduced_index] == pytest.approx(expected)
