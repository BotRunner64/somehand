import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.retargeting_config import RetargetingConfig


def test_config_validation_rejects_mismatched_vector_lengths(tmp_path):
    mjcf_path = Path("assets/mjcf/linkerhand_l20_right/model.xml").resolve()
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hand:",
                '  name: "bad"',
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
