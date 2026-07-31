import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.acceptance import synthetic_hand_pose
from somehand.application.quantitative_experiment import (
    apply_experiment_method,
    run_quantitative_replay,
)
from somehand.domain.models import HandFrame
from somehand.infrastructure.artifacts import save_hand_recording_artifact
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.quantitative_artifacts import load_quantitative_result
from somehand.infrastructure.quantitative_geometry import RobotEvaluationGeometry
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest
from somehand.infrastructure.vector_solver import VectorRetargeter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
L20_CONFIG = PROJECT_ROOT / "configs" / "retargeting" / "right" / "linkerhand_l20_right.yaml"


def test_ablation_removes_only_the_selected_constraint_family():
    full = load_retargeting_config(str(L20_CONFIG))

    no_distance = apply_experiment_method(full, "no_distance")
    no_frame = apply_experiment_method(full, "no_frame")

    assert no_distance.distance_constraints == []
    assert no_distance.frame_constraints == full.frame_constraints
    assert no_distance.vector_constraints == full.vector_constraints
    assert asdict(no_distance.solver) == asdict(full.solver)
    assert no_frame.frame_constraints == []
    assert no_frame.distance_constraints == full.distance_constraints
    assert no_frame.vector_constraints == full.vector_constraints
    assert asdict(no_frame.preprocess) == asdict(full.preprocess)


def test_solver_exposes_effective_landmarks_and_slsqp_diagnostics():
    config = load_retargeting_config(str(L20_CONFIG))
    retargeter = VectorRetargeter(HandModel(config.hand.mjcf_path), config)
    retargeter.update_targets(synthetic_hand_pose("pinch"), hand_side="right")

    qpos = retargeter.solve()
    diagnostics = retargeter.get_last_solve_diagnostics()

    assert qpos.shape == (retargeter.hand_model.nq,)
    assert retargeter.get_target_landmarks().shape == (21, 3)
    assert diagnostics is not None
    assert diagnostics.optimizer_time_ns > 0
    assert diagnostics.iterations >= 0
    assert np.isfinite(diagnostics.objective)


def test_evaluation_geometry_survives_no_frame_ablation():
    full = load_retargeting_config(str(L20_CONFIG))
    active = apply_experiment_method(full, "no_frame")
    hand_model = HandModel(active.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, active)
    geometry = RobotEvaluationGeometry(hand_model, full)
    retargeter.update_targets(synthetic_hand_pose("pinch"), hand_side="right")
    retargeter.solve()

    assert geometry.has_frame
    assert geometry.common_direction_vectors().shape == (5, 3)
    assert geometry.fingertip_positions().shape == (5, 3)
    assert geometry.fingertip_surface_distances().shape == (4,)
    assert geometry.robot_frame_rotation().shape == (3, 3)


def test_quantitative_runner_writes_safe_frame_level_artifacts(tmp_path):
    recording_path = tmp_path / "input.pkl"
    frames = [
        HandFrame(synthetic_hand_pose(name), None, "right")
        for name in ("open", "pinch", "fist")
    ]
    save_hand_recording_artifact(
        str(recording_path),
        frames,
        source_fps=80,
        source_desc="test://quantitative",
        input_type="test",
        num_frames=len(frames),
        hand_side="right",
    )

    summary = run_quantitative_replay(
        config_path=str(L20_CONFIG),
        recording_path=str(recording_path),
        output_base=tmp_path / "result" / "no_frame",
        method="no_frame",
        warmup_frames=1,
    )
    arrays, metadata = load_quantitative_result(summary.metadata_path)

    assert summary.num_frames == 3
    assert arrays["qpos"].shape == (3, 21)
    assert arrays["direction_error_deg"].shape == (3, 5)
    assert arrays["objective_direction_error_deg"].shape == (3, 15)
    assert arrays["keypoint_error"].shape == (3, 5)
    assert arrays["robot_tip_surface_distances"].shape == (3, 4)
    assert np.isfinite(arrays["thumb_frame_error_deg"]).all()
    assert metadata["active_config"]["frame_constraints"] == []
    assert metadata["evaluation_geometry"]["frame_name"] == "thumb_cmc_frame"
    assert metadata["arrays_sha256"]


def test_manifest_covers_all_right_hands_and_two_reference_ablations():
    manifest = load_quantitative_manifest(
        PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml",
        project_root=PROJECT_ROOT,
    )

    assert len(manifest.hands) == 19
    assert len(manifest.planned_runs()) == 21
    assert sum(hand.main_table for hand in manifest.hands) == 6
    assert manifest.reference_hand == "linkerhand_l20_right"


def test_available_recording_manifest_does_not_require_annotations():
    manifest = load_quantitative_manifest(
        PROJECT_ROOT / "configs" / "experiments" / "quantitative_existing_pico.yaml",
        project_root=PROJECT_ROOT,
    )

    assert manifest.annotation_path is None
    assert manifest.recording_path == (PROJECT_ROOT / "recordings" / "pico_right.pkl").resolve()
    assert len(manifest.hands) == 6
    assert len(manifest.planned_runs()) == 8


def test_omnihand_now_has_a_resolved_frame_constraint():
    config = load_retargeting_config(
        str(PROJECT_ROOT / "configs" / "retargeting" / "right" / "omnihand_right.yaml")
    )
    retargeter = VectorRetargeter(HandModel(config.hand.mjcf_path), config)

    assert len(retargeter.config.frame_constraints) == 1
