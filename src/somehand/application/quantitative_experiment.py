"""Offline orchestration for quantitative retargeting experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from somehand.domain.config import RetargetingConfig
from somehand.domain.quantitative_metrics import (
    HUMAN_COMMON_DIRECTION_PAIRS,
    HUMAN_FINGERTIP_INDICES,
    HUMAN_PINCH_PAIRS,
    direction_errors_degrees,
    directions_from_points,
    normalized_semantic_keypoint_errors,
    rotation_geodesic_degrees,
)
from somehand.infrastructure.artifacts import load_hand_recording_artifact
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.quantitative_artifacts import (
    load_annotations,
    runtime_environment_metadata,
    save_quantitative_result,
    sha256_file,
)
from somehand.infrastructure.quantitative_geometry import RobotEvaluationGeometry
from somehand.infrastructure.vector_solver import SolveDiagnostics, VectorRetargeter


EXPERIMENT_METHODS: tuple[str, ...] = ("full", "no_distance", "no_frame")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class QuantitativeRunSummary:
    hand_name: str
    method: str
    num_frames: int
    mean_direction_error_deg: float
    mean_keypoint_error: float
    mean_solve_time_ms: float
    solver_success_rate: float
    arrays_path: Path
    metadata_path: Path


def apply_experiment_method(config: RetargetingConfig, method: str) -> RetargetingConfig:
    """Clone a full config and remove exactly one constraint family for an ablation."""
    if method not in EXPERIMENT_METHODS:
        raise ValueError(f"Unknown experiment method {method!r}; expected one of {EXPERIMENT_METHODS}")
    active = deepcopy(config)
    if method == "no_distance":
        active.distance_constraints = []
    elif method == "no_frame":
        active.frame_constraints = []
    return active


def _build_retargeter(
    config_path: str,
    *,
    method: str,
) -> tuple[RetargetingConfig, RetargetingConfig, HandModel, VectorRetargeter]:
    full_config = load_retargeting_config(config_path)
    active_config = apply_experiment_method(full_config, method)
    hand_model = HandModel(active_config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, active_config)
    return full_config, active_config, hand_model, retargeter


def _warm_up(
    config_path: str,
    *,
    method: str,
    frames: list[object],
    warmup_frames: int,
) -> None:
    if warmup_frames <= 0 or not frames:
        return
    _, _, _, retargeter = _build_retargeter(config_path, method=method)
    for frame in frames[:warmup_frames]:
        retargeter.update_targets(frame.landmarks_3d, hand_side=frame.hand_side)
        retargeter.solve()


def _diagnostic_values(diagnostic: SolveDiagnostics) -> tuple[int, bool, int, int, int, int, float]:
    return (
        diagnostic.optimizer_time_ns,
        diagnostic.success,
        diagnostic.status,
        diagnostic.iterations,
        diagnostic.function_evaluations,
        diagnostic.jacobian_evaluations,
        diagnostic.objective,
    )


def _distance_targets_by_pair(retargeter: VectorRetargeter) -> np.ndarray:
    aligned = np.full(4, np.nan, dtype=np.float64)
    targets = retargeter.get_target_distances()
    if targets is None:
        return aligned
    pair_to_index = {pair: index for index, pair in enumerate(HUMAN_PINCH_PAIRS)}
    for constraint_index, constraint in enumerate(retargeter.config.distance_constraints):
        output_index = pair_to_index.get(tuple(constraint.human))
        if output_index is not None:
            aligned[output_index] = targets[constraint_index]
    return aligned


def run_quantitative_replay(
    *,
    config_path: str,
    recording_path: str,
    output_base: str | Path,
    method: str = "full",
    experiment_id: str = "dex_retargeting_quantitative_v1",
    annotation_path: str | None = None,
    warmup_frames: int = 80,
    start_frame: int = 0,
    max_frames: int | None = None,
    contact_threshold_m: float = 0.001,
    human_pinch_threshold_m: float = 0.020,
    minimum_contact_duration_s: float = 0.1,
) -> QuantitativeRunSummary:
    """Replay one recording through one hand/method and persist all frame-level data."""
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be > 0 when provided")
    if contact_threshold_m < 0.0:
        raise ValueError("contact_threshold_m must be >= 0")
    if human_pinch_threshold_m <= 0.0:
        raise ValueError("human_pinch_threshold_m must be > 0")
    if minimum_contact_duration_s < 0.0:
        raise ValueError("minimum_contact_duration_s must be >= 0")

    config_path_obj = Path(config_path).resolve()
    recording_path_obj = Path(recording_path).resolve()
    annotation_path_obj = None if annotation_path is None else Path(annotation_path).resolve()
    recording_sha256 = sha256_file(recording_path_obj)
    annotations: dict[str, object] | None = None
    if annotation_path_obj is not None:
        annotations = load_annotations(annotation_path_obj)
        annotated_recording_hash = str(annotations.get("recording_sha256", ""))
        if annotated_recording_hash and annotated_recording_hash != recording_sha256:
            raise ValueError(
                f"Annotation recording hash {annotated_recording_hash} does not match "
                f"{recording_path_obj} ({recording_sha256})"
            )
    recording = load_hand_recording_artifact(str(recording_path_obj))
    all_frames = list(recording["frames"])
    if annotations is not None:
        if int(annotations["sample_rate_hz"]) != int(recording["fps"]):
            raise ValueError("Annotation sample rate does not match the recording")
        if int(annotations["num_frames"]) != len(all_frames):
            raise ValueError("Annotation frame count does not match the recording")
    stop_frame = len(all_frames) if max_frames is None else min(len(all_frames), start_frame + max_frames)
    frames = all_frames[start_frame:stop_frame]
    if not frames:
        raise ValueError("Selected recording range contains no frames")

    _warm_up(
        str(config_path_obj),
        method=method,
        frames=frames,
        warmup_frames=min(warmup_frames, len(frames)),
    )
    full_config, active_config, hand_model, retargeter = _build_retargeter(
        str(config_path_obj),
        method=method,
    )
    full_config_snapshot = asdict(full_config)
    active_config_snapshot = asdict(active_config)
    geometry = RobotEvaluationGeometry(hand_model, full_config)
    initial_qpos = hand_model.get_qpos()

    num_frames = len(frames)
    fps = int(recording["fps"])
    num_common_directions = len(HUMAN_COMMON_DIRECTION_PAIRS)
    num_objective_directions = len(retargeter.config.vector_constraints)
    qpos = np.empty((num_frames, hand_model.nq), dtype=np.float64)
    effective_landmarks = np.empty((num_frames, 21, 3), dtype=np.float64)
    human_directions = np.empty((num_frames, num_common_directions, 3), dtype=np.float64)
    robot_directions = np.empty((num_frames, num_common_directions, 3), dtype=np.float64)
    direction_error_deg = np.empty((num_frames, num_common_directions), dtype=np.float64)
    direction_valid = np.empty((num_frames, num_common_directions), dtype=bool)
    objective_direction_error_deg = np.empty(
        (num_frames, num_objective_directions),
        dtype=np.float64,
    )
    human_keypoints = np.empty((num_frames, 5, 3), dtype=np.float64)
    robot_keypoints = np.empty((num_frames, 5, 3), dtype=np.float64)
    keypoint_error = np.empty((num_frames, 5), dtype=np.float64)
    human_palm_scale = np.empty(num_frames, dtype=np.float64)
    robot_palm_scale = np.empty(num_frames, dtype=np.float64)
    human_tip_distances = np.empty((num_frames, 4), dtype=np.float64)
    robot_tip_site_distances = np.empty((num_frames, 4), dtype=np.float64)
    robot_tip_surface_distances = np.empty((num_frames, 4), dtype=np.float64)
    target_tip_distances = np.empty((num_frames, 4), dtype=np.float64)
    human_frame_rotation = np.full((num_frames, 3, 3), np.nan, dtype=np.float64)
    robot_frame_rotation = np.full((num_frames, 3, 3), np.nan, dtype=np.float64)
    thumb_frame_error_deg = np.full(num_frames, np.nan, dtype=np.float64)
    optimizer_time_ns = np.empty(num_frames, dtype=np.int64)
    solver_success = np.empty(num_frames, dtype=bool)
    solver_status = np.empty(num_frames, dtype=np.int32)
    solver_iterations = np.empty(num_frames, dtype=np.int32)
    solver_function_evaluations = np.empty(num_frames, dtype=np.int32)
    solver_jacobian_evaluations = np.empty(num_frames, dtype=np.int32)
    solver_objective = np.empty(num_frames, dtype=np.float64)
    solver_messages: dict[str, str] = {}

    for output_index, frame in enumerate(frames):
        if frame.hand_side != active_config.hand.side:
            raise ValueError(
                f"Recording frame {start_frame + output_index} is {frame.hand_side!r}, "
                f"but config expects {active_config.hand.side!r}"
            )

        retargeter.update_targets(frame.landmarks_3d, hand_side=frame.hand_side)
        landmarks = retargeter.get_target_landmarks()
        if landmarks is None:
            raise RuntimeError("Retargeter did not expose effective target landmarks")
        effective_landmarks[output_index] = landmarks
        human_direction_values, _ = directions_from_points(landmarks, HUMAN_COMMON_DIRECTION_PAIRS)
        human_directions[output_index] = human_direction_values

        qpos[output_index] = retargeter.solve()
        diagnostic = retargeter.get_last_solve_diagnostics()
        if diagnostic is None:
            raise RuntimeError("Retargeter did not expose SLSQP diagnostics")
        (
            optimizer_time_ns[output_index],
            solver_success[output_index],
            solver_status[output_index],
            solver_iterations[output_index],
            solver_function_evaluations[output_index],
            solver_jacobian_evaluations[output_index],
            solver_objective[output_index],
        ) = _diagnostic_values(diagnostic)
        solver_messages[str(diagnostic.status)] = diagnostic.message

        robot_direction_values = geometry.common_direction_vectors()
        robot_directions[output_index] = robot_direction_values
        frame_direction_errors, frame_direction_valid = direction_errors_degrees(
            human_direction_values,
            robot_direction_values,
        )
        direction_error_deg[output_index] = frame_direction_errors
        direction_valid[output_index] = frame_direction_valid
        objective_targets = retargeter.get_target_directions()
        if objective_targets is None:
            raise RuntimeError("Retargeter did not expose objective direction targets")
        objective_errors, _ = direction_errors_degrees(
            objective_targets,
            retargeter.get_robot_vectors(),
        )
        objective_direction_error_deg[output_index] = objective_errors

        human_points = landmarks[list(HUMAN_FINGERTIP_INDICES)]
        robot_points = geometry.fingertip_positions()
        human_keypoints[output_index] = human_points
        robot_keypoints[output_index] = robot_points
        human_origin = landmarks[0]
        robot_origin = geometry.palm_origin()
        human_scale = float(np.linalg.norm(landmarks[9] - human_origin))
        robot_scale = geometry.palm_scale()
        human_palm_scale[output_index] = human_scale
        robot_palm_scale[output_index] = robot_scale
        keypoint_error[output_index] = normalized_semantic_keypoint_errors(
            human_points,
            robot_points,
            human_origin=human_origin,
            robot_origin=robot_origin,
            human_scale=human_scale,
            robot_scale=robot_scale,
        )

        human_tip_distances[output_index] = np.asarray(
            [np.linalg.norm(landmarks[second] - landmarks[first]) for first, second in HUMAN_PINCH_PAIRS],
            dtype=np.float64,
        )
        robot_tip_site_distances[output_index] = geometry.fingertip_site_distances()
        robot_tip_surface_distances[output_index] = geometry.fingertip_surface_distances()
        target_tip_distances[output_index] = _distance_targets_by_pair(retargeter)

        human_rotation = geometry.human_frame_rotation(landmarks)
        robot_rotation = geometry.robot_frame_rotation()
        if human_rotation is not None and robot_rotation is not None:
            human_frame_rotation[output_index] = human_rotation
            robot_frame_rotation[output_index] = robot_rotation
            thumb_frame_error_deg[output_index] = rotation_geodesic_degrees(
                human_rotation,
                robot_rotation,
            )

    frame_indices = np.arange(start_frame, stop_frame, dtype=np.int64)
    arrays = {
        "frame_index": frame_indices,
        "timestamp_s": frame_indices.astype(np.float64) / max(fps, 1),
        "qpos": qpos,
        "effective_landmarks": effective_landmarks,
        "human_directions": human_directions,
        "robot_directions": robot_directions,
        "direction_error_deg": direction_error_deg,
        "direction_valid": direction_valid,
        "objective_direction_error_deg": objective_direction_error_deg,
        "human_keypoints": human_keypoints,
        "robot_keypoints": robot_keypoints,
        "keypoint_error": keypoint_error,
        "human_palm_scale": human_palm_scale,
        "robot_palm_scale": robot_palm_scale,
        "human_tip_distances": human_tip_distances,
        "robot_tip_site_distances": robot_tip_site_distances,
        "robot_tip_surface_distances": robot_tip_surface_distances,
        "target_tip_distances": target_tip_distances,
        "human_frame_rotation": human_frame_rotation,
        "robot_frame_rotation": robot_frame_rotation,
        "thumb_frame_error_deg": thumb_frame_error_deg,
        "optimizer_time_ns": optimizer_time_ns,
        "solver_success": solver_success,
        "solver_status": solver_status,
        "solver_iterations": solver_iterations,
        "solver_function_evaluations": solver_function_evaluations,
        "solver_jacobian_evaluations": solver_jacobian_evaluations,
        "solver_objective": solver_objective,
    }
    common_config_path = PROJECT_ROOT / "configs" / "retargeting" / "base" / "_universal_common.yaml"
    metadata = {
        "experiment_id": experiment_id,
        "hand_name": active_config.hand.name,
        "hand_side": active_config.hand.side,
        "method": method,
        "config_path": str(config_path_obj),
        "recording_path": str(recording_path_obj),
        "recording_sha256": recording_sha256,
        "annotation_path": None if annotation_path_obj is None else str(annotation_path_obj),
        "annotation_sha256": None if annotation_path_obj is None else sha256_file(annotation_path_obj),
        "recording_fps": fps,
        "recording_num_frames": int(recording["num_frames"]),
        "selected_start_frame": start_frame,
        "selected_stop_frame_exclusive": stop_frame,
        "warmup_frames": min(warmup_frames, len(frames)),
        "contact_threshold_m": contact_threshold_m,
        "human_pinch_threshold_m": human_pinch_threshold_m,
        "minimum_contact_duration_s": minimum_contact_duration_s,
        "human_direction_pairs": [list(pair) for pair in HUMAN_COMMON_DIRECTION_PAIRS],
        "human_keypoint_indices": list(HUMAN_FINGERTIP_INDICES),
        "human_pinch_pairs": [list(pair) for pair in HUMAN_PINCH_PAIRS],
        "model": {
            "nq": hand_model.nq,
            "nv": hand_model.nv,
            "nu": hand_model.nu,
            "independent_qpos": retargeter.get_independent_dof(),
            "mimic_joints": len(hand_model.mimic_joints),
            "initial_qpos": initial_qpos.tolist(),
        },
        "evaluation_geometry": geometry.describe(),
        "full_config": full_config_snapshot,
        "active_config": active_config_snapshot,
        "resolved_active_config": asdict(retargeter.config),
        "shared_config_sha256": sha256_file(common_config_path),
        "solver_messages": solver_messages,
        "summary": {
            "mean_direction_error_deg": float(np.nanmean(direction_error_deg)),
            "mean_keypoint_error": float(np.nanmean(keypoint_error)),
            "mean_solve_time_ms": float(np.mean(optimizer_time_ns) / 1e6),
            "solver_success_rate": float(np.mean(solver_success)),
        },
        "environment": runtime_environment_metadata(PROJECT_ROOT),
    }
    arrays_path, metadata_path = save_quantitative_result(
        output_base,
        arrays=arrays,
        metadata=metadata,
    )
    return QuantitativeRunSummary(
        hand_name=active_config.hand.name,
        method=method,
        num_frames=num_frames,
        mean_direction_error_deg=float(np.nanmean(direction_error_deg)),
        mean_keypoint_error=float(np.nanmean(keypoint_error)),
        mean_solve_time_ms=float(np.mean(optimizer_time_ns) / 1e6),
        solver_success_rate=float(np.mean(solver_success)),
        arrays_path=arrays_path,
        metadata_path=metadata_path,
    )
