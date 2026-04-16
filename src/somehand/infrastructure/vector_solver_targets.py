"""Target construction helpers for vector retargeting."""

from __future__ import annotations

import math

import numpy as np

from somehand.domain import preprocess_landmarks


def human_distance_scale(landmarks: np.ndarray) -> float:
    return float(
        np.linalg.norm(landmarks[10] - landmarks[9])
        + np.linalg.norm(landmarks[11] - landmarks[10])
        + np.linalg.norm(landmarks[12] - landmarks[11])
    )


def orthonormalize_frame_axes(
    primary_vector: np.ndarray,
    secondary_vector: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    primary_norm = np.linalg.norm(primary_vector)
    if primary_norm < 1e-8:
        return None, None
    primary_axis = primary_vector / primary_norm
    secondary_rejected = secondary_vector - np.dot(secondary_vector, primary_axis) * primary_axis
    secondary_norm = np.linalg.norm(secondary_rejected)
    if secondary_norm < 1e-8:
        return primary_axis, None
    secondary_axis = secondary_rejected / secondary_norm
    return primary_axis, secondary_axis


def dist_activation(activation_type: str, threshold: float, raw_dist: float) -> float:
    if threshold <= 0.0:
        return 1.0
    if activation_type == "gaussian":
        sigma = threshold / 2.0
        return math.exp(-(raw_dist / sigma) ** 2)
    if activation_type == "linear":
        return max(0.0, 1.0 - raw_dist / threshold)
    raise ValueError(f"unknown activation_type: '{activation_type}'")


def build_target_state(retargeter, landmarks_3d: np.ndarray, *, hand_side: str) -> None:
    landmarks = preprocess_landmarks(
        landmarks_3d,
        hand_side=hand_side,
    )
    landmarks = retargeter.landmark_filter.filter(landmarks)

    directions = np.empty((len(retargeter.human_vector_pairs), 3), dtype=np.float64)
    target_vectors = np.empty((len(retargeter.human_vector_pairs), 3), dtype=np.float64)
    vector_scale = retargeter._robot_vector_scale / max(
        float(np.linalg.norm(landmarks[retargeter._vector_scale_landmark_idx])),
        1e-6,
    )
    distance_scale = retargeter._robot_distance_scale / max(human_distance_scale(landmarks), 1e-6)
    for index, (origin_idx, target_idx) in enumerate(retargeter.human_vector_pairs):
        vector = landmarks[target_idx] - landmarks[origin_idx]
        norm = np.linalg.norm(vector)
        scale = vector_scale
        if retargeter._per_vector_loss_scales[index] > 0.0:
            scale = vector_scale * retargeter._per_vector_loss_scales[index]
        target_vectors[index] = scale * vector
        if norm < 1e-8:
            directions[index] = 0.0
        else:
            directions[index] = vector / norm
    retargeter._target_directions = directions
    retargeter._target_vectors = target_vectors
    if retargeter._frame_human_indices:
        frame_primary = np.empty((len(retargeter._frame_human_indices), 3), dtype=np.float64)
        frame_secondary = np.empty((len(retargeter._frame_human_indices), 3), dtype=np.float64)
        for index, (origin_idx, primary_idx, secondary_idx) in enumerate(retargeter._frame_human_indices):
            primary_vector = landmarks[primary_idx] - landmarks[origin_idx]
            secondary_vector = landmarks[secondary_idx] - landmarks[origin_idx]
            primary_axis, secondary_axis = orthonormalize_frame_axes(primary_vector, secondary_vector)
            frame_primary[index] = 0.0 if primary_axis is None else primary_axis
            frame_secondary[index] = 0.0 if secondary_axis is None else secondary_axis
        retargeter._target_frame_primary_directions = frame_primary
        retargeter._target_frame_secondary_directions = frame_secondary
    else:
        retargeter._target_frame_primary_directions = None
        retargeter._target_frame_secondary_directions = None

    if retargeter._angle_landmarks:
        target_angles = np.zeros(len(retargeter._angle_landmarks))
        for index, (a, b, c) in enumerate(retargeter._angle_landmarks):
            v_ba = landmarks[a] - landmarks[b]
            v_bc = landmarks[c] - landmarks[b]
            norm_ba = np.linalg.norm(v_ba)
            norm_bc = np.linalg.norm(v_bc)
            if norm_ba < 1e-8 or norm_bc < 1e-8:
                flexion = 0.0
            else:
                cos_angle = np.clip(np.dot(v_ba, v_bc) / (norm_ba * norm_bc), -1.0, 1.0)
                flexion = np.pi - np.arccos(cos_angle)
            low, high = retargeter._angle_joint_ranges[index]
            normalized = flexion / np.pi
            if retargeter._angle_inverts[index]:
                normalized = 1.0 - normalized
            normalized = np.clip(normalized * retargeter._angle_scales[index], 0.0, 1.0)
            target_angles[index] = low + normalized * (high - low)
        retargeter._target_angles = target_angles
    else:
        retargeter._target_angles = None

    if retargeter._dist_human_pairs:
        target_distances = np.zeros(len(retargeter._dist_human_pairs))
        raw_human_distances = np.zeros(len(retargeter._dist_human_pairs))
        smoothed_activations = np.zeros(len(retargeter._dist_human_pairs))
        for index, (a, b) in enumerate(retargeter._dist_human_pairs):
            raw_dist = float(np.linalg.norm(landmarks[a] - landmarks[b]))
            raw_human_distances[index] = raw_dist
            if retargeter._dist_scale_modes[index] == "hand_scaled":
                target_distances[index] = retargeter._dist_scales[index] * raw_dist * distance_scale
            else:
                target_distances[index] = retargeter._dist_scales[index] * raw_dist
            raw_act = dist_activation(retargeter._dist_activation_types[index], retargeter._dist_thresholds[index], raw_dist)
            if retargeter._prev_activations is not None:
                smoothed_activations[index] = (
                    retargeter._activation_alpha * raw_act
                    + (1.0 - retargeter._activation_alpha) * retargeter._prev_activations[index]
                )
            else:
                smoothed_activations[index] = raw_act
        retargeter._target_distances = target_distances
        retargeter._raw_human_distances = raw_human_distances
        retargeter._smoothed_activations = smoothed_activations
        retargeter._prev_activations = smoothed_activations.copy()
    else:
        retargeter._target_distances = None
        retargeter._raw_human_distances = None
        retargeter._smoothed_activations = None
