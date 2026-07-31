"""Pure metrics used by the quantitative retargeting experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


HUMAN_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
HUMAN_PINCH_PAIRS: tuple[tuple[int, int], ...] = (
    (4, 8),
    (4, 12),
    (4, 16),
    (4, 20),
)
HUMAN_COMMON_DIRECTION_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (5, 8),
    (9, 12),
    (13, 16),
    (17, 20),
)


def directions_from_points(
    points: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build unit directions and a validity mask for semantic point pairs."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    directions = np.zeros((len(pairs), 3), dtype=np.float64)
    valid = np.zeros(len(pairs), dtype=bool)
    for index, (origin_index, target_index) in enumerate(pairs):
        vector = values[target_index] - values[origin_index]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            continue
        directions[index] = vector / norm
        valid[index] = True
    return directions, valid


def direction_errors_degrees(
    human_directions: np.ndarray,
    robot_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-direction angular errors in degrees and their validity mask."""
    human = np.asarray(human_directions, dtype=np.float64)
    robot = np.asarray(robot_directions, dtype=np.float64)
    if human.shape != robot.shape or human.ndim != 2 or human.shape[1] != 3:
        raise ValueError("human_directions and robot_directions must have matching shape (N, 3)")

    human_norms = np.linalg.norm(human, axis=1)
    robot_norms = np.linalg.norm(robot, axis=1)
    valid = (human_norms >= 1e-8) & (robot_norms >= 1e-8)
    errors = np.full(len(human), np.nan, dtype=np.float64)
    if np.any(valid):
        human_unit = human[valid] / human_norms[valid, None]
        robot_unit = robot[valid] / robot_norms[valid, None]
        cosines = np.sum(human_unit * robot_unit, axis=1)
        errors[valid] = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return errors, valid


def normalized_semantic_keypoint_errors(
    human_keypoints: np.ndarray,
    robot_keypoints: np.ndarray,
    *,
    human_origin: np.ndarray,
    robot_origin: np.ndarray,
    human_scale: float,
    robot_scale: float,
) -> np.ndarray:
    """Compare semantic points after translation and palm-scale normalization."""
    human = np.asarray(human_keypoints, dtype=np.float64)
    robot = np.asarray(robot_keypoints, dtype=np.float64)
    if human.shape != robot.shape or human.ndim != 2 or human.shape[1] != 3:
        raise ValueError("human_keypoints and robot_keypoints must have matching shape (K, 3)")
    if human_scale <= 1e-8 or robot_scale <= 1e-8:
        raise ValueError("human_scale and robot_scale must be positive")

    human_normalized = (human - np.asarray(human_origin, dtype=np.float64)) / human_scale
    robot_normalized = (robot - np.asarray(robot_origin, dtype=np.float64)) / robot_scale
    return np.linalg.norm(robot_normalized - human_normalized, axis=1)


def orthonormal_frame(
    primary_vector: np.ndarray,
    secondary_vector: np.ndarray,
) -> np.ndarray | None:
    """Construct a right-handed rotation matrix whose columns are frame axes."""
    primary = np.asarray(primary_vector, dtype=np.float64)
    secondary = np.asarray(secondary_vector, dtype=np.float64)
    primary_norm = float(np.linalg.norm(primary))
    if primary_norm < 1e-8:
        return None
    primary_axis = primary / primary_norm

    secondary_rejected = secondary - float(np.dot(secondary, primary_axis)) * primary_axis
    secondary_norm = float(np.linalg.norm(secondary_rejected))
    if secondary_norm < 1e-8:
        return None
    secondary_axis = secondary_rejected / secondary_norm
    tertiary_axis = np.cross(primary_axis, secondary_axis)
    tertiary_norm = float(np.linalg.norm(tertiary_axis))
    if tertiary_norm < 1e-8:
        return None
    tertiary_axis /= tertiary_norm
    return np.column_stack((primary_axis, secondary_axis, tertiary_axis))


def rotation_geodesic_degrees(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute SO(3) geodesic distance in degrees."""
    reference_rotation = np.asarray(reference, dtype=np.float64)
    estimate_rotation = np.asarray(estimate, dtype=np.float64)
    if reference_rotation.shape != (3, 3) or estimate_rotation.shape != (3, 3):
        raise ValueError("reference and estimate must have shape (3, 3)")
    cosine = (float(np.trace(estimate_rotation.T @ reference_rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest contiguous run of true values."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def pinch_peak_contact_success(
    distances: np.ndarray,
    *,
    contact_threshold: float,
) -> bool:
    """Return whether an episode reaches the contact threshold at least once."""
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be >= 0")
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    return bool(np.any(values <= contact_threshold))


def pinch_episode_success(
    distances: np.ndarray,
    *,
    contact_threshold: float,
    sample_rate_hz: float,
    minimum_contact_duration_s: float = 0.1,
) -> bool:
    """Return whether contact persists for a minimum duration within an episode."""
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be >= 0")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be > 0")
    if minimum_contact_duration_s < 0.0:
        raise ValueError("minimum_contact_duration_s must be >= 0")
    minimum_frames = max(1, int(math.ceil(sample_rate_hz * minimum_contact_duration_s)))
    return longest_true_run(np.asarray(distances) <= contact_threshold) >= minimum_frames


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for episode means."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be > 0")
    if len(samples) == 0:
        return float("nan"), float("nan")
    if len(samples) == 1:
        value = float(samples[0])
        return value, value

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(samples), size=(num_resamples, len(samples)))
    bootstrap_means = np.mean(samples[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return float(low), float(high)


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial success proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0, total]")
    proportion = successes / total
    z_squared = confidence_z * confidence_z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        confidence_z
        * math.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total))
        / denominator
    )
    return center - margin, center + margin
