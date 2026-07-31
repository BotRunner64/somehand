import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.domain.quantitative_metrics import (
    bootstrap_mean_interval,
    direction_errors_degrees,
    longest_true_run,
    normalized_semantic_keypoint_errors,
    orthonormal_frame,
    pinch_episode_success,
    pinch_peak_contact_success,
    rotation_geodesic_degrees,
    wilson_interval,
)
from somehand.domain.quantitative_protocol import (
    ProtocolDurations,
    build_protocol_episodes,
    protocol_phase_at_frame,
    protocol_total_frames,
)


def test_direction_errors_cover_zero_right_and_straight_angles():
    human = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    robot = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [-4.0, 0.0, 0.0],
        ]
    )

    errors, valid = direction_errors_degrees(human, robot)

    assert valid.tolist() == [True, True, True]
    assert errors == pytest.approx([0.0, 90.0, 180.0])


def test_normalized_keypoint_error_is_scale_and_translation_invariant():
    human = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    human_origin = np.array([0.0, 0.0, 0.0])
    robot_origin = np.array([5.0, -3.0, 2.0])
    robot = robot_origin + 3.0 * human

    errors = normalized_semantic_keypoint_errors(
        human,
        robot,
        human_origin=human_origin,
        robot_origin=robot_origin,
        human_scale=1.0,
        robot_scale=3.0,
    )

    assert errors == pytest.approx([0.0, 0.0])


def test_frame_geodesic_recovers_known_rotation():
    reference = orthonormal_frame(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    estimate = orthonormal_frame(np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))

    assert reference is not None
    assert estimate is not None
    assert np.linalg.det(reference) == pytest.approx(1.0)
    assert rotation_geodesic_degrees(reference, estimate) == pytest.approx(90.0)


def test_pinch_success_requires_contiguous_contact():
    distances = np.array([0.02, 0.0005, 0.0005, 0.02, 0.0005, 0.0005])

    assert pinch_peak_contact_success(distances, contact_threshold=0.001)
    assert longest_true_run(distances <= 0.001) == 2
    assert pinch_episode_success(
        distances,
        contact_threshold=0.001,
        sample_rate_hz=20,
        minimum_contact_duration_s=0.1,
    )
    assert not pinch_episode_success(
        distances,
        contact_threshold=0.001,
        sample_rate_hz=30,
        minimum_contact_duration_s=0.1,
    )


def test_peak_pinch_success_accepts_brief_contact_but_rejects_near_miss():
    assert pinch_peak_contact_success(
        np.array([0.004, 0.0008, 0.003]),
        contact_threshold=0.001,
    )
    assert not pinch_peak_contact_success(
        np.array([0.004, 0.0012, 0.003]),
        contact_threshold=0.001,
    )


def test_episode_statistics_are_deterministic_and_bounded():
    first = bootstrap_mean_interval(np.array([1.0, 2.0, 3.0]), num_resamples=1000, seed=4)
    second = bootstrap_mean_interval(np.array([1.0, 2.0, 3.0]), num_resamples=1000, seed=4)
    low, high = wilson_interval(4, 5)

    assert first == second
    assert first[0] <= 2.0 <= first[1]
    assert 0.0 <= low <= 0.8 <= high <= 1.0


def test_protocol_contains_seven_actions_repeated_five_times():
    episodes = build_protocol_episodes(
        sample_rate_hz=80,
        repetitions=5,
        durations=ProtocolDurations(),
    )

    assert len(episodes) == 35
    assert protocol_total_frames(episodes) == 35 * 80 * 4.5
    episode, phase = protocol_phase_at_frame(episodes[0].hold_start_frame, episodes)
    assert episode.action == "fist"
    assert phase == "hold"
