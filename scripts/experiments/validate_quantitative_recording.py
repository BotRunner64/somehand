"""Validate the frozen PICO recording before running robot-hand outputs."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.domain.preprocessing import preprocess_landmarks
from somehand.domain.quantitative_metrics import HUMAN_PINCH_PAIRS, longest_true_run
from somehand.domain.quantitative_protocol import QUANTITATIVE_ACTIONS
from somehand.infrastructure.artifacts import load_hand_recording_artifact
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.quantitative_artifacts import load_annotations, sha256_file
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest
from somehand.infrastructure.vector_solver_primitives import TemporalFilter


PINCH_ACTION_TO_PAIR = {
    "thumb_index_pinch": 0,
    "thumb_middle_pinch": 1,
    "thumb_ring_pinch": 2,
    "thumb_little_pinch": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a quantitative PICO recording")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml"),
    )
    parser.add_argument("--max-repeated-frames", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    recording = load_hand_recording_artifact(str(manifest.recording_path))
    annotations = load_annotations(manifest.annotation_path)
    if sha256_file(manifest.recording_path) != annotations.get("recording_sha256"):
        raise ValueError("Recording hash does not match annotations")
    if int(recording["fps"]) != manifest.sample_rate_hz:
        raise ValueError("Recording fps does not match manifest")

    frames = list(recording["frames"])
    expected_frames = int(annotations["num_frames"])
    if len(frames) != expected_frames or int(recording["num_frames"]) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} complete frames, got {len(frames)} "
            f"(metadata num_frames={recording['num_frames']})"
        )
    if any(frame.hand_side != "right" for frame in frames):
        raise ValueError("Formal recording must contain only right-hand frames")

    episodes = list(annotations["episodes"])
    counts = Counter(str(episode["action"]) for episode in episodes)
    expected_counts = {action: 5 for action in QUANTITATIVE_ACTIONS}
    if counts != expected_counts:
        raise ValueError(f"Expected five episodes per action, got {dict(counts)}")
    previous_stop = 0
    for episode in episodes:
        start = int(episode["start_frame"])
        stop = int(episode["end_frame_exclusive"])
        hold_start = int(episode["hold_start_frame"])
        hold_stop = int(episode["hold_end_frame_exclusive"])
        if not (previous_stop == start <= hold_start < hold_stop <= stop <= len(frames)):
            raise ValueError(f"Invalid or non-contiguous episode bounds: {episode}")
        previous_stop = stop
    if previous_stop != len(frames):
        raise ValueError("Episode annotations do not cover the full recording")

    reference_spec = next(hand for hand in manifest.hands if hand.name == manifest.reference_hand)
    reference_config = load_retargeting_config(str(reference_spec.config_path))
    landmark_filter = TemporalFilter(alpha=reference_config.preprocess.temporal_filter_alpha)
    effective_landmarks = np.asarray(
        [
            landmark_filter.filter(preprocess_landmarks(frame.landmarks_3d, hand_side=frame.hand_side))
            for frame in frames
        ],
        dtype=np.float64,
    )

    raw_landmarks = np.asarray([frame.landmarks_3d for frame in frames], dtype=np.float64)
    repeated = np.all(np.isclose(np.diff(raw_landmarks, axis=0), 0.0, atol=1e-12), axis=(1, 2))
    max_repeated = longest_true_run(repeated)
    if max_repeated > args.max_repeated_frames:
        raise ValueError(
            f"Recording contains {max_repeated + 1} identical consecutive samples; "
            f"limit is {args.max_repeated_frames + 1}"
        )

    required_active_frames = max(
        1,
        math.ceil(manifest.sample_rate_hz * manifest.minimum_contact_duration_s),
    )
    for episode in episodes:
        action = str(episode["action"])
        pair_index = PINCH_ACTION_TO_PAIR.get(action)
        if pair_index is None:
            continue
        start = int(episode["hold_start_frame"])
        stop = int(episode["hold_end_frame_exclusive"])
        first, second = HUMAN_PINCH_PAIRS[pair_index]
        distances = np.linalg.norm(
            effective_landmarks[start:stop, second] - effective_landmarks[start:stop, first],
            axis=1,
        )
        active_run = longest_true_run(distances <= manifest.human_pinch_threshold_m)
        if active_run < required_active_frames:
            raise ValueError(
                f"{action} repetition {episode['repetition']} has only {active_run} consecutive active frames; "
                f"requires {required_active_frames} below "
                f"{manifest.human_pinch_threshold_m * 1000:.1f} mm"
            )
        print(
            f"{action:24s} repetition={episode['repetition']}"
            f" min_human_gap={1000.0 * float(np.min(distances)):.2f} mm"
            f" active_run={active_run}"
        )

    print(
        f"Recording validation passed: frames={len(frames)}, episodes={len(episodes)}, "
        f"longest_identical_run={max_repeated + 1}"
    )


if __name__ == "__main__":
    main()
