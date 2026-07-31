import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.domain.quantitative_protocol import QUANTITATIVE_ACTIONS
from somehand.infrastructure.quantitative_artifacts import (
    save_annotations,
    save_quantitative_result,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _episodes() -> list[dict[str, int | str]]:
    episodes = []
    cursor = 0
    for action in QUANTITATIVE_ACTIONS:
        for repetition in range(1, 6):
            episodes.append(
                {
                    "action": action,
                    "repetition": repetition,
                    "start_frame": cursor,
                    "transition_start_frame": cursor + 1,
                    "hold_start_frame": cursor + 2,
                    "hold_end_frame_exclusive": cursor + 3,
                    "end_frame_exclusive": cursor + 4,
                }
            )
            cursor += 4
    return episodes


def _arrays(method: str, episodes: list[dict[str, int | str]]) -> dict[str, np.ndarray]:
    num_frames = 140
    human_tip_distances = np.full((num_frames, 4), 0.05)
    robot_surface_distances = np.full((num_frames, 4), 0.02)
    robot_site_distances = np.full((num_frames, 4), 0.03)
    pinch_pairs = {
        "thumb_index_pinch": 0,
        "thumb_middle_pinch": 1,
        "thumb_ring_pinch": 2,
        "thumb_little_pinch": 3,
    }
    for episode in episodes:
        pair_index = pinch_pairs.get(str(episode["action"]))
        if pair_index is None:
            continue
        hold_frame = int(episode["hold_start_frame"])
        human_tip_distances[hold_frame, pair_index] = 0.01
        if method == "full":
            robot_surface_distances[hold_frame, pair_index] = 0.0005
            robot_site_distances[hold_frame, pair_index] = 0.005

    frame_error = 5.0 if method == "full" else 20.0
    return {
        "frame_index": np.arange(num_frames),
        "timestamp_s": np.arange(num_frames) / 10.0,
        "direction_error_deg": np.full((num_frames, 5), 10.0),
        "keypoint_error": np.full((num_frames, 5), 0.2),
        "human_tip_distances": human_tip_distances,
        "robot_tip_surface_distances": robot_surface_distances,
        "robot_tip_site_distances": robot_site_distances,
        "optimizer_time_ns": np.full(num_frames, 2_000_000),
        "solver_success": np.ones(num_frames, dtype=bool),
        "thumb_frame_error_deg": np.full(num_frames, frame_error),
    }


def test_summary_script_builds_all_tables_from_frozen_artifacts(tmp_path):
    recording_path = tmp_path / "input.pkl"
    recording_path.write_bytes(b"frozen recording")
    episodes = _episodes()
    annotation_path = tmp_path / "input.annotations.json"
    save_annotations(
        annotation_path,
        {
            "experiment_id": "test_quantitative",
            "recording_sha256": sha256_file(recording_path),
            "sample_rate_hz": 10,
            "num_frames": 140,
            "episodes": episodes,
        },
    )
    annotation_hash = sha256_file(annotation_path)
    result_dir = tmp_path / "results"
    hand_name = "linkerhand_l20_right"
    for method in ("full", "no_distance", "no_frame"):
        save_quantitative_result(
            result_dir / hand_name / method,
            arrays=_arrays(method, episodes),
            metadata={
                "experiment_id": "test_quantitative",
                "hand_name": hand_name,
                "method": method,
                "recording_sha256": sha256_file(recording_path),
                "annotation_sha256": annotation_hash,
                "recording_num_frames": 140,
                "recording_fps": 10,
                "contact_threshold_m": 0.001,
                "human_pinch_threshold_m": 0.020,
                "minimum_contact_duration_s": 0.1,
                "model": {
                    "independent_qpos": 16,
                    "nq": 21,
                    "mimic_joints": 5,
                },
            },
        )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                'experiment_id: "test_quantitative"',
                f'recording: "{recording_path}"',
                f'annotations: "{annotation_path}"',
                f'output_dir: "{result_dir}"',
                "sample_rate_hz: 10",
                "warmup_frames: 1",
                "contact_threshold_m: 0.001",
                "human_pinch_threshold_m: 0.020",
                "minimum_contact_duration_s: 0.1",
                f'reference_hand: "{hand_name}"',
                "hands:",
                f'  - config: "{PROJECT_ROOT / "configs/retargeting/right/linkerhand_l20_right.yaml"}"',
                "    main_table: true",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "summarize_quantitative.py"),
            "--manifest",
            str(manifest_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary_dir = result_dir / "summary"
    assert (summary_dir / "cross_hand_all.csv").exists()
    assert (summary_dir / "cross_hand_main.csv").exists()
    assert (summary_dir / "distance_ablation.csv").exists()
    assert (summary_dir / "distance_ablation_curve.csv").exists()
    assert (summary_dir / "frame_ablation.csv").exists()
    assert (summary_dir / "distance_ablation.png").exists()
    assert (summary_dir / "frame_ablation.png").exists()
    with (summary_dir / "cross_hand_main.csv").open(
        encoding="utf-8",
        newline="",
    ) as file_obj:
        cross_rows = list(csv.DictReader(file_obj))
    assert "pinch_success_percent" not in cross_rows[0]
    assert float(cross_rows[0]["site_distance_mae_mm"]) == pytest.approx(5.0)
    with (summary_dir / "distance_ablation.csv").open(
        encoding="utf-8",
        newline="",
    ) as file_obj:
        distance_rows = list(csv.DictReader(file_obj))
    assert float(distance_rows[0]["site_distance_mae_mm"]) == pytest.approx(20.0)
    assert float(distance_rows[1]["site_distance_mae_mm"]) == pytest.approx(5.0)


def test_available_summary_uses_continuous_metrics_without_annotations(tmp_path):
    recording_path = tmp_path / "existing.pkl"
    recording_path.write_bytes(b"existing recording")
    recording_hash = sha256_file(recording_path)
    result_dir = tmp_path / "results"
    hand_name = "linkerhand_l20_right"
    num_frames = 30
    human_tip_distances = np.full((num_frames, 4), 0.05)
    human_tip_distances[2:6, 0] = 0.01
    human_tip_distances[12:15, 2] = 0.01

    for method in ("full", "no_distance", "no_frame"):
        site_error = 0.02 if method == "no_distance" else 0.001
        robot_site_distances = human_tip_distances + site_error
        save_quantitative_result(
            result_dir / hand_name / method,
            arrays={
                "frame_index": np.arange(num_frames),
                "timestamp_s": np.arange(num_frames) / 10.0,
                "direction_error_deg": np.full((num_frames, 5), 10.0),
                "keypoint_error": np.full((num_frames, 5), 0.2),
                "human_tip_distances": human_tip_distances,
                "robot_tip_site_distances": robot_site_distances,
                "optimizer_time_ns": np.full(num_frames, 2_000_000),
                "solver_success": np.ones(num_frames, dtype=bool),
                "thumb_frame_error_deg": np.full(
                    num_frames,
                    20.0 if method == "no_frame" else 5.0,
                ),
            },
            metadata={
                "experiment_id": "test_available",
                "hand_name": hand_name,
                "method": method,
                "recording_sha256": recording_hash,
                "recording_num_frames": num_frames,
                "recording_fps": 10,
                "model": {
                    "independent_qpos": 16,
                    "nq": 21,
                    "mimic_joints": 5,
                },
            },
        )

    manifest_path = tmp_path / "available.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                'experiment_id: "test_available"',
                f'recording: "{recording_path}"',
                f'output_dir: "{result_dir}"',
                "sample_rate_hz: 10",
                "warmup_frames: 1",
                "contact_threshold_m: 0.001",
                "human_pinch_threshold_m: 0.020",
                "minimum_contact_duration_s: 0.1",
                f'reference_hand: "{hand_name}"',
                "hands:",
                f'  - config: "{PROJECT_ROOT / "configs/retargeting/right/linkerhand_l20_right.yaml"}"',
                "    main_table: true",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "experiments"
                / "summarize_available_quantitative.py"
            ),
            "--manifest",
            str(manifest_path),
            "--no-plots",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary_dir = result_dir / "summary_available"
    with (summary_dir / "summary_available.json").open(encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    assert "detected_pinch_bouts" not in summary
    assert (
        summary["distance_metric_scope"]
        == "all_available_frames_and_four_thumb_finger_pairs"
    )
    assert not (summary_dir / "pinch_bouts_detected.csv").exists()
    with (summary_dir / "distance_ablation_available.csv").open(
        encoding="utf-8",
        newline="",
    ) as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["method"] == "no_distance"
    assert float(rows[0]["site_distance_mae_mm"]) == pytest.approx(20.0)
    assert rows[1]["method"] == "full"
    assert float(rows[1]["site_distance_mae_mm"]) == pytest.approx(1.0)
    assert "pinch_success_percent" not in rows[1]
