"""Aggregate frozen experiment results and generate paper-ready tables/plots."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.domain.quantitative_metrics import bootstrap_mean_interval
from somehand.domain.quantitative_protocol import QUANTITATIVE_ACTIONS
from somehand.infrastructure.quantitative_artifacts import (
    load_annotations,
    load_quantitative_result,
    sha256_file,
)
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest

PINCH_ACTION_TO_PAIR = {
    "thumb_index_pinch": 0,
    "thumb_middle_pinch": 1,
    "thumb_ring_pinch": 2,
    "thumb_little_pinch": 3,
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_mask(
    frame_indices: np.ndarray,
    episode: dict[str, object],
    *,
    hold_only: bool,
) -> np.ndarray:
    start_key = "hold_start_frame" if hold_only else "start_frame"
    end_key = "hold_end_frame_exclusive" if hold_only else "end_frame_exclusive"
    return (frame_indices >= int(episode[start_key])) & (frame_indices < int(episode[end_key]))


def _load_run(
    manifest,
    *,
    hand_name: str,
    method: str,
    annotation_hash: str,
    recording_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata_path = manifest.output_dir / hand_name / f"{method}.json"
    arrays, metadata = load_quantitative_result(metadata_path)
    if metadata["experiment_id"] != manifest.experiment_id:
        raise ValueError(f"{metadata_path}: experiment_id mismatch")
    if metadata["method"] != method or metadata["hand_name"] != hand_name:
        raise ValueError(f"{metadata_path}: hand/method metadata mismatch")
    if metadata["recording_sha256"] != recording_hash:
        raise ValueError(f"{metadata_path}: recording hash mismatch")
    if metadata.get("annotation_sha256") != annotation_hash:
        raise ValueError(f"{metadata_path}: annotation hash mismatch")
    if int(metadata["recording_fps"]) != manifest.sample_rate_hz:
        raise ValueError(f"{metadata_path}: sample-rate mismatch")
    if len(arrays["frame_index"]) != int(metadata["recording_num_frames"]):
        raise ValueError(f"{metadata_path}: formal summaries require a complete recording replay")
    return arrays, metadata


def _distance_episode_rows(
    arrays: dict[str, np.ndarray],
    episodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame_indices = arrays["frame_index"]
    rows: list[dict[str, object]] = []
    for episode in episodes:
        action = str(episode["action"])
        pair_index = PINCH_ACTION_TO_PAIR.get(action)
        if pair_index is None:
            continue
        hold_mask = _episode_mask(frame_indices, episode, hold_only=True)
        human_distances = np.asarray(
            arrays["human_tip_distances"][hold_mask, pair_index],
            dtype=np.float64,
        )
        robot_distances = np.asarray(
            arrays["robot_tip_site_distances"][hold_mask, pair_index],
            dtype=np.float64,
        )
        if len(human_distances) == 0:
            raise ValueError(
                f"Annotated hold window is empty for {action} "
                f"repetition {episode['repetition']}"
            )
        absolute_errors = np.abs(robot_distances - human_distances)
        rows.append(
            {
                "action": action,
                "repetition": int(episode["repetition"]),
                "pair_index": pair_index,
                "mean_human_distance_m": float(np.mean(human_distances)),
                "mean_robot_site_distance_m": float(np.mean(robot_distances)),
                "mean_abs_site_distance_error_m": float(np.mean(absolute_errors)),
                "p95_abs_site_distance_error_m": float(
                    np.percentile(absolute_errors, 95)
                ),
                "hold_frames": len(human_distances),
            }
        )
    return rows


def _cross_hand_rows(
    manifest,
    annotations: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episodes = list(annotations["episodes"])
    annotation_hash = sha256_file(manifest.annotation_path)
    recording_hash = str(annotations["recording_sha256"])
    rows: list[dict[str, object]] = []
    main_rows: list[dict[str, object]] = []

    for hand in manifest.hands:
        arrays, metadata = _load_run(
            manifest,
            hand_name=hand.name,
            method="full",
            annotation_hash=annotation_hash,
            recording_hash=recording_hash,
        )
        frame_indices = arrays["frame_index"]
        direction_episode_means = np.asarray(
            [
                np.nanmean(arrays["direction_error_deg"][_episode_mask(frame_indices, episode, hold_only=False)])
                for episode in episodes
            ],
            dtype=np.float64,
        )
        keypoint_episode_means = np.asarray(
            [
                np.nanmean(arrays["keypoint_error"][_episode_mask(frame_indices, episode, hold_only=False)])
                for episode in episodes
            ],
            dtype=np.float64,
        )
        direction_low, direction_high = bootstrap_mean_interval(direction_episode_means)
        keypoint_low, keypoint_high = bootstrap_mean_interval(keypoint_episode_means)
        distance_rows = _distance_episode_rows(
            arrays,
            episodes,
        )
        distance_episode_means = np.asarray(
            [
                row["mean_abs_site_distance_error_m"]
                for row in distance_rows
            ],
            dtype=np.float64,
        )
        distance_low, distance_high = bootstrap_mean_interval(
            distance_episode_means
        )
        model = metadata["model"]
        row = {
            "hand": hand.name,
            "main_table": hand.main_table,
            "independent_dof": model["independent_qpos"],
            "joint_coordinates": model["nq"],
            "mimic_joints": model["mimic_joints"],
            "direction_error_deg": float(np.nanmean(arrays["direction_error_deg"])),
            "direction_error_ci95_low": direction_low,
            "direction_error_ci95_high": direction_high,
            "normalized_keypoint_error": float(np.nanmean(arrays["keypoint_error"])),
            "normalized_keypoint_error_ci95_low": keypoint_low,
            "normalized_keypoint_error_ci95_high": keypoint_high,
            "site_distance_mae_mm": 1000.0 * float(
                np.mean(distance_episode_means)
            ),
            "site_distance_mae_ci95_low_mm": 1000.0 * distance_low,
            "site_distance_mae_ci95_high_mm": 1000.0 * distance_high,
            "mean_solve_time_ms": float(np.mean(arrays["optimizer_time_ns"]) / 1e6),
            "p95_solve_time_ms": float(np.percentile(arrays["optimizer_time_ns"], 95) / 1e6),
            "solver_success_percent": 100.0 * float(np.mean(arrays["solver_success"])),
        }
        rows.append(row)
        if hand.main_table:
            main_rows.append({key: value for key, value in row.items() if key != "main_table"})
    return rows, main_rows


def _distance_ablation(
    manifest,
    annotations: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episodes = [
        episode for episode in annotations["episodes"] if episode["action"] == "thumb_index_pinch"
    ]
    annotation_hash = sha256_file(manifest.annotation_path)
    recording_hash = str(annotations["recording_sha256"])
    method_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    segment_start = int(episodes[0]["start_frame"])
    segment_stop = int(episodes[-1]["end_frame_exclusive"])
    for method in ("no_distance", "full"):
        arrays, _ = _load_run(
            manifest,
            hand_name=manifest.reference_hand,
            method=method,
            annotation_hash=annotation_hash,
            recording_hash=recording_hash,
        )
        episode_rows = _distance_episode_rows(
            arrays,
            episodes,
        )
        errors = np.asarray(
            [row["mean_abs_site_distance_error_m"] for row in episode_rows],
            dtype=np.float64,
        )
        robot_distances = np.asarray(
            [row["mean_robot_site_distance_m"] for row in episode_rows],
            dtype=np.float64,
        )
        human_distances = np.asarray(
            [row["mean_human_distance_m"] for row in episode_rows],
            dtype=np.float64,
        )
        low, high = bootstrap_mean_interval(errors)
        method_rows.append(
            {
                "method": method,
                "evaluation_scope": "annotated_thumb_index_hold_frames",
                "mean_human_tip_distance_mm": 1000.0
                * float(np.mean(human_distances)),
                "mean_robot_site_distance_mm": 1000.0
                * float(np.mean(robot_distances)),
                "site_distance_mae_mm": 1000.0 * float(np.mean(errors)),
                "site_distance_mae_ci95_low_mm": 1000.0 * low,
                "site_distance_mae_ci95_high_mm": 1000.0 * high,
            }
        )

        segment_mask = (arrays["frame_index"] >= segment_start) & (arrays["frame_index"] < segment_stop)
        for frame_index, timestamp_s, human_distance, robot_distance in zip(
            arrays["frame_index"][segment_mask],
            arrays["timestamp_s"][segment_mask],
            arrays["human_tip_distances"][segment_mask, 0],
            arrays["robot_tip_site_distances"][segment_mask, 0],
        ):
            curve_rows.append(
                {
                    "method": method,
                    "frame_index": int(frame_index),
                    "time_from_segment_start_s": float(timestamp_s - segment_start / manifest.sample_rate_hz),
                    "human_tip_distance_mm": 1000.0 * float(human_distance),
                    "robot_site_distance_mm": 1000.0 * float(robot_distance),
                    "absolute_tracking_error_mm": 1000.0
                    * abs(float(robot_distance) - float(human_distance)),
                }
            )
    return method_rows, curve_rows


def _frame_ablation(manifest, annotations: dict[str, object]) -> list[dict[str, object]]:
    episodes = [
        episode for episode in annotations["episodes"] if episode["action"] == "thumb_opposition"
    ]
    annotation_hash = sha256_file(manifest.annotation_path)
    recording_hash = str(annotations["recording_sha256"])
    rows: list[dict[str, object]] = []
    for method in ("no_frame", "full"):
        arrays, _ = _load_run(
            manifest,
            hand_name=manifest.reference_hand,
            method=method,
            annotation_hash=annotation_hash,
            recording_hash=recording_hash,
        )
        episode_means = np.asarray(
            [
                np.nanmean(
                    arrays["thumb_frame_error_deg"][
                        _episode_mask(arrays["frame_index"], episode, hold_only=True)
                    ]
                )
                for episode in episodes
            ],
            dtype=np.float64,
        )
        low, high = bootstrap_mean_interval(episode_means)
        rows.append(
            {
                "method": method,
                "mean_thumb_frame_error_deg": float(np.mean(episode_means)),
                "mean_thumb_frame_error_ci95_low_deg": low,
                "mean_thumb_frame_error_ci95_high_deg": high,
            }
        )
    return rows


def _plot_distance(curve_rows: list[dict[str, object]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    styles = {"no_distance": ("#D55E00", "w/o distance"), "full": ("#0072B2", "Full")}
    for method, (color, label) in styles.items():
        selected = [row for row in curve_rows if row["method"] == method]
        axis.plot(
            [row["time_from_segment_start_s"] for row in selected],
            [row["robot_site_distance_mm"] for row in selected],
            color=color,
            linewidth=1.4,
            label=label,
        )
    target_rows = [row for row in curve_rows if row["method"] == "full"]
    axis.plot(
        [row["time_from_segment_start_s"] for row in target_rows],
        [row["human_tip_distance_mm"] for row in target_rows],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Human target",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Thumb–index site distance (mm)")
    axis.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "distance_ablation.png", dpi=300)
    figure.savefig(output_dir / "distance_ablation.pdf")
    plt.close(figure)


def _plot_frame(rows: list[dict[str, object]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["w/o frame", "Full"]
    means = [float(row["mean_thumb_frame_error_deg"]) for row in rows]
    lower = [
        means[index] - float(row["mean_thumb_frame_error_ci95_low_deg"])
        for index, row in enumerate(rows)
    ]
    upper = [
        float(row["mean_thumb_frame_error_ci95_high_deg"]) - means[index]
        for index, row in enumerate(rows)
    ]
    figure, axis = plt.subplots(figsize=(3.8, 3.6))
    axis.bar(labels, means, color=["#D55E00", "#0072B2"], width=0.62)
    axis.errorbar(
        np.arange(2),
        means,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.0,
    )
    axis.set_ylabel("Mean thumb-frame error (deg)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "frame_ablation.png", dpi=300)
    figure.savefig(output_dir / "frame_ablation.pdf")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize somehand quantitative experiment results")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    annotations = load_annotations(manifest.annotation_path)
    if sha256_file(manifest.recording_path) != annotations["recording_sha256"]:
        raise ValueError("Frozen recording hash does not match the annotation file")
    if int(annotations["sample_rate_hz"]) != manifest.sample_rate_hz:
        raise ValueError("Annotation sample rate does not match the manifest")
    if len(annotations["episodes"]) != 35:
        raise ValueError("Formal protocol must contain exactly 35 episodes")
    action_counts = Counter(str(episode["action"]) for episode in annotations["episodes"])
    if action_counts != {action: 5 for action in QUANTITATIVE_ACTIONS}:
        raise ValueError(f"Formal protocol must contain five repetitions per action: {dict(action_counts)}")

    output_dir = (
        manifest.output_dir / "summary"
        if args.output_dir is None
        else Path(args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_rows, main_rows = _cross_hand_rows(manifest, annotations)
    distance_rows, curve_rows = _distance_ablation(manifest, annotations)
    frame_rows = _frame_ablation(manifest, annotations)

    _write_csv(output_dir / "cross_hand_all.csv", cross_rows)
    _write_csv(output_dir / "cross_hand_main.csv", main_rows)
    _write_csv(output_dir / "distance_ablation.csv", distance_rows)
    _write_csv(output_dir / "distance_ablation_curve.csv", curve_rows)
    _write_csv(output_dir / "frame_ablation.csv", frame_rows)
    if not args.no_plots:
        _plot_distance(curve_rows, output_dir)
        _plot_frame(frame_rows, output_dir)
    print(f"Saved quantitative tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
