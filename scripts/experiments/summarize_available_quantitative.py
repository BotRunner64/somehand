"""Summarize an exploratory quantitative pass over any available recording."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.infrastructure.quantitative_artifacts import (
    load_quantitative_result,
    sha256_file,
)
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(
    manifest,
    *,
    hand_name: str,
    method: str,
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
    if int(metadata["recording_fps"]) != manifest.sample_rate_hz:
        raise ValueError(f"{metadata_path}: sample-rate mismatch")
    selected_start = int(metadata.get("selected_start_frame", 0))
    selected_stop = int(
        metadata.get("selected_stop_frame_exclusive", len(arrays["frame_index"]))
    )
    if selected_start != 0 or selected_stop != len(arrays["frame_index"]):
        raise ValueError(
            f"{metadata_path}: exploratory summaries require all saved detection frames"
        )
    return arrays, metadata


def _distance_tracking_stats(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    human = np.asarray(arrays["human_tip_distances"], dtype=np.float64)
    robot = np.asarray(arrays["robot_tip_site_distances"], dtype=np.float64)
    if human.shape != robot.shape or human.ndim != 2 or human.shape[1] != 4:
        raise ValueError(
            "human_tip_distances and robot_tip_site_distances must have matching shape (N, 4)"
        )
    absolute_error_mm = 1000.0 * np.abs(robot - human)
    finite = absolute_error_mm[np.isfinite(absolute_error_mm)]
    if len(finite) == 0:
        raise ValueError("Site-distance tracking error contains no finite observations")
    return {
        "site_distance_mae_mm": float(np.mean(finite)),
        "site_distance_median_abs_error_mm": float(np.median(finite)),
        "site_distance_p95_abs_error_mm": float(np.percentile(finite, 95)),
        "site_distance_observations": len(finite),
    }


def _plot_distance_curve(
    output_dir: Path,
    full_arrays: dict[str, np.ndarray],
    no_distance_arrays: dict[str, np.ndarray],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = full_arrays["timestamp_s"]
    no_distance_error_mm = 1000.0 * np.nanmean(
        np.abs(
            no_distance_arrays["robot_tip_site_distances"]
            - no_distance_arrays["human_tip_distances"]
        ),
        axis=1,
    )
    full_error_mm = 1000.0 * np.nanmean(
        np.abs(
            full_arrays["robot_tip_site_distances"]
            - full_arrays["human_tip_distances"]
        ),
        axis=1,
    )
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    axis.plot(
        time_s,
        no_distance_error_mm,
        color="#D55E00",
        linewidth=1.1,
        label="w/o distance",
    )
    axis.plot(
        time_s,
        full_error_mm,
        color="#0072B2",
        linewidth=1.1,
        label="Full",
    )
    axis.set_xlabel("Time in available recording (s)")
    axis.set_ylabel("Mean site-distance absolute error (mm)")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "distance_ablation_available.png", dpi=300)
    figure.savefig(output_dir / "distance_ablation_available.pdf")
    plt.close(figure)


def _plot_frame_ablation(output_dir: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["w/o frame", "Full"]
    means = [
        float(next(row for row in rows if row["method"] == method)["mean_thumb_frame_error_deg"])
        for method in ("no_frame", "full")
    ]
    figure, axis = plt.subplots(figsize=(3.8, 3.6))
    axis.bar(labels, means, color=["#D55E00", "#0072B2"], width=0.62)
    axis.set_ylabel("All-frame thumb-frame error (deg)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "frame_ablation_available.png", dpi=300)
    figure.savefig(output_dir / "frame_ablation_available.pdf")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize available-data experiments without a fixed action protocol"
    )
    parser.add_argument(
        "--manifest",
        default=str(
            PROJECT_ROOT / "configs" / "experiments" / "quantitative_existing_pico.yaml"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    recording_hash = sha256_file(manifest.recording_path)
    reference_full, _ = _load_run(
        manifest,
        hand_name=manifest.reference_hand,
        method="full",
        recording_hash=recording_hash,
    )

    cross_hand_rows: list[dict[str, object]] = []
    for hand in manifest.hands:
        arrays, metadata = _load_run(
            manifest,
            hand_name=hand.name,
            method="full",
            recording_hash=recording_hash,
        )
        row = {
            "hand": hand.name,
            "independent_dof": metadata["model"]["independent_qpos"],
            "joint_coordinates": metadata["model"]["nq"],
            "mimic_joints": metadata["model"]["mimic_joints"],
            "direction_error_deg": float(np.nanmean(arrays["direction_error_deg"])),
            "normalized_keypoint_error": float(np.nanmean(arrays["keypoint_error"])),
            **_distance_tracking_stats(arrays),
            "mean_solve_time_ms": float(np.mean(arrays["optimizer_time_ns"]) / 1e6),
            "p95_solve_time_ms": float(
                np.percentile(arrays["optimizer_time_ns"], 95) / 1e6
            ),
            "solver_success_percent": 100.0 * float(np.mean(arrays["solver_success"])),
        }
        cross_hand_rows.append(row)

    distance_rows: list[dict[str, object]] = []
    distance_arrays: dict[str, dict[str, np.ndarray]] = {}
    for method in ("no_distance", "full"):
        arrays, _ = _load_run(
            manifest,
            hand_name=manifest.reference_hand,
            method=method,
            recording_hash=recording_hash,
        )
        distance_arrays[method] = arrays
        distance_rows.append(
            {
                "method": method,
                "evaluation_scope": "all_available_frames",
                **_distance_tracking_stats(arrays),
            }
        )

    frame_rows: list[dict[str, object]] = []
    for method in ("no_frame", "full"):
        arrays, _ = _load_run(
            manifest,
            hand_name=manifest.reference_hand,
            method=method,
            recording_hash=recording_hash,
        )
        errors = arrays["thumb_frame_error_deg"]
        finite = errors[np.isfinite(errors)]
        frame_rows.append(
            {
                "method": method,
                "evaluation_scope": "all_available_frames",
                "num_frames": len(finite),
                "mean_thumb_frame_error_deg": float(np.mean(finite)),
                "median_thumb_frame_error_deg": float(np.median(finite)),
                "p95_thumb_frame_error_deg": float(np.percentile(finite, 95)),
            }
        )

    output_dir = (
        manifest.output_dir / "summary_available"
        if args.output_dir is None
        else Path(args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cross_hand_available.csv", cross_hand_rows)
    _write_csv(output_dir / "distance_ablation_available.csv", distance_rows)
    _write_csv(output_dir / "frame_ablation_available.csv", frame_rows)
    with (output_dir / "summary_available.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "experiment_id": manifest.experiment_id,
                "mode": "exploratory_available_recording",
                "recording_path": str(manifest.recording_path),
                "recording_sha256": recording_hash,
                "sample_rate_hz": manifest.sample_rate_hz,
                "num_frames": len(reference_full["frame_index"]),
                "frame_ablation_scope": "all_available_frames",
                "distance_metric_scope": "all_available_frames_and_four_thumb_finger_pairs",
                "distance_metric_definition": (
                    "absolute difference between robot fingertip-site distance and "
                    "the corresponding human fingertip distance"
                ),
                "caveat": (
                    "The available recording has no action labels; all continuous "
                    "metrics therefore use every saved frame."
                ),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file_obj.write("\n")
    if not args.no_plots:
        _plot_distance_curve(
            output_dir,
            distance_arrays["full"],
            distance_arrays["no_distance"],
        )
        _plot_frame_ablation(output_dir, frame_rows)
    print(
        f"Saved available-data continuous-metric summary for "
        f"{len(cross_hand_rows)} hands to {output_dir}"
    )


if __name__ == "__main__":
    main()
