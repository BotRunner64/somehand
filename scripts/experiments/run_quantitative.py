"""Run the frozen quantitative experiment matrix without visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.application.quantitative_experiment import run_quantitative_replay
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run somehand quantitative retargeting experiments")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml"),
        help="Experiment manifest YAML",
    )
    parser.add_argument("--recording", default=None, help="Optional recording override")
    parser.add_argument("--annotations", default=None, help="Optional annotation override")
    parser.add_argument("--output-dir", default=None, help="Optional result-directory override")
    parser.add_argument("--hand", action="append", default=[], help="Run only this config stem (repeatable)")
    parser.add_argument(
        "--method",
        action="append",
        choices=["full", "no_distance", "no_frame"],
        default=[],
        help="Run only this method (repeatable)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--allow-unannotated", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing hand/method result")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected matrix without running it")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    recording_path = manifest.recording_path if args.recording is None else Path(args.recording).resolve()
    annotation_path = (
        manifest.annotation_path
        if args.annotations is None
        else Path(args.annotations).resolve()
    )
    output_dir = manifest.output_dir if args.output_dir is None else Path(args.output_dir).resolve()
    warmup_frames = manifest.warmup_frames if args.warmup_frames is None else args.warmup_frames

    if not recording_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    if (
        annotation_path is not None
        and not annotation_path.exists()
        and not args.allow_unannotated
        and not args.dry_run
    ):
        raise FileNotFoundError(
            f"Annotations not found: {annotation_path}; pass --allow-unannotated only for smoke checks"
        )

    selected_hands = set(args.hand)
    selected_methods = set(args.method)
    runs = [
        (hand, method)
        for hand, method in manifest.planned_runs()
        if (not selected_hands or hand.name in selected_hands)
        and (not selected_methods or method in selected_methods)
    ]
    if not runs:
        raise ValueError("No runs matched the selected hand/method filters")

    for hand, method in runs:
        output_base = output_dir / hand.name / method
        print(f"{hand.name:28s} {method:12s} -> {output_base}")
        if args.dry_run:
            continue
        if output_base.with_suffix(".json").exists() and not args.force:
            print("  skipped: result already exists (pass --force to overwrite)")
            continue
        summary = run_quantitative_replay(
            config_path=str(hand.config_path),
            recording_path=str(recording_path),
            annotation_path=(
                None
                if annotation_path is None or not annotation_path.exists()
                else str(annotation_path)
            ),
            output_base=output_base,
            method=method,
            experiment_id=manifest.experiment_id,
            warmup_frames=warmup_frames,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            contact_threshold_m=manifest.contact_threshold_m,
            human_pinch_threshold_m=manifest.human_pinch_threshold_m,
            minimum_contact_duration_s=manifest.minimum_contact_duration_s,
        )
        print(
            "  done:"
            f" frames={summary.num_frames}"
            f" direction={summary.mean_direction_error_deg:.3f} deg"
            f" keypoint={summary.mean_keypoint_error:.4f}"
            f" solve={summary.mean_solve_time_ms:.3f} ms"
            f" success={summary.solver_success_rate:.2%}"
        )


if __name__ == "__main__":
    main()
