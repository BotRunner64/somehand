"""Record the fixed PICO action protocol and frame-aligned annotations."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.domain.quantitative_protocol import (
    ProtocolDurations,
    build_protocol_episodes,
    protocol_phase_at_frame,
    protocol_total_frames,
)
from somehand.infrastructure.artifacts import save_hand_recording_artifact
from somehand.infrastructure.quantitative_artifacts import save_annotations, sha256_file
from somehand.runtime import create_pico_source
from somehand.runtime.source_sampling import FixedRateHandTrackingSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record the somehand quantitative PICO protocol")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "recordings" / "quantitative" / "pico_right_v1.pkl"),
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--sample-rate", type=int, default=80)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prepare-seconds", type=float, default=1.0)
    parser.add_argument("--transition-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--release-seconds", type=float, default=1.0)
    parser.add_argument("--countdown-seconds", type=int, default=5)
    parser.add_argument("--pico-host", default="0.0.0.0")
    parser.add_argument("--pico-port", type=int, default=63901)
    parser.add_argument("--pico-advertise-ip", default=None)
    parser.add_argument("--no-pico-discovery", action="store_true")
    parser.add_argument("--pico-timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser


def _print_cue(action: str, repetition: int, phase: str, *, repetitions: int) -> None:
    labels = {
        "prepare": "OPEN / PREPARE",
        "transition": "MOVE",
        "hold": "HOLD",
        "release": "RELEASE TO OPEN",
    }
    print(f"\a[{action} {repetition}/{repetitions}] {labels[phase]}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).resolve()
    annotation_path = (
        output_path.with_suffix(".annotations.json")
        if args.annotations is None
        else Path(args.annotations).resolve()
    )
    if not args.force:
        existing = [path for path in (output_path, annotation_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing protocol artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    durations = ProtocolDurations(
        prepare_s=args.prepare_seconds,
        transition_s=args.transition_seconds,
        hold_s=args.hold_seconds,
        release_s=args.release_seconds,
    )
    episodes = build_protocol_episodes(
        sample_rate_hz=args.sample_rate,
        repetitions=args.repetitions,
        durations=durations,
    )
    total_frames = protocol_total_frames(episodes)
    expected_seconds = total_frames / args.sample_rate
    print(
        f"Protocol: {len(episodes)} episodes, {total_frames} frames, "
        f"{expected_seconds:.1f}s at {args.sample_rate} Hz"
    )
    print("Keep the right hand active and return to a fully open pose during every prepare/release phase.")

    source = FixedRateHandTrackingSource(
        create_pico_source(
            hand_side="right",
            timeout=args.pico_timeout,
            host=args.pico_host,
            port=args.pico_port,
            discovery=not args.no_pico_discovery,
            advertise_ip=args.pico_advertise_ip,
        ),
        sample_fps=args.sample_rate,
    )
    frames = []
    missing_frames: list[int] = []
    try:
        print("Waiting for an active right-hand frame...")
        warmup = source.get_frame()
        if warmup.detection is None:
            raise RuntimeError("PICO connected but the right hand is not active")

        for remaining in range(max(args.countdown_seconds, 0), 0, -1):
            print(f"Starting in {remaining}...", flush=True)
            time.sleep(1.0)

        previous_cue: tuple[str, int, str] | None = None
        capture_started = time.monotonic()
        for frame_index in range(total_frames):
            episode, phase = protocol_phase_at_frame(frame_index, episodes)
            cue = (episode.action, episode.repetition, phase)
            if cue != previous_cue:
                _print_cue(*cue, repetitions=args.repetitions)
                previous_cue = cue

            source_frame = source.get_frame()
            if source_frame.detection is None:
                missing_frames.append(frame_index)
                continue
            frames.append(source_frame.detection)
        capture_elapsed_s = time.monotonic() - capture_started
        source_stats = source.stats_snapshot()
    finally:
        source.close()

    if missing_frames:
        raise RuntimeError(
            f"Recording had {len(missing_frames)} missing frames; artifacts were not saved. "
            "Keep the PICO hand active and record again."
        )
    if len(frames) != total_frames:
        raise RuntimeError(f"Expected {total_frames} frames, captured {len(frames)}")

    save_hand_recording_artifact(
        str(output_path),
        frames,
        source_fps=args.sample_rate,
        source_desc="pico://right",
        input_type="pico_quantitative_protocol",
        num_frames=total_frames,
        hand_side="right",
        num_detected=total_frames,
    )
    annotation_payload = {
        "experiment_id": "dex_retargeting_quantitative_v1",
        "recording_path": str(output_path),
        "recording_sha256": sha256_file(output_path),
        "sample_rate_hz": args.sample_rate,
        "num_frames": total_frames,
        "num_episodes": len(episodes),
        "repetitions": args.repetitions,
        "durations": asdict(durations),
        "capture_elapsed_s": capture_elapsed_s,
        "source_stats": source_stats,
        "episodes": [episode.to_dict() for episode in episodes],
    }
    save_annotations(annotation_path, annotation_payload)
    print(f"Saved annotations to {annotation_path}")
    print("Recording is complete. Do not edit either file after freezing their hashes.")


if __name__ == "__main__":
    main()
