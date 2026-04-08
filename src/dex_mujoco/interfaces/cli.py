"""Thin CLI interface over the application-layer pipeline."""

from __future__ import annotations

import argparse

from dex_mujoco.application import RetargetingEngine, RetargetingSession
from dex_mujoco.infrastructure import (
    AsyncLandmarkOutputSink,
    OpenCvPreviewWindow,
    RecordingHandTrackingSource,
    RobotHandOutputSink,
    TerminalRecordingController,
    create_hc_mocap_bvh_source,
    create_hc_mocap_udp_source,
    create_pico_source,
    create_recording_source,
    save_hand_recording_artifact,
)
from dex_mujoco.infrastructure.sources import MediaPipeInputSource
from dex_mujoco.paths import DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to retargeting config YAML",
    )
    parser.add_argument(
        "-H",
        "--hand",
        choices=["Left", "Right"],
        default="Right",
        help="Operator hand to retarget",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        help="Output pickle file for recorded hand-tracking frames",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dex-retarget", description="Unified dex hand retargeting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    webcam = subparsers.add_parser("webcam", help="Retarget from a live webcam stream")
    _add_common_args(webcam)
    webcam.add_argument("--camera", type=int, default=0, help="Webcam device index")
    webcam.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if capture reports the opposite hand",
    )

    video = subparsers.add_parser("video", help="Retarget from a video file")
    _add_common_args(video)
    video.add_argument("--video", required=True, help="Path to input video file")
    video.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if this video reports the opposite hand",
    )

    replay = subparsers.add_parser("replay", help="Replay a saved hand-tracking recording")
    _add_common_args(replay)
    replay.add_argument("--recording", required=True, help="Path to a saved hand-tracking recording")
    replay.add_argument("--loop", action="store_true", help="Loop the saved recording indefinitely")

    pico = subparsers.add_parser("pico", help="Retarget from live PICO hand tracking via XRoboToolkit")
    _add_common_args(pico)
    pico.add_argument(
        "--pico-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds while waiting for PICO hand-tracking frames",
    )

    hc_mocap = subparsers.add_parser("hc-mocap", help="Retarget from Teleopit hc_mocap data")
    hc_subparsers = hc_mocap.add_subparsers(dest="hc_command", required=True)

    hc_bvh = hc_subparsers.add_parser("bvh", help="Retarget from an offline hc_mocap BVH file")
    _add_common_args(hc_bvh)
    hc_bvh.add_argument("--bvh", required=True, help="Offline hc_mocap BVH file")
    hc_bvh.add_argument(
        "--teleopit-root",
        default=None,
        help="Optional Teleopit repo root for offline BVH loading if package is not installed",
    )
    hc_bvh.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between offline BVH frames using the source fps",
    )
    hc_bvh.add_argument("--loop", action="store_true", help="Loop offline hc_mocap BVH input indefinitely")

    hc_udp = hc_subparsers.add_parser("udp", help="Retarget from a live hc_mocap UDP stream")
    _add_common_args(hc_udp)
    hc_udp.add_argument(
        "--reference-bvh",
        default=str(DEFAULT_HC_MOCAP_REFERENCE_BVH),
        help="Reference hc_mocap BVH file for UDP joint ordering",
    )
    hc_udp.add_argument("--udp-host", default="", help="UDP bind host for hc_mocap input")
    hc_udp.add_argument("--udp-port", type=int, default=1118, help="UDP port for hc_mocap input")
    hc_udp.add_argument("--udp-timeout", type=float, default=30.0, help="UDP startup timeout in seconds")
    hc_udp.add_argument(
        "--udp-stats-every",
        type=int,
        default=120,
        help="Print UDP receive statistics every N processed frames (0 disables)",
    )

    return parser


def _build_engine(args: argparse.Namespace, *, input_type: str) -> RetargetingEngine:
    return RetargetingEngine.from_config_path(args.config, input_type=input_type)


def _build_session(engine: RetargetingEngine, *, visualize: bool, show_preview: bool, key_callback=None) -> RetargetingSession:
    sinks = []
    frame_sinks = []
    if visualize:
        frame_sinks.append(AsyncLandmarkOutputSink())
        sinks.extend(
            [
                RobotHandOutputSink(engine.hand_model, key_callback=key_callback),
            ]
        )
    preview_window = OpenCvPreviewWindow() if show_preview else None
    return RetargetingSession(engine, sinks=sinks, frame_sinks=frame_sinks, preview_window=preview_window)


def _print_startup(engine: RetargetingEngine, *, source_desc: str, tracking_desc: str, extra_lines: list[str] | None = None) -> None:
    details = engine.describe()
    print(f"Model: {details['model_name']} ({details['dof']} DOF)")
    print(f"Retargeting: {details['vector_pairs']} vector pairs")
    print(f"Input source: {source_desc}")
    print(tracking_desc)
    if extra_lines:
        for line in extra_lines:
            print(line)


def _finalize_run(
    args: argparse.Namespace,
    *,
    summary,
    source,
) -> None:
    print(f"Processed {summary.num_frames} frames, detected hand in {summary.num_detected} frames")
    if isinstance(source, RecordingHandTrackingSource):
        save_hand_recording_artifact(
            args.record_output,
            source.recorded_frames,
            source_fps=source.fps,
            source_desc=summary.source_desc,
            input_type=summary.input_type,
            num_frames=summary.num_frames,
            handedness=getattr(args, "hand", None),
            num_detected=summary.num_detected,
        )


def _wrap_source_for_recording(source, *, record_output_path: str | None):
    if not record_output_path:
        return source
    return RecordingHandTrackingSource(source)


def _wrap_source_for_interactive_recording(source, *, record_output_path: str | None):
    if not record_output_path:
        return source, None
    recording_source = RecordingHandTrackingSource(source, recording_enabled=False)
    return recording_source, TerminalRecordingController(recording_source)


def _run_webcam(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        MediaPipeInputSource(
            args.camera,
            target_hand=args.hand,
            swap_handedness=args.swap_hands,
            source_desc=f"camera://{args.camera}",
        ),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="webcam")
    session = _build_session(engine, visualize=True, show_preview=True)
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}",
        extra_lines=["Press 'q' in the camera window to quit."],
    )
    summary = session.run(source, input_type="webcam")
    _finalize_run(args, summary=summary, source=source)


def _run_video(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        MediaPipeInputSource(
            args.video,
            target_hand=args.hand,
            swap_handedness=args.swap_hands,
            source_desc=args.video,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="video")
    session = _build_session(engine, visualize=True, show_preview=False)
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}",
    )
    summary = session.run(source, input_type="video")
    _finalize_run(args, summary=summary, source=source)


def _run_replay(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_recording_source(recording_path=args.recording),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="replay")
    session = _build_session(engine, visualize=True, show_preview=False)
    metadata = getattr(source, "recording_metadata", {})
    extra_lines = [
        f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
        f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
    ]
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="replay", realtime=True, loop=args.loop)
    _finalize_run(args, summary=summary, source=source)


def _run_pico(args: argparse.Namespace) -> None:
    source, recording_controller = _wrap_source_for_interactive_recording(
        create_pico_source(handedness=args.hand, timeout=args.pico_timeout),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="pico")
    session = _build_session(
        engine,
        visualize=True,
        show_preview=False,
        key_callback=None if recording_controller is None else recording_controller.handle_keypress,
    )
    extra_lines = ["Requires xrobotoolkit_sdk plus active PICO hand tracking / gesture mode."]
    if recording_controller is not None:
        extra_lines.append("Press 'r' in the terminal or robot-hand viewer to start recording.")
        extra_lines.append("Press 's' in the terminal or robot-hand viewer to stop recording, save, and exit.")
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    if recording_controller is not None:
        recording_controller.start()
    try:
        summary = session.run(
            source,
            input_type="pico",
            stop_condition=None if recording_controller is None else (lambda: recording_controller.stop_requested),
        )
    finally:
        if recording_controller is not None:
            recording_controller.close()
    _finalize_run(args, summary=summary, source=source)


def _run_hc_mocap_bvh(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_hc_mocap_bvh_source(
            bvh_path=args.bvh,
            handedness=args.hand,
            teleopit_root=args.teleopit_root,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="hc_mocap")
    session = _build_session(engine, visualize=True, show_preview=False)
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
    )
    summary = session.run(source, input_type="hc_mocap", realtime=args.realtime, loop=args.loop)
    _finalize_run(args, summary=summary, source=source)


def _run_hc_mocap_udp(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_hc_mocap_udp_source(
            reference_bvh=args.reference_bvh,
            handedness=args.hand,
            host=args.udp_host,
            port=args.udp_port,
            timeout=args.udp_timeout,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="hc_mocap")
    session = _build_session(engine, visualize=True, show_preview=False)
    stats = source.stats_snapshot()
    extra_lines: list[str] = []
    if stats:
        extra_lines.append(
            "UDP packet format:"
            f" expected_floats={stats.get('expected_float_count', 0)}"
            f" bind={args.udp_host or '0.0.0.0'}:{args.udp_port}"
        )
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="hc_mocap", stats_every=args.udp_stats_every)
    _finalize_run(args, summary=summary, source=source)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webcam":
        _run_webcam(args)
        return
    if args.command == "video":
        _run_video(args)
        return
    if args.command == "replay":
        _run_replay(args)
        return
    if args.command == "pico":
        _run_pico(args)
        return
    if args.command == "hc-mocap" and args.hc_command == "bvh":
        _run_hc_mocap_bvh(args)
        return
    if args.command == "hc-mocap" and args.hc_command == "udp":
        _run_hc_mocap_udp(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")
