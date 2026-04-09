"""Thin CLI interface over the application-layer pipeline."""

from __future__ import annotations

import argparse
import warnings

from dex_mujoco.application import (
    BiHandRetargetingEngine,
    BiHandRetargetingSession,
    RetargetingEngine,
    RetargetingSession,
)
from dex_mujoco.domain import display_hand_side, normalize_hand_side
from dex_mujoco.infrastructure import (
    AsyncLandmarkOutputSink,
    BiHandMediaPipeInputSource,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    OpenCvPreviewWindow,
    RecordingBiHandTrackingSource,
    RecordingHandTrackingSource,
    RobotHandOutputSink,
    RobotHandVideoOutputSink,
    TerminalRecordingController,
    create_bihand_hc_mocap_udp_source,
    create_bihand_pico_source,
    create_bihand_recording_source,
    create_hc_mocap_udp_source,
    create_pico_source,
    create_recording_source,
    save_bihand_recording_artifact,
    save_hand_recording_artifact,
)
from dex_mujoco.infrastructure.sources import MediaPipeInputSource
from dex_mujoco.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


def _close_resource(resource: object) -> None:
    close_fn = getattr(resource, "close", None)
    if callable(close_fn):
        close_fn()


def _interactive_visualization_available() -> tuple[bool, str | None]:
    try:
        import glfw
    except Exception as exc:
        return False, str(exc)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            available = bool(glfw.init())
        except Exception as exc:
            return False, str(exc)

    if not available:
        return False, "GLFW is unavailable"

    glfw.terminate()
    return True, None


def _parse_hand_selector(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "both":
        return "both"
    return normalize_hand_side(value)


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
        type=_parse_hand_selector,
        choices=["left", "right", "both"],
        default="right",
        help="Hand side for the current channel, or 'both' for bi-hand mode",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        help="Output pickle file for recorded hand-tracking frames",
    )


def _add_bihand_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_BIHAND_CONFIG_PATH),
        help="Path to bi-hand retargeting config YAML",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        help="Output pickle file for recorded bi-hand tracking frames",
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
    replay.add_argument(
        "--dump-video",
        default=None,
        help="Optional MP4 path for dumping the robot-hand replay video while replaying",
    )

    pico = subparsers.add_parser("pico", help="Retarget from live PICO hand tracking via XRoboToolkit")
    _add_common_args(pico)
    pico.add_argument(
        "--pico-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds while waiting for PICO hand-tracking frames",
    )

    hc_mocap = subparsers.add_parser("hc-mocap", help="Retarget from a live hc_mocap UDP stream")
    _add_common_args(hc_mocap)
    hc_mocap.add_argument(
        "--reference-bvh",
        default=str(DEFAULT_HC_MOCAP_REFERENCE_BVH),
        help="Reference hc_mocap BVH file for UDP joint ordering",
    )
    hc_mocap.add_argument("--udp-host", default="", help="UDP bind host for hc_mocap input")
    hc_mocap.add_argument("--udp-port", type=int, default=1118, help="UDP port for hc_mocap input")
    hc_mocap.add_argument("--udp-timeout", type=float, default=30.0, help="UDP startup timeout in seconds")
    hc_mocap.add_argument(
        "--udp-stats-every",
        type=int,
        default=120,
        help="Print UDP receive statistics every N processed frames (0 disables)",
    )

    bihand = subparsers.add_parser("bihand", help="Retarget both hands simultaneously")
    bihand_subparsers = bihand.add_subparsers(dest="bihand_command", required=True)

    bihand_webcam = bihand_subparsers.add_parser("webcam", help="Retarget both hands from a live webcam stream")
    _add_bihand_common_args(bihand_webcam)
    bihand_webcam.add_argument("--camera", type=int, default=0, help="Webcam device index")
    bihand_webcam.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if capture reports the opposite hand",
    )

    bihand_video = bihand_subparsers.add_parser("video", help="Retarget both hands from a video file")
    _add_bihand_common_args(bihand_video)
    bihand_video.add_argument("--video", required=True, help="Path to input video file")
    bihand_video.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if this video reports the opposite hand",
    )

    bihand_replay = bihand_subparsers.add_parser("replay", help="Replay a saved bi-hand tracking recording")
    _add_bihand_common_args(bihand_replay)
    bihand_replay.add_argument("--recording", required=True, help="Path to a saved bi-hand tracking recording")
    bihand_replay.add_argument("--loop", action="store_true", help="Loop the saved recording indefinitely")
    bihand_replay.add_argument(
        "--dump-video",
        default=None,
        help="Optional MP4 path for dumping the bi-hand replay video while replaying",
    )

    bihand_pico = bihand_subparsers.add_parser("pico", help="Retarget both hands from live PICO hand tracking")
    _add_bihand_common_args(bihand_pico)
    bihand_pico.add_argument(
        "--pico-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds while waiting for PICO hand-tracking frames",
    )

    bihand_hc_mocap = bihand_subparsers.add_parser("hc-mocap", help="Retarget both hands from a live hc_mocap UDP stream")
    _add_bihand_common_args(bihand_hc_mocap)
    bihand_hc_mocap.add_argument(
        "--reference-bvh",
        default=str(DEFAULT_HC_MOCAP_REFERENCE_BVH),
        help="Reference hc_mocap BVH file for UDP joint ordering",
    )
    bihand_hc_mocap.add_argument("--udp-host", default="", help="UDP bind host for hc_mocap input")
    bihand_hc_mocap.add_argument("--udp-port", type=int, default=1118, help="UDP port for hc_mocap input")
    bihand_hc_mocap.add_argument("--udp-timeout", type=float, default=30.0, help="UDP startup timeout in seconds")
    bihand_hc_mocap.add_argument(
        "--udp-stats-every",
        type=int,
        default=120,
        help="Print UDP receive statistics every N processed frames (0 disables)",
    )

    return parser


def _build_engine(args: argparse.Namespace, *, input_type: str) -> RetargetingEngine:
    return RetargetingEngine.from_config_path(args.config, input_type=input_type)


def _build_bihand_engine(args: argparse.Namespace, *, input_type: str) -> BiHandRetargetingEngine:
    return BiHandRetargetingEngine.from_config_path(args.config, input_type=input_type)


def _use_default_bihand_config_if_needed(args: argparse.Namespace) -> None:
    if getattr(args, "config", None) == str(DEFAULT_CONFIG_PATH):
        args.config = str(DEFAULT_BIHAND_CONFIG_PATH)


def _build_session(
    engine: RetargetingEngine,
    *,
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
    allow_visualization_fallback: bool = False,
) -> RetargetingSession:
    sinks = []
    frame_sinks = []
    if visualize and allow_visualization_fallback and video_output_path is not None:
        visualize_available, reason = _interactive_visualization_available()
        if not visualize_available:
            visualize = False
            print(f"Warning: visualization disabled during replay video dump: {reason}")
    if visualize:
        try:
            frame_sinks.append(AsyncLandmarkOutputSink())
            sinks.append(RobotHandOutputSink(engine.hand_model, key_callback=key_callback))
        except BaseException as exc:
            for sink in reversed(frame_sinks):
                _close_resource(sink)
            for sink in reversed(sinks):
                _close_resource(sink)
            frame_sinks = []
            sinks = []
            if not allow_visualization_fallback or video_output_path is None:
                raise
            print(f"Warning: visualization disabled during replay video dump: {exc}")
    if video_output_path is not None:
        if video_output_fps is None:
            raise ValueError("video_output_fps is required when video_output_path is provided")
        sinks.append(
            RobotHandVideoOutputSink(
                engine.hand_model,
                output_path=video_output_path,
                fps=video_output_fps,
            )
        )
    preview_window = OpenCvPreviewWindow() if show_preview else None
    return RetargetingSession(engine, sinks=sinks, frame_sinks=frame_sinks, preview_window=preview_window)


def _build_bihand_session(
    engine: BiHandRetargetingEngine,
    *,
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
    allow_visualization_fallback: bool = False,
) -> BiHandRetargetingSession:
    sinks = []
    if visualize and allow_visualization_fallback and video_output_path is not None:
        visualize_available, reason = _interactive_visualization_available()
        if not visualize_available:
            visualize = False
            print(f"Warning: visualization disabled during replay video dump: {reason}")
    if visualize:
        try:
            sinks.append(
                BiHandOutputWindowSink(
                    engine.left_engine.hand_model,
                    engine.right_engine.hand_model,
                    key_callback=key_callback,
                    panel_width=engine.config.viewer.panel_width,
                    panel_height=engine.config.viewer.panel_height,
                    window_name=engine.config.viewer.window_name,
                    left_pos=engine.config.viewer.left_pos,
                    right_pos=engine.config.viewer.right_pos,
                    camera_lookat=engine.config.viewer.camera_lookat,
                    left_quat=engine.config.viewer.left_quat,
                    right_quat=engine.config.viewer.right_quat,
                )
            )
        except BaseException as exc:
            for sink in reversed(sinks):
                _close_resource(sink)
            sinks = []
            if not allow_visualization_fallback or video_output_path is None:
                raise
            print(f"Warning: visualization disabled during replay video dump: {exc}")
    if video_output_path is not None:
        if video_output_fps is None:
            raise ValueError("video_output_fps is required when video_output_path is provided")
        sinks.append(
            BiHandVideoOutputSink(
                engine.left_engine.hand_model,
                engine.right_engine.hand_model,
                output_path=video_output_path,
                fps=video_output_fps,
                panel_width=engine.config.viewer.panel_width,
                panel_height=engine.config.viewer.panel_height,
                left_pos=engine.config.viewer.left_pos,
                right_pos=engine.config.viewer.right_pos,
                camera_lookat=engine.config.viewer.camera_lookat,
                left_quat=engine.config.viewer.left_quat,
                right_quat=engine.config.viewer.right_quat,
            )
        )
    preview_window = OpenCvPreviewWindow("Bi-Hand Detection") if show_preview else None
    return BiHandRetargetingSession(engine, sinks=sinks, preview_window=preview_window)


def _print_startup(engine: RetargetingEngine, *, source_desc: str, tracking_desc: str, extra_lines: list[str] | None = None) -> None:
    details = engine.describe()
    print(f"Model: {details['model_name']} ({details['dof']} DOF)")
    print(f"Retargeting: {details['vector_pairs']} vector pairs")
    print(f"Input source: {source_desc}")
    print(tracking_desc)
    if extra_lines:
        for line in extra_lines:
            print(line)


def _print_bihand_startup(
    engine: BiHandRetargetingEngine,
    *,
    source_desc: str,
    tracking_desc: str,
    extra_lines: list[str] | None = None,
) -> None:
    details = engine.describe()
    print(
        "Models:"
        f" left={details['left_model_name']} ({details['left_dof']} DOF)"
        f" | right={details['right_model_name']} ({details['right_dof']} DOF)"
    )
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
            hand_side=getattr(args, "hand", None),
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


def _wrap_bihand_source_for_recording(source, *, record_output_path: str | None):
    if not record_output_path:
        return source
    return RecordingBiHandTrackingSource(source)


def _wrap_bihand_source_for_interactive_recording(source, *, record_output_path: str | None):
    if not record_output_path:
        return source, None
    recording_source = RecordingBiHandTrackingSource(source, recording_enabled=False)
    return recording_source, TerminalRecordingController(recording_source)


def _run_webcam(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        MediaPipeInputSource(
            args.camera,
            hand_side=args.hand,
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
        tracking_desc=f"Tracking operator hand: {display_hand_side(args.hand)} | Swap hands: {args.swap_hands}",
        extra_lines=["Press 'q' in the camera window to quit."],
    )
    summary = session.run(source, input_type="webcam")
    _finalize_run(args, summary=summary, source=source)


def _run_video(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        MediaPipeInputSource(
            args.video,
            hand_side=args.hand,
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
        tracking_desc=f"Tracking operator hand: {display_hand_side(args.hand)} | Swap hands: {args.swap_hands}",
    )
    summary = session.run(source, input_type="video")
    _finalize_run(args, summary=summary, source=source)


def _run_replay(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_recording_source(recording_path=args.recording),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="replay")
    session = _build_session(
        engine,
        visualize=True,
        show_preview=False,
        video_output_path=args.dump_video,
        video_output_fps=source.fps,
        allow_visualization_fallback=True,
    )
    metadata = getattr(source, "recording_metadata", {})
    extra_lines = [
        f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
        f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
    ]
    if args.dump_video:
        extra_lines.append(f"Replay video dump: {args.dump_video}")
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="replay", realtime=True, loop=args.loop)
    _finalize_run(args, summary=summary, source=source)


def _run_pico(args: argparse.Namespace) -> None:
    source, recording_controller = _wrap_source_for_interactive_recording(
        create_pico_source(hand_side=args.hand, timeout=args.pico_timeout),
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
        tracking_desc=f"Tracking hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
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


def _run_hc_mocap_udp(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_hc_mocap_udp_source(
            reference_bvh=args.reference_bvh,
            hand_side=args.hand,
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
        tracking_desc=f"Tracking hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="hc_mocap", stats_every=args.udp_stats_every)
    _finalize_run(args, summary=summary, source=source)


def _finalize_bihand_run(
    args: argparse.Namespace,
    *,
    summary,
    source,
) -> None:
    print(
        f"Processed {summary.num_frames} frames, detected any hand in {summary.num_detected} frames "
        f"(left={summary.num_detected_left}, right={summary.num_detected_right}, both={summary.num_detected_both})"
    )
    if isinstance(source, RecordingBiHandTrackingSource):
        save_bihand_recording_artifact(
            args.record_output,
            source.recorded_frames,
            source_fps=source.fps,
            source_desc=summary.source_desc,
            input_type=summary.input_type,
            num_frames=summary.num_frames,
            num_detected=summary.num_detected,
        )


def _run_bihand_webcam(args: argparse.Namespace) -> None:
    source = _wrap_bihand_source_for_recording(
        BiHandMediaPipeInputSource(
            args.camera,
            swap_handedness=args.swap_hands,
            source_desc=f"camera://{args.camera}",
        ),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="webcam")
    session = _build_bihand_session(engine, visualize=True, show_preview=True)
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hands: Left+Right | Swap hands: {args.swap_hands}",
        extra_lines=["Press 'q' in the camera window to quit."],
    )
    summary = session.run(source, input_type="webcam")
    _finalize_bihand_run(args, summary=summary, source=source)


def _run_bihand_video(args: argparse.Namespace) -> None:
    source = _wrap_bihand_source_for_recording(
        BiHandMediaPipeInputSource(
            args.video,
            swap_handedness=args.swap_hands,
            source_desc=args.video,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="video")
    session = _build_bihand_session(engine, visualize=True, show_preview=False)
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hands: Left+Right | Swap hands: {args.swap_hands}",
    )
    summary = session.run(source, input_type="video")
    _finalize_bihand_run(args, summary=summary, source=source)


def _run_bihand_replay(args: argparse.Namespace) -> None:
    source = _wrap_bihand_source_for_recording(
        create_bihand_recording_source(recording_path=args.recording),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="replay")
    session = _build_bihand_session(
        engine,
        visualize=True,
        show_preview=False,
        video_output_path=args.dump_video,
        video_output_fps=source.fps,
        allow_visualization_fallback=True,
    )
    metadata = getattr(source, "recording_metadata", {})
    extra_lines = [
        f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
        f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
    ]
    if args.dump_video:
        extra_lines.append(f"Replay video dump: {args.dump_video}")
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hands: Left+Right | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="replay", realtime=True, loop=args.loop)
    _finalize_bihand_run(args, summary=summary, source=source)


def _run_bihand_pico(args: argparse.Namespace) -> None:
    source, recording_controller = _wrap_bihand_source_for_interactive_recording(
        create_bihand_pico_source(timeout=args.pico_timeout),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="pico")
    session = _build_bihand_session(
        engine,
        visualize=True,
        show_preview=False,
        key_callback=None if recording_controller is None else recording_controller.handle_keypress,
    )
    extra_lines = ["Requires xrobotoolkit_sdk plus active PICO hand tracking / gesture mode."]
    if recording_controller is not None:
        extra_lines.append("Press 'r' in the terminal or bi-hand viewer to start recording.")
        extra_lines.append("Press 's' in the terminal or bi-hand viewer to stop recording, save, and exit.")
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc="Tracking hands: Left+Right | Source fps: 120",
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
    _finalize_bihand_run(args, summary=summary, source=source)


def _run_bihand_hc_mocap_udp(args: argparse.Namespace) -> None:
    source = _wrap_bihand_source_for_recording(
        create_bihand_hc_mocap_udp_source(
            reference_bvh=args.reference_bvh,
            host=args.udp_host,
            port=args.udp_port,
            timeout=args.udp_timeout,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="hc_mocap")
    session = _build_bihand_session(engine, visualize=True, show_preview=False)
    stats = source.stats_snapshot()
    extra_lines: list[str] = []
    if stats:
        extra_lines.append(
            "UDP packet format:"
            f" expected_floats={stats.get('expected_float_count', 0)}"
            f" bind={args.udp_host or '0.0.0.0'}:{args.udp_port}"
        )
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hands: Left+Right | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="hc_mocap", stats_every=args.udp_stats_every)
    _finalize_bihand_run(args, summary=summary, source=source)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webcam":
        if args.hand == "both":
            _use_default_bihand_config_if_needed(args)
            _run_bihand_webcam(args)
            return
        _run_webcam(args)
        return
    if args.command == "video":
        if args.hand == "both":
            _use_default_bihand_config_if_needed(args)
            _run_bihand_video(args)
            return
        _run_video(args)
        return
    if args.command == "replay":
        if args.hand == "both":
            _use_default_bihand_config_if_needed(args)
            _run_bihand_replay(args)
            return
        _run_replay(args)
        return
    if args.command == "pico":
        if args.hand == "both":
            _use_default_bihand_config_if_needed(args)
            _run_bihand_pico(args)
            return
        _run_pico(args)
        return
    if args.command == "hc-mocap":
        if args.hand == "both":
            _use_default_bihand_config_if_needed(args)
            _run_bihand_hc_mocap_udp(args)
            return
        _run_hc_mocap_udp(args)
        return
    if args.command == "bihand":
        if args.bihand_command == "webcam":
            _run_bihand_webcam(args)
            return
        if args.bihand_command == "video":
            _run_bihand_video(args)
            return
        if args.bihand_command == "replay":
            _run_bihand_replay(args)
            return
        if args.bihand_command == "pico":
            _run_bihand_pico(args)
            return
        if args.bihand_command == "hc-mocap":
            _run_bihand_hc_mocap_udp(args)
            return
        raise ValueError(f"Unsupported bi-hand command: {args.bihand_command}")

    raise ValueError(f"Unsupported command: {args.command}")
