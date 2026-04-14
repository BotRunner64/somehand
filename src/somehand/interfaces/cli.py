"""Thin CLI interface over the application-layer pipeline."""

from __future__ import annotations

import argparse

from somehand.application import (
    BiHandRetargetingEngine,
    BiHandRetargetingSession,
    ControlledRetargetingSession,
    RetargetingEngine,
    RetargetingSession,
)
from somehand.domain import display_hand_side, normalize_hand_side
from somehand.infrastructure import (
    AsyncBiHandLandmarkOutputSink,
    AsyncLandmarkOutputSink,
    BiHandMediaPipeInputSource,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    LinkerHandModelAdapter,
    LinkerHandSdkController,
    MediaPipeInputSource,
    MujocoSimController,
    OpenCvPreviewWindow,
    RecordingBiHandTrackingSource,
    RecordingHandTrackingSource,
    RobotHandOutputSink,
    RobotHandTargetOutputSink,
    RobotHandVideoOutputSink,
    TerminalRecordingController,
    create_bihand_hc_mocap_udp_source,
    create_bihand_pico_source,
    create_bihand_recording_source,
    create_hc_mocap_udp_source,
    create_pico_source,
    create_recording_source,
    infer_linkerhand_model_family,
    save_bihand_recording_artifact,
    save_hand_recording_artifact,
)
from somehand.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


class _SomehandArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed_args, extras = super().parse_known_args(args, namespace)
        _normalize_both_hand_args(parsed_args)
        return parsed_args, extras


def _close_resource(resource: object) -> None:
    close_fn = getattr(resource, "close", None)
    if callable(close_fn):
        close_fn()


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
        help="Hand side for the current channel, or 'both' for two-hand mode",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        help="Output pickle file for recorded hand-tracking frames",
    )
    parser.add_argument(
        "--backend",
        choices=["viewer", "sim", "real"],
        default="viewer",
        help="Execution backend for robot-hand output",
    )
    parser.add_argument("--control-rate", type=int, default=100, help="Controller update rate in Hz")
    parser.add_argument("--sim-rate", type=int, default=500, help="MuJoCo simulation rate in Hz")
    parser.add_argument("--transport", choices=["can", "modbus"], default="can", help="Real-hand transport mode")
    parser.add_argument("--can-interface", default="can0", help="CAN interface name for real-hand mode")
    parser.add_argument("--modbus-port", default="None", help="MODBUS serial port for real-hand mode")
    parser.add_argument("--sdk-root", default=None, help="Optional LinkerHand SDK root directory")
    parser.add_argument("--model-family", default=None, help="Optional LinkerHand SDK model family override")


def _add_dump_video_args(parser: argparse.ArgumentParser) -> None:
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
        help="Hand side for the current channel, or 'both' for two-hand mode",
    )
    parser.add_argument("--recording", required=True, help="Path to a saved hand-tracking recording")
    parser.add_argument("--output", required=True, help="Output MP4 path for the rendered replay video")


def build_parser() -> argparse.ArgumentParser:
    parser = _SomehandArgumentParser(prog="somehand", description="Unified dex hand retargeting CLI")
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

    dump_video = subparsers.add_parser("dump-video", help="Render a replay recording to MP4 as fast as possible")
    _add_dump_video_args(dump_video)

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
        help="Optional custom hc_mocap BVH override; default uses built-in joint ordering",
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

    return parser


def _build_engine(args: argparse.Namespace, *, input_type: str) -> RetargetingEngine:
    return RetargetingEngine.from_config_path(args.config, input_type=input_type)


def _build_bihand_engine(args: argparse.Namespace, *, input_type: str) -> BiHandRetargetingEngine:
    return BiHandRetargetingEngine.from_config_path(args.config, input_type=input_type)


def _normalize_both_hand_args(args: argparse.Namespace) -> None:
    if getattr(args, "hand", None) != "both":
        return
    if getattr(args, "config", None) == str(DEFAULT_CONFIG_PATH):
        args.config = str(DEFAULT_BIHAND_CONFIG_PATH)


def _build_session(
    engine: RetargetingEngine,
    *,
    backend: str = "viewer",
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
) -> RetargetingSession:
    sinks = []
    frame_sinks = []
    if visualize:
        try:
            frame_sinks.append(AsyncLandmarkOutputSink(window_title="Input Landmarks"))
            if backend == "sim":
                sinks.append(
                    RobotHandTargetOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Retargeting",
                    )
                )
                sinks.append(
                    RobotHandOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Sim State",
                    )
                )
            else:
                sinks.append(
                    RobotHandOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Retargeting",
                    )
                )
        except BaseException as exc:
            for sink in reversed(frame_sinks):
                _close_resource(sink)
            for sink in reversed(sinks):
                _close_resource(sink)
            raise
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


def _build_control_backend(args: argparse.Namespace, engine: RetargetingEngine):
    if args.backend == "sim":
        return MujocoSimController(
            engine.config.hand.mjcf_path,
            control_rate_hz=args.control_rate,
            sim_rate_hz=args.sim_rate,
        )
    if args.backend == "real":
        family = args.model_family or engine.config.controller.model_family or infer_linkerhand_model_family(
            engine.config.hand.name
        )
        adapter = LinkerHandModelAdapter(
            engine.hand_model,
            family=family,
            hand_side=engine.config.hand.side,
            sdk_root="" if args.sdk_root is None else args.sdk_root,
        )
        return LinkerHandSdkController(
            adapter,
            transport=args.transport,
            can_interface=args.can_interface,
            modbus_port=args.modbus_port,
            default_speed=engine.config.controller.default_speed or adapter.default_speed,
            default_torque=engine.config.controller.default_torque or adapter.default_torque,
            sdk_root="" if args.sdk_root is None else args.sdk_root,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def _build_runtime_session(
    engine: RetargetingEngine,
    args: argparse.Namespace,
    *,
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
    include_landmark_viewer: bool = True,
    include_sim_state_viewer: bool = True,
):
    if args.backend == "viewer":
        return _build_session(
            engine,
            backend=args.backend,
            visualize=visualize,
            show_preview=show_preview,
            key_callback=key_callback,
            video_output_path=video_output_path,
            video_output_fps=video_output_fps,
        )

    sinks = []
    frame_sinks = []
    if visualize:
        try:
            if include_landmark_viewer:
                frame_sinks.append(AsyncLandmarkOutputSink(window_title="Input Landmarks"))
            if args.backend == "sim":
                sinks.append(
                    RobotHandTargetOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Retargeting",
                    )
                )
                if include_sim_state_viewer:
                    sinks.append(
                        RobotHandOutputSink(
                            engine.hand_model,
                            key_callback=key_callback,
                            window_title="Sim State",
                        )
                    )
            elif args.backend == "real":
                sinks.append(
                    RobotHandTargetOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Retargeting",
                    )
                )
            else:
                sinks.append(
                    RobotHandOutputSink(
                        engine.hand_model,
                        key_callback=key_callback,
                        window_title="Retargeting",
                    )
                )
        except BaseException as exc:
            for sink in reversed(frame_sinks):
                _close_resource(sink)
            for sink in reversed(sinks):
                _close_resource(sink)
            raise
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
    controller = _build_control_backend(args, engine)
    return ControlledRetargetingSession(
        engine,
        controller,
        sinks=sinks,
        frame_sinks=frame_sinks,
        preview_window=preview_window,
    )


def _build_bihand_session(
    engine: BiHandRetargetingEngine,
    *,
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
) -> BiHandRetargetingSession:
    sinks = []
    frame_sinks = []
    if visualize:
        try:
            frame_sinks.append(
                AsyncBiHandLandmarkOutputSink(
                    left_pos=engine.config.viewer.left_pos,
                    right_pos=engine.config.viewer.right_pos,
                    left_quat=engine.config.viewer.left_quat,
                    right_quat=engine.config.viewer.right_quat,
                )
            )
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
            for sink in reversed(frame_sinks):
                _close_resource(sink)
            for sink in reversed(sinks):
                _close_resource(sink)
            raise
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
    return BiHandRetargetingSession(engine, sinks=sinks, frame_sinks=frame_sinks, preview_window=preview_window)


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
    session = _build_runtime_session(engine, args, visualize=True, show_preview=True)
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {display_hand_side(args.hand)} | Swap hands: {args.swap_hands}",
        extra_lines=[f"Backend: {args.backend}", "Press 'q' in the camera window to quit."],
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
    session = _build_runtime_session(engine, args, visualize=True, show_preview=False)
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {display_hand_side(args.hand)} | Swap hands: {args.swap_hands}",
        extra_lines=[f"Backend: {args.backend}"],
    )
    summary = session.run(source, input_type="video")
    _finalize_run(args, summary=summary, source=source)


def _run_replay(args: argparse.Namespace) -> None:
    source = _wrap_source_for_recording(
        create_recording_source(recording_path=args.recording),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="replay")
    session = _build_runtime_session(
        engine,
        args,
        visualize=True,
        show_preview=False,
        include_landmark_viewer=True,
        include_sim_state_viewer=True,
    )
    metadata = getattr(source, "recording_metadata", {})
    extra_lines = [
        f"Backend: {args.backend}",
        f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
        f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
    ]
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="replay", realtime=True, loop=args.loop)
    _finalize_run(args, summary=summary, source=source)


def _run_dump_video(args: argparse.Namespace) -> None:
    source = create_recording_source(recording_path=args.recording)
    engine = _build_engine(args, input_type="replay")
    session = _build_session(
        engine,
        visualize=False,
        show_preview=False,
        video_output_path=args.output,
        video_output_fps=source.fps,
    )
    metadata = getattr(source, "recording_metadata", {})
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Rendering replay video for hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
        extra_lines=[
            f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
            f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
            f"Video output: {args.output}",
            "Render mode: offline",
        ],
    )
    summary = session.run(source, input_type="replay", realtime=False)
    _finalize_run(args, summary=summary, source=source)


def _run_pico(args: argparse.Namespace) -> None:
    source, recording_controller = _wrap_source_for_interactive_recording(
        create_pico_source(hand_side=args.hand, timeout=args.pico_timeout),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="pico")
    session = _build_runtime_session(
        engine,
        args,
        visualize=True,
        show_preview=False,
        key_callback=None if recording_controller is None else recording_controller.handle_keypress,
    )
    extra_lines = [f"Backend: {args.backend}", "Requires xrobotoolkit_sdk plus active PICO hand tracking / gesture mode."]
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
    session = _build_runtime_session(engine, args, visualize=True, show_preview=False)
    stats = source.stats_snapshot()
    extra_lines: list[str] = [f"Backend: {args.backend}"]
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
    )
    metadata = getattr(source, "recording_metadata", {})
    extra_lines = [
        f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
        f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
    ]
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hands: Left+Right | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    summary = session.run(source, input_type="replay", realtime=True, loop=args.loop)
    _finalize_bihand_run(args, summary=summary, source=source)


def _run_bihand_dump_video(args: argparse.Namespace) -> None:
    source = create_bihand_recording_source(recording_path=args.recording)
    engine = _build_bihand_engine(args, input_type="replay")
    session = _build_bihand_session(
        engine,
        visualize=False,
        show_preview=False,
        video_output_path=args.output,
        video_output_fps=source.fps,
    )
    metadata = getattr(source, "recording_metadata", {})
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Rendering replay video for hands: Left+Right | Source fps: {source.fps}",
        extra_lines=[
            f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
            f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
            f"Video output: {args.output}",
            "Render mode: offline",
        ],
    )
    summary = session.run(source, input_type="replay", realtime=False)
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
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            _run_bihand_webcam(args)
            return
        _run_webcam(args)
        return
    if args.command == "video":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            _run_bihand_video(args)
            return
        _run_video(args)
        return
    if args.command == "replay":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            _run_bihand_replay(args)
            return
        _run_replay(args)
        return
    if args.command == "dump-video":
        if args.hand == "both":
            _run_bihand_dump_video(args)
            return
        _run_dump_video(args)
        return
    if args.command == "pico":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            _run_bihand_pico(args)
            return
        _run_pico(args)
        return
    if args.command == "hc-mocap":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            _run_bihand_hc_mocap_udp(args)
            return
        _run_hc_mocap_udp(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")
