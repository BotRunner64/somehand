"""CLI command handlers."""

from __future__ import annotations

import argparse

from somehand.domain import display_hand_side
from somehand.runtime import (
    BiHandMediaPipeInputSource,
    MediaPipeInputSource,
    RecordingBiHandTrackingSource,
    RecordingHandTrackingSource,
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
from somehand.runtime.source_sampling import FixedRateBiHandTrackingSource, FixedRateHandTrackingSource

from .runtime import (
    build_bihand_engine as _build_bihand_engine,
    build_bihand_session as _build_bihand_session,
    build_engine as _build_engine,
    build_runtime_session as _build_runtime_session,
    build_session as _build_session,
)


def _print_startup(engine, *, source_desc: str, tracking_desc: str, extra_lines: list[str] | None = None) -> None:
    details = engine.describe()
    print(f"Model: {details['model_name']} ({details['dof']} DOF)")
    print(f"Retargeting: {details['vector_pairs']} vector pairs")
    print(f"Input source: {source_desc}")
    print(tracking_desc)
    if extra_lines:
        for line in extra_lines:
            print(line)


def _print_bihand_startup(engine, *, source_desc: str, tracking_desc: str, extra_lines: list[str] | None = None) -> None:
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


def _finalize_run(args: argparse.Namespace, *, summary, source) -> None:
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


def _wrap_live_hand_source(source, *, args: argparse.Namespace):
    return FixedRateHandTrackingSource(source, sample_fps=getattr(args, "signal_fps", None))


def _wrap_live_bihand_source(source, *, args: argparse.Namespace):
    return FixedRateBiHandTrackingSource(source, sample_fps=getattr(args, "signal_fps", None))


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
    _print_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {display_hand_side(args.hand)} | Source fps: {source.fps}",
        extra_lines=[
            f"Backend: {args.backend}",
            f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
            f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
        ],
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
        _wrap_live_hand_source(
            create_pico_source(
                hand_side=args.hand,
                timeout=args.pico_timeout,
                host=args.pico_host,
                port=args.pico_port,
                discovery=not args.no_pico_discovery,
                advertise_ip=args.pico_advertise_ip,
            ),
            args=args,
        ),
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
    extra_lines = [
        f"Backend: {args.backend}",
        f"Signal sampling: {source.fps} fps",
        f"PICO Bridge receiver: {args.pico_host}:{args.pico_port}",
        "Requires the PICO Bridge PC receiver package and headset app.",
    ]
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
        _wrap_live_hand_source(
            create_hc_mocap_udp_source(
                reference_bvh=args.reference_bvh,
                hand_side=args.hand,
                host=args.udp_host,
                port=args.udp_port,
                timeout=args.udp_timeout,
            ),
            args=args,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_engine(args, input_type="hc_mocap")
    session = _build_runtime_session(engine, args, visualize=True, show_preview=False)
    stats = source.stats_snapshot()
    extra_lines: list[str] = [f"Backend: {args.backend}", f"Signal sampling: {source.fps} fps"]
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


def _finalize_bihand_run(args: argparse.Namespace, *, summary, source) -> None:
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
    session = _build_bihand_session(engine, visualize=True, show_preview=False)
    metadata = getattr(source, "recording_metadata", {})
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hands: Left+Right | Source fps: {source.fps}",
        extra_lines=[
            f"Recorded source: {metadata.get('input_source', args.recording)} | Recorded input type: {metadata.get('input_type', 'unknown')}",
            f"Recorded fps: {source.fps} | Recorded detections: {metadata.get('num_detected', 0)}",
        ],
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
        _wrap_live_bihand_source(
            create_bihand_pico_source(
                timeout=args.pico_timeout,
                host=args.pico_host,
                port=args.pico_port,
                discovery=not args.no_pico_discovery,
                advertise_ip=args.pico_advertise_ip,
            ),
            args=args,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="pico")
    session = _build_bihand_session(
        engine,
        visualize=True,
        show_preview=False,
        key_callback=None if recording_controller is None else recording_controller.handle_keypress,
    )
    extra_lines = [
        f"Signal sampling: {source.fps} fps",
        f"PICO Bridge receiver: {args.pico_host}:{args.pico_port}",
        "Requires the PICO Bridge PC receiver package and headset app.",
    ]
    if recording_controller is not None:
        extra_lines.append("Press 'r' in the terminal or bi-hand viewer to start recording.")
        extra_lines.append("Press 's' in the terminal or bi-hand viewer to stop recording, save, and exit.")
    _print_bihand_startup(
        engine,
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hands: Left+Right | Source fps: {source.fps}",
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
        _wrap_live_bihand_source(
            create_bihand_hc_mocap_udp_source(
                reference_bvh=args.reference_bvh,
                host=args.udp_host,
                port=args.udp_port,
                timeout=args.udp_timeout,
            ),
            args=args,
        ),
        record_output_path=args.record_output,
    )
    engine = _build_bihand_engine(args, input_type="hc_mocap")
    session = _build_bihand_session(engine, visualize=True, show_preview=False)
    stats = source.stats_snapshot()
    extra_lines: list[str] = [f"Signal sampling: {source.fps} fps"]
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


__all__ = [name for name in globals() if name.startswith("_") and name != "__all__"]
