"""Unified CLI for dex retargeting."""

from __future__ import annotations

import argparse
import time

import cv2

from .paths import DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


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
    parser.add_argument("-o", "--output", default=None, help="Output pickle file for joint trajectory")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show two MuJoCo viewers: input hand landmarks and retargeted robot hand",
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


def _runtime_from_args(args: argparse.Namespace, *, input_type: str):
    from .runtime import RetargetRuntime, RuntimeOptions

    options = RuntimeOptions(
        config_path=args.config,
        visualize=args.visualize,
        output_path=args.output,
    )
    return RetargetRuntime(options, input_type=input_type)


def _run_source(
    source,
    runtime,
    *,
    handedness: str,
    show_preview: bool = False,
    preview_window_name: str = "Hand Detection",
    realtime: bool = False,
    loop: bool = False,
    stats_every: int = 0,
) -> None:
    frame_count = 0
    detected_count = 0
    frame_period = 1.0 / max(source.fps, 1)

    try:
        while True:
            if not source.is_available():
                if loop and source.reset():
                    continue
                break

            tic = time.monotonic()
            try:
                frame = source.get_frame()
            except StopIteration:
                break

            frame_count += 1
            detection = frame.detection
            if detection is not None:
                detected_count += 1
                runtime.process_detection(detection)

            if show_preview and frame.preview_frame is not None:
                preview = frame.preview_frame
                if detection is not None and hasattr(source, "annotate_preview"):
                    preview = source.annotate_preview(preview, detection)
                cv2.imshow(preview_window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if stats_every > 0 and frame_count % stats_every == 0:
                stats = source.stats_snapshot()
                if stats:
                    print(
                        "UDP stats:"
                        f" recv={stats.get('packets_received', 0)}"
                        f" valid={stats.get('packets_valid', 0)}"
                        f" bad_size={stats.get('packets_bad_size', 0)}"
                        f" bad_decode={stats.get('packets_bad_decode', 0)}"
                        f" floats={stats.get('last_float_count', 0)}/{stats.get('expected_float_count', 0)}"
                        f" bytes={stats.get('last_packet_bytes', 0)}"
                        f" sender={stats.get('last_sender')}"
                    )

            if not runtime.is_running:
                break

            if realtime:
                elapsed = time.monotonic() - tic
                sleep_s = frame_period - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        source.close()
        runtime.close()
        if show_preview:
            cv2.destroyAllWindows()

    runtime.print_summary(num_frames=frame_count, num_detected=detected_count)
    runtime.save_output(
        source_desc=source.source_desc,
        num_frames=frame_count,
        num_detected=detected_count,
        handedness=handedness,
    )


def _run_webcam(args: argparse.Namespace) -> None:
    from .input_sources import MediaPipeInputSource

    runtime = _runtime_from_args(args, input_type="webcam")
    source = MediaPipeInputSource(
        args.camera,
        target_hand=args.hand,
        swap_handedness=args.swap_hands,
        source_desc=f"camera://{args.camera}",
    )
    runtime.print_startup(
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}",
        extra_lines=["Press 'q' in the camera window to quit."],
    )
    _run_source(
        source,
        runtime,
        handedness=args.hand,
        show_preview=True,
    )


def _run_video(args: argparse.Namespace) -> None:
    from .input_sources import MediaPipeInputSource

    runtime = _runtime_from_args(args, input_type="video")
    source = MediaPipeInputSource(
        args.video,
        target_hand=args.hand,
        swap_handedness=args.swap_hands,
        source_desc=args.video,
    )
    runtime.print_startup(
        source_desc=source.source_desc,
        tracking_desc=f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}",
    )
    _run_source(source, runtime, handedness=args.hand)


def _run_pico(args: argparse.Namespace) -> None:
    from .input_sources import HCMocapInputSource
    from .pico_input import create_pico_provider

    provider = create_pico_provider(
        handedness=args.hand,
        timeout=args.pico_timeout,
    )
    source = HCMocapInputSource(
        provider,
        source_desc=f"pico://{args.hand.lower()}",
    )
    runtime = _runtime_from_args(args, input_type="pico")
    runtime.print_startup(
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
        extra_lines=[
            "Requires xrobotoolkit_sdk plus active PICO hand tracking / gesture mode.",
        ],
    )
    _run_source(source, runtime, handedness=args.hand)


def _run_hc_mocap_bvh(args: argparse.Namespace) -> None:
    from .hc_mocap_input import create_hc_mocap_bvh_provider
    from .input_sources import HCMocapInputSource

    provider = create_hc_mocap_bvh_provider(
        bvh_path=args.bvh,
        handedness=args.hand,
        teleopit_root=args.teleopit_root,
    )
    source = HCMocapInputSource(provider, source_desc=args.bvh)
    runtime = _runtime_from_args(args, input_type="hc_mocap")
    runtime.print_startup(
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
    )
    _run_source(
        source,
        runtime,
        handedness=args.hand,
        realtime=args.realtime,
        loop=args.loop,
    )


def _run_hc_mocap_udp(args: argparse.Namespace) -> None:
    from .hc_mocap_input import create_hc_mocap_udp_provider
    from .input_sources import HCMocapInputSource

    provider = create_hc_mocap_udp_provider(
        reference_bvh=args.reference_bvh,
        handedness=args.hand,
        host=args.udp_host,
        port=args.udp_port,
        timeout=args.udp_timeout,
    )
    source = HCMocapInputSource(
        provider,
        source_desc=f"udp://{args.udp_host or '0.0.0.0'}:{args.udp_port}",
    )
    runtime = _runtime_from_args(args, input_type="hc_mocap")
    stats = source.stats_snapshot()
    extra_lines = []
    if stats:
        extra_lines.append(
            "UDP packet format:"
            f" expected_floats={stats.get('expected_float_count', 0)}"
            f" bind={args.udp_host or '0.0.0.0'}:{args.udp_port}"
        )
    runtime.print_startup(
        source_desc=source.source_desc,
        tracking_desc=f"Tracking hand: {args.hand} | Source fps: {source.fps}",
        extra_lines=extra_lines,
    )
    _run_source(
        source,
        runtime,
        handedness=args.hand,
        stats_every=args.udp_stats_every,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webcam":
        _run_webcam(args)
        return
    if args.command == "video":
        _run_video(args)
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


if __name__ == "__main__":
    main()
