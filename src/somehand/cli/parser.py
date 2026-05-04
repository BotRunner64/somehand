"""Argument parsing for the somehand CLI."""

from __future__ import annotations

import argparse

from somehand.domain import normalize_hand_side
from somehand.paths import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_HC_MOCAP_REFERENCE_BVH


class _SomehandArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed_args, extras = super().parse_known_args(args, namespace)
        normalize_both_hand_args(parsed_args)
        return parsed_args, extras


def parse_hand_selector(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "both":
        return "both"
    return normalize_hand_side(value)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to retargeting config YAML",
    )
    parser.add_argument(
        "-H",
        "--hand",
        type=parse_hand_selector,
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


def add_live_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--signal-fps",
        type=int,
        default=None,
        help="Fixed output sampling rate for live mocap input; defaults to the source nominal fps",
    )


def add_dump_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to retargeting config YAML",
    )
    parser.add_argument(
        "-H",
        "--hand",
        type=parse_hand_selector,
        choices=["left", "right", "both"],
        default="right",
        help="Hand side for the current channel, or 'both' for two-hand mode",
    )
    parser.add_argument("--recording", required=True, help="Path to a saved hand-tracking recording")
    parser.add_argument("--output", required=True, help="Output MP4 path for the rendered replay video")


def normalize_both_hand_args(args: argparse.Namespace) -> None:
    if getattr(args, "hand", None) != "both":
        return
    if getattr(args, "config", None) == str(DEFAULT_CONFIG_PATH):
        args.config = str(DEFAULT_BIHAND_CONFIG_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = _SomehandArgumentParser(prog="somehand", description="Unified dex hand retargeting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    webcam = subparsers.add_parser("webcam", help="Retarget from a live webcam stream")
    add_common_args(webcam)
    webcam.add_argument("--camera", type=int, default=0, help="Webcam device index")
    webcam.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if capture reports the opposite hand",
    )

    video = subparsers.add_parser("video", help="Retarget from a video file")
    add_common_args(video)
    video.add_argument("--video", required=True, help="Path to input video file")
    video.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if this video reports the opposite hand",
    )

    replay = subparsers.add_parser("replay", help="Replay a saved hand-tracking recording")
    add_common_args(replay)
    replay.add_argument("--recording", required=True, help="Path to a saved hand-tracking recording")
    replay.add_argument("--loop", action="store_true", help="Loop the saved recording indefinitely")

    dump_video = subparsers.add_parser("dump-video", help="Render a replay recording to MP4 as fast as possible")
    add_dump_video_args(dump_video)

    pico = subparsers.add_parser("pico", help="Retarget from live PICO hand tracking via PICO Bridge")
    add_common_args(pico)
    add_live_sampling_args(pico)
    pico.add_argument("--pico-host", default="0.0.0.0", help="PICO Bridge receiver bind host")
    pico.add_argument("--pico-port", type=int, default=63901, help="PICO Bridge receiver TCP port")
    pico.add_argument(
        "--pico-advertise-ip",
        default=None,
        help="Optional PC IPv4 address advertised to the headset",
    )
    pico.add_argument(
        "--no-pico-discovery",
        action="store_true",
        help="Disable PICO Bridge UDP discovery broadcasts",
    )
    pico.add_argument(
        "--pico-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds while waiting for PICO Bridge hand-tracking frames",
    )

    hc_mocap = subparsers.add_parser("hc-mocap", help="Retarget from a live hc_mocap UDP stream")
    add_common_args(hc_mocap)
    add_live_sampling_args(hc_mocap)
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


__all__ = [
    "_SomehandArgumentParser",
    "add_common_args",
    "add_dump_video_args",
    "add_live_sampling_args",
    "build_parser",
    "normalize_both_hand_args",
    "parse_hand_selector",
]
