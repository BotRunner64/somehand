"""CLI entrypoint and command dispatch."""

from __future__ import annotations

from .parser import build_parser


def _load_commands():
    try:
        from . import commands
    except ImportError as exc:
        raise RuntimeError(
            "somehand CLI optional dependencies are missing. "
            "Install them with `pip install 'somehand[cli]'`."
        ) from exc
    return commands


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = _load_commands()

    if args.command == "webcam":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_webcam(args)
            return
        commands._run_webcam(args)
        return
    if args.command == "video":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_video(args)
            return
        commands._run_video(args)
        return
    if args.command == "replay":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_replay(args)
            return
        commands._run_replay(args)
        return
    if args.command == "dump-video":
        if args.hand == "both":
            commands._run_bihand_dump_video(args)
            return
        commands._run_dump_video(args)
        return
    if args.command == "pico":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_pico(args)
            return
        commands._run_pico(args)
        return
    if args.command == "hc-mocap":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_hc_mocap_udp(args)
            return
        commands._run_hc_mocap_udp(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")
