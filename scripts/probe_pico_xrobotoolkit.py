#!/usr/bin/env python3
"""Probe XRoboToolkit / PICO hand tracking status with clear terminal output."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.domain import display_hand_side, normalize_hand_side


INPUT_MODE_LABELS = {
    0: "head mode",
    1: "controller mode",
    2: "gesture mode",
}


def _ansi(code: str) -> str:
    return code if sys.stdout.isatty() else ""


RESET = _ansi("\033[0m")
BOLD = _ansi("\033[1m")
GREEN = _ansi("\033[32m")
YELLOW = _ansi("\033[33m")
RED = _ansi("\033[31m")
CYAN = _ansi("\033[36m")


def _status_prefix(level: str) -> str:
    if level == "OK":
        color = GREEN
    elif level == "WARN":
        color = YELLOW
    else:
        color = RED
    return f"{color}[{level:^4}]{RESET}"


def _print_header(title: str) -> None:
    bar = "━" * max(12, len(title) + 2)
    print(f"{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}")


def _print_status(level: str, label: str, detail: str) -> None:
    print(f"{_status_prefix(level)} {label}: {detail}")


def _format_vec3(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "n/a"
    return f"({values[0]:+.3f}, {values[1]:+.3f}, {values[2]:+.3f})"


def _tail_text(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="ignore")


def _latest_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return int(matches[-1])


def _count_int(pattern: str, text: str, expected: int) -> int:
    return sum(1 for match in re.findall(pattern, text) if int(match) == expected)


@dataclass
class LogSummary:
    path: Path
    exists: bool
    latest_input_mode: int | None
    latest_hand_mode: int | None
    left_active_count: int
    right_active_count: int


def _summarize_player_log(path: Path) -> LogSummary:
    text = _tail_text(path)
    return LogSummary(
        path=path,
        exists=path.exists(),
        latest_input_mode=_latest_int(r'Input\\?":\s*(\d)', text),
        latest_hand_mode=_latest_int(r'handMode\\?":\s*(\d)', text),
        left_active_count=_count_int(r'leftHand\\?":\{\\?"isActive\\?":(\d)', text, 1),
        right_active_count=_count_int(r'rightHand\\?":\{\\?"isActive\\?":(\d)', text, 1),
    )


def _service_running() -> bool:
    result = subprocess.run(
        ["ps", "-ef"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "RoboticsServiceProcess" in result.stdout


@dataclass
class HandStats:
    samples: int = 0
    active_samples: int = 0
    moved_samples: int = 0
    last_wrist: tuple[float, float, float] | None = None
    last_active_wrist: tuple[float, float, float] | None = None


@dataclass
class ProbeStats:
    left: HandStats
    right: HandStats
    sdk_timestamp_nonzero: bool


def _wrist_from_state(state: object) -> tuple[float, float, float]:
    wrist = state[0][:3]
    return (float(wrist[0]), float(wrist[1]), float(wrist[2]))


def _is_nonzero_wrist(wrist: tuple[float, float, float], threshold: float = 1e-6) -> bool:
    return any(abs(value) > threshold for value in wrist)


def _poll_sdk(duration: float, interval: float) -> ProbeStats:
    import xrobotoolkit_sdk as xrt

    left = HandStats()
    right = HandStats()
    sdk_timestamp_nonzero = False

    xrt.init()
    try:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            left_active = int(xrt.get_left_hand_is_active())
            right_active = int(xrt.get_right_hand_is_active())
            left_state = xrt.get_left_hand_tracking_state()
            right_state = xrt.get_right_hand_tracking_state()
            timestamp_ns = int(xrt.get_time_stamp_ns())

            left_wrist = _wrist_from_state(left_state)
            right_wrist = _wrist_from_state(right_state)

            left.samples += 1
            right.samples += 1
            left.last_wrist = left_wrist
            right.last_wrist = right_wrist

            if left_active == 1:
                left.active_samples += 1
                left.last_active_wrist = left_wrist
            if right_active == 1:
                right.active_samples += 1
                right.last_active_wrist = right_wrist

            if _is_nonzero_wrist(left_wrist):
                left.moved_samples += 1
            if _is_nonzero_wrist(right_wrist):
                right.moved_samples += 1

            if timestamp_ns != 0:
                sdk_timestamp_nonzero = True

            time.sleep(interval)
    finally:
        xrt.close()

    return ProbeStats(
        left=left,
        right=right,
        sdk_timestamp_nonzero=sdk_timestamp_nonzero,
    )


def _summarize_hand(name: str, stats: HandStats) -> tuple[str, str]:
    if stats.samples == 0:
        return "FAIL", "no samples collected"
    if stats.active_samples > 0:
        return (
            "OK",
            f"active {stats.active_samples}/{stats.samples}, last active wrist={_format_vec3(stats.last_active_wrist)}",
        )
    if stats.moved_samples > 0:
        return (
            "WARN",
            f"no active frames, but wrist changed in {stats.moved_samples}/{stats.samples} samples, last wrist={_format_vec3(stats.last_wrist)}",
        )
    return (
        "WARN",
        f"always inactive, wrist stayed near zero, last wrist={_format_vec3(stats.last_wrist)}",
    )


def _next_steps(log_summary: LogSummary, probe_stats: ProbeStats, target_hand: str) -> list[str]:
    steps: list[str] = []
    if log_summary.latest_input_mode == 1:
        steps.append("头显当前更像在 controller mode；切到手势/Hand Tracking 模式。")
    if log_summary.latest_input_mode in (None, 0):
        steps.append("确认 XRoboToolkit / RobotLinuxDemo 在头显前台，并已连接 PC service。")
    if log_summary.latest_hand_mode == 1:
        steps.append("`handMode=1` 表示仍偏向手柄输入；先放下或关闭手柄再试。")
    if probe_stats.left.active_samples == 0 and probe_stats.right.active_samples == 0:
        steps.append("确认 PICO 系统权限里的 Hand Tracking 已打开。")
    if target_hand == "right" and probe_stats.right.active_samples == 0:
        steps.append("右手始终 inactive；先在头显前做明显张合动作，保持右手进入相机视野。")
    if target_hand == "left" and probe_stats.left.active_samples == 0:
        steps.append("左手始终 inactive；先在头显前做明显张合动作，保持左手进入相机视野。")
    if not steps:
        steps.append("探测正常；现在可以重新运行 `dex-retarget pico --hand right --visualize`。")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PICO / XRoboToolkit hand-tracking status")
    parser.add_argument("--duration", type=float, default=6.0, help="Polling duration in seconds")
    parser.add_argument("--interval", type=float, default=0.05, help="Polling interval in seconds")
    parser.add_argument(
        "--hand",
        type=lambda value: "both" if value.strip().lower() == "both" else normalize_hand_side(value),
        choices=["left", "right", "both"],
        default="both",
        help="Which hand to focus on in the final suggestions",
    )
    parser.add_argument(
        "--log-path",
        default=str(Path.home() / ".config/unity3d/DefaultCompany/PXREAClientUnity/Player.log"),
        help="Path to XRoboToolkit Unity Player.log",
    )
    args = parser.parse_args()

    _print_header("PICO / XRoboToolkit Probe")

    try:
        import xrobotoolkit_sdk  # noqa: F401
    except ImportError as exc:
        _print_status("FAIL", "SDK import", f"xrobotoolkit_sdk unavailable: {exc}")
        raise SystemExit(2) from exc

    _print_status("OK", "SDK import", "xrobotoolkit_sdk is available")
    service_running = _service_running()
    _print_status(
        "OK" if service_running else "WARN",
        "PC service",
        (
            "RoboticsServiceProcess is running"
            if service_running
            else "RoboticsServiceProcess not found in `ps -ef` (service may still be launched by another wrapper)"
        ),
    )

    log_summary = _summarize_player_log(Path(args.log_path))
    if not log_summary.exists:
        _print_status("WARN", "Player.log", f"not found at {log_summary.path}")
    else:
        input_mode = (
            f"{log_summary.latest_input_mode} ({INPUT_MODE_LABELS.get(log_summary.latest_input_mode, 'unknown')})"
            if log_summary.latest_input_mode is not None
            else "unknown"
        )
        hand_mode = str(log_summary.latest_hand_mode) if log_summary.latest_hand_mode is not None else "unknown"
        _print_status(
            "OK" if log_summary.latest_input_mode == 2 else "WARN",
            "Player.log",
            f"latest Input={input_mode}, latest handMode={hand_mode}, "
            f"log active counts left={log_summary.left_active_count} right={log_summary.right_active_count}",
        )

    try:
        probe_stats = _poll_sdk(duration=args.duration, interval=args.interval)
    except Exception as exc:
        _print_status("FAIL", "SDK probe", f"{type(exc).__name__}: {exc}")
        raise SystemExit(2) from exc

    _print_status(
        "OK" if probe_stats.sdk_timestamp_nonzero else "WARN",
        "SDK stream",
        "timestamp became non-zero during probe" if probe_stats.sdk_timestamp_nonzero else "timestamp stayed zero during probe",
    )

    left_level, left_detail = _summarize_hand(display_hand_side("left"), probe_stats.left)
    right_level, right_detail = _summarize_hand(display_hand_side("right"), probe_stats.right)
    _print_status(left_level, f"{display_hand_side('left')} hand", left_detail)
    _print_status(right_level, f"{display_hand_side('right')} hand", right_detail)

    print()
    print(f"{BOLD}Next steps{RESET}")
    for index, step in enumerate(_next_steps(log_summary, probe_stats, args.hand), start=1):
        print(f"  {index}. {step}")


if __name__ == "__main__":
    main()
