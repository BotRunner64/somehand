"""Offline ROS 2 integration probe for ManusRos2InputSource."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from somehand.runtime.manus_source import ManusRos2InputSource


def _side_text(value: Any) -> str:
    """Normalize a string or enum-like hand side for diagnostics."""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-timeout", type=float, default=1.0)
    parser.add_argument("--overall-timeout", type=float, default=15.0)
    args = parser.parse_args()

    source = ManusRos2InputSource(
        topic=args.topic,
        hand_side=args.side,
        timeout=args.frame_timeout,
        nominal_fps=30,
    )

    landmarks_history: list[np.ndarray] = []
    observed_sides: list[str] = []
    deadline = time.monotonic() + args.overall_timeout

    try:
        while (
            len(landmarks_history) < args.frames
            and time.monotonic() < deadline
        ):
            source_frame = source.get_frame()
            detection = source_frame.detection

            if detection is None:
                continue

            landmarks = np.asarray(
                detection.landmarks_3d,
                dtype=np.float64,
            )

            if landmarks.shape != (21, 3):
                raise RuntimeError(
                    f"Expected landmarks shape (21, 3), "
                    f"got {landmarks.shape}"
                )

            if not np.all(np.isfinite(landmarks)):
                raise RuntimeError(
                    "Received landmarks containing NaN or infinity"
                )

            actual_side = _side_text(detection.hand_side)

            if actual_side != args.side:
                raise RuntimeError(
                    f"Expected side={args.side}, got {actual_side}"
                )

            landmarks_history.append(landmarks.copy())
            observed_sides.append(actual_side)

        stats = source.stats_snapshot()

    finally:
        source.close()

    if len(landmarks_history) != args.frames:
        raise RuntimeError(
            f"Expected {args.frames} converted frames, "
            f"got {len(landmarks_history)}"
        )

    stacked = np.stack(landmarks_history, axis=0)

    motion_span = float(
        np.max(
            np.max(stacked, axis=0)
            - np.min(stacked, axis=0)
        )
    )

    if motion_span <= 1e-8:
        raise RuntimeError(
            f"Synthetic sequence did not move: "
            f"motion_span={motion_span}"
        )

    converted = int(stats["frames_converted"])
    received = int(stats["messages_received"])

    if converted < args.frames:
        raise RuntimeError(
            f"frames_converted={converted}, expected at least {args.frames}"
        )

    if received < converted:
        raise RuntimeError(
            f"messages_received={received} is less than "
            f"frames_converted={converted}"
        )

    result = {
        "status": "PASS",
        "topic": args.topic,
        "expected_side": args.side,
        "observed_sides": sorted(set(observed_sides)),
        "requested_frames": args.frames,
        "received_frames": len(landmarks_history),
        "landmark_shape": list(stacked.shape[1:]),
        "motion_span": motion_span,
        "first_wrist": stacked[0, 0].tolist(),
        "first_thumb_tip": stacked[0, 4].tolist(),
        "first_index_mcp": stacked[0, 5].tolist(),
        "stats": stats,
    }

    print(
        "PHASE1E_MANUS_SOURCE_PROBE="
        + json.dumps(result, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
