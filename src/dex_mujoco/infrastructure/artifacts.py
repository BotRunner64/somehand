"""Serialization of runtime artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from dex_mujoco.domain import BiHandFrame, HandFrame, normalize_hand_side


_HAND_RECORDING_FORMAT = "dex_mujoco.hand_recording.v1"
_BIHAND_RECORDING_FORMAT = "dex_mujoco.bihand_recording.v1"


def _serialize_hand_frame(frame: HandFrame) -> dict[str, object]:
    return {
        "landmarks_3d": np.array(frame.landmarks_3d, copy=True),
        "landmarks_2d": None if frame.landmarks_2d is None else np.array(frame.landmarks_2d, copy=True),
        "hand_side": frame.hand_side,
    }


def _deserialize_hand_frame(payload: dict[str, object]) -> HandFrame:
    serialized_hand_side = payload.get("hand_side")
    if serialized_hand_side is None:
        serialized_hand_side = payload["handedness"]
    return HandFrame(
        landmarks_3d=np.array(payload["landmarks_3d"], copy=True),
        landmarks_2d=None if payload["landmarks_2d"] is None else np.array(payload["landmarks_2d"], copy=True),
        hand_side=normalize_hand_side(str(serialized_hand_side)),
    )


def _serialize_bihand_frame(frame: BiHandFrame) -> dict[str, object]:
    return {
        "left": None if frame.left is None else _serialize_hand_frame(frame.left),
        "right": None if frame.right is None else _serialize_hand_frame(frame.right),
    }


def _deserialize_bihand_frame(payload: dict[str, object]) -> BiHandFrame:
    return BiHandFrame(
        left=None if payload.get("left") is None else _deserialize_hand_frame(payload["left"]),
        right=None if payload.get("right") is None else _deserialize_hand_frame(payload["right"]),
    )


def save_trajectory_artifact(
    output_path: str | None,
    trajectory: list[np.ndarray],
    *,
    joint_names: list[str],
    config_path: str,
    num_frames: int,
    source_desc: str,
    input_type: str,
    hand_side: str | None = None,
    num_detected: int | None = None,
) -> None:
    if not output_path or not trajectory:
        return

    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trajectory": np.array(trajectory),
        "joint_names": joint_names,
        "config_path": config_path,
        "num_frames": num_frames,
        "input_source": source_desc,
        "input_type": input_type,
    }
    if hand_side is not None:
        payload["hand_side"] = normalize_hand_side(hand_side)
    if num_detected is not None:
        payload["num_detected"] = num_detected

    with artifact_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    print(f"Saved trajectory ({len(trajectory)} frames) to {artifact_path}")


def save_hand_recording_artifact(
    output_path: str | None,
    frames: list[HandFrame],
    *,
    source_fps: int,
    source_desc: str,
    input_type: str,
    num_frames: int,
    hand_side: str | None = None,
    num_detected: int | None = None,
) -> None:
    if not output_path or not frames:
        return

    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _HAND_RECORDING_FORMAT,
        "frames": [_serialize_hand_frame(frame) for frame in frames],
        "fps": source_fps,
        "num_frames": num_frames,
        "num_detected": len(frames) if num_detected is None else num_detected,
        "input_source": source_desc,
        "input_type": input_type,
    }
    if hand_side is not None:
        payload["hand_side"] = normalize_hand_side(hand_side)

    with artifact_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    print(f"Saved hand recording ({len(frames)} frames) to {artifact_path}")


def load_hand_recording_artifact(recording_path: str) -> dict[str, object]:
    artifact_path = Path(recording_path)
    with artifact_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    format_name = payload.get("format")
    if format_name != _HAND_RECORDING_FORMAT:
        raise ValueError(f"Unsupported hand recording format: {format_name!r}")

    return {
        "frames": [_deserialize_hand_frame(frame_payload) for frame_payload in payload["frames"]],
        "fps": int(payload.get("fps", 30)),
        "num_frames": int(payload.get("num_frames", len(payload["frames"]))),
        "num_detected": int(payload.get("num_detected", len(payload["frames"]))),
        "input_source": str(payload.get("input_source", artifact_path.as_posix())),
        "input_type": str(payload.get("input_type", "recording")),
        "hand_side": payload.get("hand_side", payload.get("handedness")),
    }


def save_bihand_recording_artifact(
    output_path: str | None,
    frames: list[BiHandFrame],
    *,
    source_fps: int,
    source_desc: str,
    input_type: str,
    num_frames: int,
    num_detected: int | None = None,
) -> None:
    if not output_path or not frames:
        return

    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _BIHAND_RECORDING_FORMAT,
        "frames": [_serialize_bihand_frame(frame) for frame in frames],
        "fps": source_fps,
        "num_frames": num_frames,
        "num_detected": len(frames) if num_detected is None else num_detected,
        "input_source": source_desc,
        "input_type": input_type,
    }

    with artifact_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    print(f"Saved bi-hand recording ({len(frames)} frames) to {artifact_path}")


def load_bihand_recording_artifact(recording_path: str) -> dict[str, object]:
    artifact_path = Path(recording_path)
    with artifact_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    format_name = payload.get("format")
    if format_name != _BIHAND_RECORDING_FORMAT:
        raise ValueError(f"Unsupported bi-hand recording format: {format_name!r}")

    return {
        "frames": [_deserialize_bihand_frame(frame_payload) for frame_payload in payload["frames"]],
        "fps": int(payload.get("fps", 30)),
        "num_frames": int(payload.get("num_frames", len(payload["frames"]))),
        "num_detected": int(payload.get("num_detected", len(payload["frames"]))),
        "input_source": str(payload.get("input_source", artifact_path.as_posix())),
        "input_type": str(payload.get("input_type", "recording")),
    }
