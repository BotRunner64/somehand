"""Calibrated MANUS finger-curl mapping for Revo2.

The default retargeting engine remains unchanged. This module is enabled only
when the MANUS CLI receives an explicit calibration profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from somehand.domain import HandFrame, RetargetingStepResult
from somehand.domain.hand_side import normalize_hand_side
from somehand.infrastructure.config_loader import load_retargeting_config

from .engine import RetargetingEngine


FINGER_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGER_NAMES = tuple(FINGER_INDICES)
FOUR_FINGER_NAMES = ("index", "middle", "ring", "pinky")

DEFAULT_DEADBANDS = {
    "thumb": 0.22,
    "index": 0.08,
    "middle": 0.08,
    "ring": 0.08,
    "pinky": 0.14,
}


@dataclass(frozen=True, slots=True)
class FingerCalibration:
    open_deg: float
    comfortable_closed_deg: float
    metric: str = "current"

    @property
    def span_deg(self) -> float:
        return self.comfortable_closed_deg - self.open_deg


@dataclass(frozen=True, slots=True)
class ManusCalibrationProfile:
    version: int
    side: str
    fingers: Mapping[str, FingerCalibration]


def _finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_manus_calibration_profile(
    path: str | Path,
    *,
    expected_side: str | None = None,
) -> ManusCalibrationProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"MANUS calibration file not found: {profile_path}"
        )

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    if version != 1:
        raise ValueError(
            f"Unsupported MANUS calibration version: {version}"
        )

    side = normalize_hand_side(data.get("side", ""))
    if expected_side is not None:
        normalized_expected = normalize_hand_side(expected_side)
        if side != normalized_expected:
            raise ValueError(
                "MANUS calibration side "
                f"{side!r} does not match requested side "
                f"{normalized_expected!r}"
            )

    raw_fingers = data.get("fingers")
    if not isinstance(raw_fingers, dict):
        raise ValueError(
            "MANUS calibration must contain a 'fingers' object"
        )

    missing = [
        name for name in FINGER_NAMES
        if name not in raw_fingers
    ]
    extra = sorted(set(raw_fingers) - set(FINGER_NAMES))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "Invalid MANUS calibration fingers: "
            + "; ".join(details)
        )

    fingers: dict[str, FingerCalibration] = {}
    for name in FINGER_NAMES:
        item = raw_fingers[name]
        if not isinstance(item, dict):
            raise ValueError(
                f"Calibration for finger {name!r} must be an object"
            )
        metric = str(
            item.get("metric", "current")
        ).strip().lower()

        if metric not in {
            "current",
            "joint_sum",
        }:
            raise ValueError(
                f"Unsupported calibration metric for "
                f"{name!r}: {metric!r}"
            )

        calibration = FingerCalibration(
            open_deg=_finite_float(
                item["open_deg"],
                label=f"{name}.open_deg",
            ),
            comfortable_closed_deg=_finite_float(
                item["comfortable_closed_deg"],
                label=f"{name}.comfortable_closed_deg",
            ),
            metric=metric,
        )
        if calibration.span_deg <= 1e-6:
            raise ValueError(
                f"Calibration span for {name!r} must be positive"
            )
        fingers[name] = calibration

    return ManusCalibrationProfile(
        version=version,
        side=side,
        fingers=fingers,
    )


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def _vector_angle_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_unit = unit(first)
    second_unit = unit(second)

    cosine = float(
        np.clip(
            np.dot(first_unit, second_unit),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def finger_curl_degrees(
    landmarks: np.ndarray,
    indices: tuple[int, int, int, int],
) -> float:
    mcp, pip, dip, tip = indices

    proximal = landmarks[pip] - landmarks[mcp]
    distal = landmarks[tip] - landmarks[dip]

    return _vector_angle_degrees(
        proximal,
        distal,
    )


def finger_joint_sum_degrees(
    landmarks: np.ndarray,
    indices: tuple[int, int, int, int],
) -> float:
    """Sum MCP, PIP and DIP flexion angles."""

    mcp, pip, dip, tip = indices

    palm_segment = landmarks[mcp] - landmarks[0]
    proximal_segment = landmarks[pip] - landmarks[mcp]
    middle_segment = landmarks[dip] - landmarks[pip]
    distal_segment = landmarks[tip] - landmarks[dip]

    return float(
        _vector_angle_degrees(
            palm_segment,
            proximal_segment,
        )
        + _vector_angle_degrees(
            proximal_segment,
            middle_segment,
        )
        + _vector_angle_degrees(
            middle_segment,
            distal_segment,
        )
    )


def apply_deadband(value: float, deadband: float) -> float:
    normalized = float(np.clip(value, 0.0, 1.0))
    if not 0.0 <= deadband < 1.0:
        raise ValueError("deadband must be in [0, 1)")
    if normalized <= deadband:
        return 0.0
    return float((normalized - deadband) / (1.0 - deadband))


def map_to_safe_range(
    normalized: float,
    bound: tuple[float | None, float | None],
    safe_fraction: float,
) -> float:
    low, high = bound
    if low is None or high is None:
        raise RuntimeError(f"Missing valid joint bound: {bound}")
    if not 0.0 <= safe_fraction <= 1.0:
        raise ValueError("safe_fraction must be in [0, 1]")

    low_value = float(low)
    high_value = float(high)
    safe_high = low_value + safe_fraction * (high_value - low_value)
    clipped = float(np.clip(normalized, 0.0, 1.0))
    return low_value + clipped * (safe_high - low_value)


class Revo2ManusQposMapper:
    """Convert calibrated finger curls into conservative Revo2 qpos."""

    def __init__(
        self,
        *,
        hand_model,
        bounds,
        profile: ManusCalibrationProfile,
        side: str,
        four_finger_safe_range: float = 0.65,
        thumb_metacarpal_safe_range: float = 0.35,
        thumb_proximal_safe_range: float = 0.55,
        deadbands: Mapping[str, float] = DEFAULT_DEADBANDS,
        pinky_ring_dominance_compensation: float = 0.60,
        smooth_alpha: float = 0.22,
        max_step_rad: float = 0.035,
    ) -> None:
        self.side = normalize_hand_side(side)
        if profile.side != self.side:
            raise ValueError(
                "MANUS calibration side does not match hand model side"
            )

        self.hand_model = hand_model
        self.bounds = bounds
        self.profile = profile
        self.four_finger_safe_range = float(
            four_finger_safe_range
        )
        self.thumb_metacarpal_safe_range = float(
            thumb_metacarpal_safe_range
        )
        self.thumb_proximal_safe_range = float(
            thumb_proximal_safe_range
        )
        self.deadbands = {
            name: float(deadbands[name])
            for name in FINGER_NAMES
        }
        self.pinky_ring_dominance_compensation = float(
            pinky_ring_dominance_compensation
        )
        self.smooth_alpha = float(smooth_alpha)
        self.max_step_rad = float(max_step_rad)

        if not 0.0 < self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if self.max_step_rad <= 0.0:
            raise ValueError("max_step_rad must be > 0")

        joint_to_qpos = (
            hand_model.get_joint_name_to_qpos_index()
        )
        required_names = {
            "thumb_metacarpal":
                f"{self.side}_thumb_metacarpal_joint",
            "thumb_proximal":
                f"{self.side}_thumb_proximal_joint",
            "index":
                f"{self.side}_index_proximal_joint",
            "middle":
                f"{self.side}_middle_proximal_joint",
            "ring":
                f"{self.side}_ring_proximal_joint",
            "pinky":
                f"{self.side}_pinky_proximal_joint",
        }

        missing = [
            joint_name
            for joint_name in required_names.values()
            if joint_name not in joint_to_qpos
        ]
        if missing:
            raise ValueError(
                "Revo2 MANUS mapping is missing joints: "
                + ", ".join(missing)
            )

        self.qpos_indices = {
            key: int(joint_to_qpos[joint_name])
            for key, joint_name in required_names.items()
        }
        self.controlled_indices = [
            self.qpos_indices["thumb_metacarpal"],
            self.qpos_indices["thumb_proximal"],
            self.qpos_indices["index"],
            self.qpos_indices["middle"],
            self.qpos_indices["ring"],
            self.qpos_indices["pinky"],
        ]
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def map(
        self,
        landmarks: np.ndarray,
        base_qpos: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (21, 3):
            raise ValueError(
                f"Expected MANUS landmarks shape (21, 3), "
                f"got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "MANUS landmarks contain NaN or infinity"
            )

        raw_normalized: dict[str, float] = {}
        for finger_name in FINGER_NAMES:
            calibration = self.profile.fingers[
                finger_name
            ]

            if calibration.metric == "current":
                curl = finger_curl_degrees(
                    points,
                    FINGER_INDICES[finger_name],
                )
            elif calibration.metric == "joint_sum":
                curl = finger_joint_sum_degrees(
                    points,
                    FINGER_INDICES[finger_name],
                )
            else:
                raise RuntimeError(
                    "Unsupported MANUS finger metric: "
                    f"{calibration.metric!r}"
                )

            normalized = (
                (curl - calibration.open_deg)
                / calibration.span_deg
            )
            raw_normalized[finger_name] = float(
                np.clip(normalized, 0.0, 1.0)
            )

        corrected = {
            finger_name: apply_deadband(
                raw_normalized[finger_name],
                self.deadbands[finger_name],
            )
            for finger_name in FINGER_NAMES
        }

        pinky_excess = max(
            0.0,
            corrected["pinky"] - corrected["ring"],
        )
        corrected["ring"] = float(
            np.clip(
                corrected["ring"]
                - self.pinky_ring_dominance_compensation
                * pinky_excess,
                0.0,
                1.0,
            )
        )

        desired = np.asarray(
            base_qpos,
            dtype=np.float64,
        ).copy()

        thumb_metacarpal = self.qpos_indices[
            "thumb_metacarpal"
        ]
        thumb_proximal = self.qpos_indices[
            "thumb_proximal"
        ]

        desired[thumb_metacarpal] = map_to_safe_range(
            corrected["thumb"],
            self.bounds[thumb_metacarpal],
            self.thumb_metacarpal_safe_range,
        )
        desired[thumb_proximal] = map_to_safe_range(
            corrected["thumb"],
            self.bounds[thumb_proximal],
            self.thumb_proximal_safe_range,
        )

        for finger_name in FOUR_FINGER_NAMES:
            qpos_index = self.qpos_indices[finger_name]
            desired[qpos_index] = map_to_safe_range(
                corrected[finger_name],
                self.bounds[qpos_index],
                self.four_finger_safe_range,
            )

        previous = self._previous
        if previous is None:
            previous = desired.copy()

        filtered = previous.copy()
        for qpos_index in self.controlled_indices:
            smoothed_target = (
                previous[qpos_index]
                + self.smooth_alpha
                * (
                    desired[qpos_index]
                    - previous[qpos_index]
                )
            )
            delta = float(
                np.clip(
                    smoothed_target - previous[qpos_index],
                    -self.max_step_rad,
                    self.max_step_rad,
                )
            )
            filtered[qpos_index] = previous[qpos_index] + delta

        filtered = self.hand_model.apply_mimic_constraints(
            filtered
        )
        self._previous = filtered.copy()
        return filtered, raw_normalized, corrected


class CalibratedManusRetargetingEngine(RetargetingEngine):
    """Retargeting engine with explicit calibrated MANUS override."""

    def __init__(
        self,
        config,
        *,
        calibration_path: str | Path,
        input_type: str = "manus_ros2",
    ) -> None:
        super().__init__(config, input_type=input_type)
        if not config.hand.name.startswith("revo2_"):
            raise ValueError(
                "Calibrated MANUS mapping currently supports "
                "Revo2 configs only"
            )

        self.calibration_path = str(
            Path(calibration_path).expanduser().resolve()
        )
        self.calibration_profile = (
            load_manus_calibration_profile(
                self.calibration_path,
                expected_side=config.hand.side,
            )
        )
        self._calibrated_mapper = Revo2ManusQposMapper(
            hand_model=self.hand_model,
            bounds=self.retargeter._bounds,
            profile=self.calibration_profile,
            side=config.hand.side,
        )
        self._calibrated_frame_count = 0

    @classmethod
    def from_config_path(
        cls,
        config_path: str,
        *,
        calibration_path: str | Path,
        input_type: str = "manus_ros2",
    ) -> "CalibratedManusRetargetingEngine":
        return cls(
            load_retargeting_config(config_path),
            calibration_path=calibration_path,
            input_type=input_type,
        )

    def process(
        self,
        frame: HandFrame,
    ) -> RetargetingStepResult:
        base_result = super().process(frame)
        qpos, raw, corrected = self._calibrated_mapper.map(
            frame.landmarks_3d,
            base_result.qpos,
        )

        self._calibrated_frame_count += 1
        frame_count = self._calibrated_frame_count
        if frame_count <= 3 or frame_count % 30 == 0:
            raw_display = {
                name: round(value, 3)
                for name, value in raw.items()
            }
            corrected_display = {
                name: round(value, 3)
                for name, value in corrected.items()
            }
            print(
                "CALIBRATED_MANUS_TRACE "
                f"frame={frame_count} "
                f"raw={raw_display} "
                f"corrected={corrected_display} "
                f"qpos={np.round(qpos, 4).tolist()}",
                flush=True,
            )

        return RetargetingStepResult(
            qpos=qpos.copy(),
            target_directions=base_result.target_directions,
            processed_landmarks=base_result.processed_landmarks,
            hand_side=base_result.hand_side,
            target_qpos=base_result.target_qpos,
            backend=base_result.backend,
        )


__all__ = [
    "CalibratedManusRetargetingEngine",
    "FingerCalibration",
    "ManusCalibrationProfile",
    "Revo2ManusQposMapper",
    "apply_deadband",
    "finger_curl_degrees",
    "finger_joint_sum_degrees",
    "load_manus_calibration_profile",
    "map_to_safe_range",
]
