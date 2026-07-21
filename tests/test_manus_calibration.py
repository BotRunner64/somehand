"""Tests for calibrated MANUS -> Revo2 mapping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from somehand.application.manus_calibration import (
    FINGER_NAMES,
    ManusCalibrationProfile,
    Revo2ManusQposMapper,
    apply_deadband,
    load_manus_calibration_profile,
)


def _profile(side: str = "right") -> ManusCalibrationProfile:
    from somehand.application.manus_calibration import (
        FingerCalibration,
    )

    return ManusCalibrationProfile(
        version=1,
        side=side,
        fingers={
            name: FingerCalibration(
                open_deg=0.0,
                comfortable_closed_deg=90.0,
            )
            for name in FINGER_NAMES
        },
    )


def _closed_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float64)
    bases = (1, 5, 9, 13, 17)
    for base in bases:
        landmarks[base] = [0.0, 0.0, 0.0]
        landmarks[base + 1] = [1.0, 0.0, 0.0]
        landmarks[base + 2] = [1.0, 0.0, 0.0]
        landmarks[base + 3] = [1.0, 1.0, 0.0]
    return landmarks


class _FakeHandModel:
    def get_joint_name_to_qpos_index(self):
        return {
            "right_thumb_metacarpal_joint": 0,
            "right_thumb_proximal_joint": 1,
            "right_index_proximal_joint": 3,
            "right_middle_proximal_joint": 5,
            "right_ring_proximal_joint": 7,
            "right_pinky_proximal_joint": 9,
        }

    def apply_mimic_constraints(self, qpos):
        result = np.asarray(qpos, dtype=np.float64).copy()
        result[2] = result[1]
        result[4] = result[3]
        result[6] = result[5]
        result[8] = result[7]
        result[10] = result[9]
        return result


def test_load_profile_validates_side(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "side": "Right",
                "fingers": {
                    name: {
                        "open_deg": 0.0,
                        "comfortable_closed_deg": 90.0,
                    }
                    for name in FINGER_NAMES
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_manus_calibration_profile(
        path,
        expected_side="right",
    )
    assert profile.side == "right"

    with pytest.raises(ValueError, match="does not match"):
        load_manus_calibration_profile(
            path,
            expected_side="left",
        )


def test_load_profile_rejects_non_positive_span(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "side": "right",
                "fingers": {
                    name: {
                        "open_deg": 10.0,
                        "comfortable_closed_deg": 10.0,
                    }
                    for name in FINGER_NAMES
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="span"):
        load_manus_calibration_profile(path)


def test_apply_deadband():
    assert apply_deadband(0.05, 0.08) == 0.0
    assert apply_deadband(1.0, 0.08) == pytest.approx(1.0)
    assert apply_deadband(0.54, 0.08) == pytest.approx(0.5)


def test_mapper_applies_safe_ranges_and_mimic_constraints():
    mapper = Revo2ManusQposMapper(
        hand_model=_FakeHandModel(),
        bounds=[(0.0, 1.0)] * 11,
        profile=_profile(),
        side="right",
    )

    qpos, raw, corrected = mapper.map(
        _closed_landmarks(),
        np.zeros(11, dtype=np.float64),
    )

    assert raw == pytest.approx(
        {name: 1.0 for name in FINGER_NAMES}
    )
    assert corrected == pytest.approx(
        {name: 1.0 for name in FINGER_NAMES}
    )

    assert qpos[0] == pytest.approx(0.35)
    assert qpos[1] == pytest.approx(0.55)
    assert qpos[2] == pytest.approx(0.55)

    for source, mimic in ((3, 4), (5, 6), (7, 8), (9, 10)):
        assert qpos[source] == pytest.approx(0.65)
        assert qpos[mimic] == pytest.approx(0.65)


def test_mapper_rejects_wrong_side():
    with pytest.raises(ValueError, match="side"):
        Revo2ManusQposMapper(
            hand_model=_FakeHandModel(),
            bounds=[(0.0, 1.0)] * 11,
            profile=_profile(side="left"),
            side="right",
        )
