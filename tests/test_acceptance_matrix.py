import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.acceptance import closure_metrics, synthetic_hand_pose
from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.vector_solver import VectorRetargeter


_DEEP_CONFIGS = [
    "configs/retargeting/right/linkerhand_l20_right.yaml",
    "configs/retargeting/right/linkerhand_o6_right.yaml",
    "configs/retargeting/right/dexhand021_right.yaml",
]
_ALL_RIGHT_CONFIGS = sorted(Path("configs/retargeting/right").glob("*_right.yaml"))


def _solve_pose(config_path: str, pose_name: str) -> dict[str, float]:
    config = load_retargeting_config(config_path)
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)
    retargeter.update_targets(synthetic_hand_pose(pose_name), hand_side="right")
    retargeter.solve()
    return closure_metrics(retargeter)


@pytest.mark.parametrize("config_path", _DEEP_CONFIGS)
def test_representative_hands_improve_pinch_and_solve_fist(config_path: str):
    open_metrics = _solve_pose(config_path, "open")
    pinch_metrics = _solve_pose(config_path, "pinch")
    _solve_pose(config_path, "fist")

    assert pinch_metrics["pinch_thumb_index_gap_scaled"] < open_metrics["pinch_thumb_index_gap_scaled"]


@pytest.mark.parametrize("config_path", [str(path) for path in _ALL_RIGHT_CONFIGS])
def test_all_right_configs_smoke_open_pinch_fist(config_path: str):
    open_metrics = _solve_pose(config_path, "open")
    pinch_metrics = _solve_pose(config_path, "pinch")
    fist_metrics = _solve_pose(config_path, "fist")

    for metrics in (open_metrics, pinch_metrics, fist_metrics):
        for value in metrics.values():
            assert np.isfinite(value)

    assert pinch_metrics["pinch_thumb_index_gap_scaled"] < open_metrics["pinch_thumb_index_gap_scaled"]
