"""Validate the frozen hand matrix before recording or running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.quantitative_geometry import RobotEvaluationGeometry
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest
from somehand.infrastructure.vector_solver import VectorRetargeter


def _load_common_defaults() -> dict:
    path = PROJECT_ROOT / "configs" / "retargeting" / "base" / "_universal_common.yaml"
    with path.open(encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)["retargeting"]


def _assert_shared_parameters(config, defaults: dict) -> None:
    if asdict(config.preprocess) != defaults["preprocess"]:
        raise ValueError(f"{config.hand.name}: preprocess settings differ from universal defaults")
    if asdict(config.solver) != defaults["solver"]:
        raise ValueError(f"{config.hand.name}: solver settings differ from universal defaults")

    vector_defaults = defaults["constraint_defaults"]["vector"]
    for constraint in config.vector_constraints:
        expected = (
            float(vector_defaults["terminal_weight"])
            if constraint.robot_types[1] == "site"
            else float(vector_defaults["weight"])
        )
        if constraint.weight != expected:
            raise ValueError(f"{config.hand.name}: vector weight differs for {constraint.human}")

    distance_defaults = defaults["constraint_defaults"]["distance"]
    weights = distance_defaults["weights_by_human"]
    for constraint in config.distance_constraints:
        pair_key = f"{constraint.human[0]},{constraint.human[1]}"
        expected = (
            float(weights[pair_key]),
            float(distance_defaults["scale"]),
            float(distance_defaults["threshold"]),
            str(distance_defaults["activation_type"]),
            str(distance_defaults["scale_mode"]),
        )
        actual = (
            constraint.weight,
            constraint.scale,
            constraint.threshold,
            constraint.activation_type,
            constraint.scale_mode,
        )
        if actual != expected:
            raise ValueError(f"{config.hand.name}: distance settings differ for {constraint.human}")

    frame_defaults = defaults["constraint_defaults"]["frame"]
    for constraint in config.frame_constraints:
        if (
            constraint.primary_weight != float(frame_defaults["primary_weight"])
            or constraint.secondary_weight != float(frame_defaults["secondary_weight"])
        ):
            raise ValueError(f"{config.hand.name}: frame settings differ for {constraint.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight the quantitative hand/config matrix")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml"),
    )
    parser.add_argument("--json", default=None, help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    defaults = _load_common_defaults()
    report: list[dict[str, object]] = []

    for hand in manifest.hands:
        config = load_retargeting_config(str(hand.config_path))
        _assert_shared_parameters(config, defaults)
        if len(config.distance_constraints) != 4:
            raise ValueError(f"{config.hand.name}: expected 4 distance constraints")
        if len(config.frame_constraints) != 1:
            raise ValueError(f"{config.hand.name}: expected 1 frame constraint")

        configured_vectors = len(config.vector_constraints)
        configured_distances = len(config.distance_constraints)
        configured_frames = len(config.frame_constraints)
        hand_model = HandModel(config.hand.mjcf_path)
        retargeter = VectorRetargeter(hand_model, config)
        if len(retargeter.config.vector_constraints) != configured_vectors:
            raise ValueError(f"{config.hand.name}: not all vector constraints resolved")
        if len(retargeter.config.distance_constraints) != configured_distances:
            raise ValueError(f"{config.hand.name}: not all distance constraints resolved")
        if len(retargeter.config.frame_constraints) != configured_frames:
            raise ValueError(f"{config.hand.name}: not all frame constraints resolved")
        geometry = RobotEvaluationGeometry(hand_model, config)
        if geometry.palm_scale() <= 1e-8:
            raise ValueError(f"{config.hand.name}: invalid palm scale")

        item = {
            "hand": config.hand.name,
            "main_table": hand.main_table,
            "nq": hand_model.nq,
            "independent_qpos": retargeter.get_independent_dof(),
            "mimic_joints": len(hand_model.mimic_joints),
            "vector_constraints": configured_vectors,
            "distance_constraints": configured_distances,
            "frame_constraints": configured_frames,
            "palm_scale_m": geometry.palm_scale(),
        }
        report.append(item)
        print(
            f"{config.hand.name:28s}"
            f" independent={item['independent_qpos']:2d}"
            f" nq={item['nq']:2d}"
            f" mimic={item['mimic_joints']:2d}"
            f" vectors={configured_vectors:2d}"
            " distance=4 frame=1"
        )

    print(f"Preflight passed for {len(report)} hands; planned runs={len(manifest.planned_runs())}")
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "experiment_id": manifest.experiment_id,
                    "hands": report,
                    "planned_runs": len(manifest.planned_runs()),
                },
                file_obj,
                indent=2,
                sort_keys=True,
            )
            file_obj.write("\n")


if __name__ == "__main__":
    main()
