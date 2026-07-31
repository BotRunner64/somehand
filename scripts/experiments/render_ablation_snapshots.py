"""Render the same opposition frame for Full and w/o-frame results."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from somehand.infrastructure.config_loader import load_retargeting_config
from somehand.infrastructure.hand_model import HandModel
from somehand.infrastructure.quantitative_artifacts import load_annotations, load_quantitative_result
from somehand.infrastructure.quantitative_manifest import load_quantitative_manifest
from somehand.runtime.sink_rendering import create_offscreen_renderer
from somehand.runtime.viewer_camera import configure_free_camera, try_frame_hand_camera


def _result_qpos_for_frame(manifest, method: str, frame_index: int) -> np.ndarray:
    metadata_path = manifest.output_dir / manifest.reference_hand / f"{method}.json"
    arrays, _ = load_quantitative_result(metadata_path)
    matches = np.flatnonzero(arrays["frame_index"] == frame_index)
    if len(matches) != 1:
        raise ValueError(f"{metadata_path}: frame {frame_index} is unavailable")
    return arrays["qpos"][matches[0]]


def _camera_from_full_pose(hand_model: HandModel, qpos: np.ndarray, *, width: int, height: int):
    hand_model.set_qpos(qpos)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    if not try_frame_hand_camera(
        camera,
        model=hand_model.model,
        data=hand_model.data,
        aspect_ratio=width / height,
    ):
        raise RuntimeError("Could not frame the reference hand")
    return {
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
        "lookat": tuple(float(value) for value in camera.lookat),
    }


def _render_pose(
    config_path: Path,
    qpos: np.ndarray,
    *,
    camera_settings: dict[str, object],
    width: int,
    height: int,
) -> np.ndarray:
    config = load_retargeting_config(str(config_path))
    hand_model = HandModel(config.hand.mjcf_path)
    hand_model.set_qpos(qpos)
    renderer = create_offscreen_renderer(hand_model.model, width=width, height=height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    configure_free_camera(camera, **camera_settings)
    try:
        renderer.update_scene(hand_model.data, camera=camera)
        return np.array(renderer.render(), copy=True)
    finally:
        renderer.close()


def _load_frame_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Full/w-o-frame opposition snapshots")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "quantitative_v1.yaml"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_quantitative_manifest(args.manifest, project_root=PROJECT_ROOT)
    annotations = load_annotations(manifest.annotation_path)
    opposition = [
        episode for episode in annotations["episodes"] if episode["action"] == "thumb_opposition"
    ]
    if len(opposition) != 5:
        raise ValueError("Expected five thumb-opposition episodes")
    reference_episode = opposition[2]
    frame_index = (
        int(reference_episode["hold_start_frame"])
        + int(reference_episode["hold_end_frame_exclusive"])
    ) // 2

    reference_spec = next(hand for hand in manifest.hands if hand.name == manifest.reference_hand)
    full_qpos = _result_qpos_for_frame(manifest, "full", frame_index)
    no_frame_qpos = _result_qpos_for_frame(manifest, "no_frame", frame_index)
    reference_config = load_retargeting_config(str(reference_spec.config_path))
    reference_model = HandModel(reference_config.hand.mjcf_path)
    camera_settings = _camera_from_full_pose(
        reference_model,
        full_qpos,
        width=args.width,
        height=args.height,
    )
    images = {
        "no_frame": _render_pose(
            reference_spec.config_path,
            no_frame_qpos,
            camera_settings=camera_settings,
            width=args.width,
            height=args.height,
        ),
        "full": _render_pose(
            reference_spec.config_path,
            full_qpos,
            camera_settings=camera_settings,
            width=args.width,
            height=args.height,
        ),
    }

    output_dir = (
        manifest.output_dir / "summary"
        if args.output_dir is None
        else Path(args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for method, image in images.items():
        plt.imsave(output_dir / f"opposition_{method}.png", image)

    frame_rows = _load_frame_rows(output_dir / "frame_ablation.csv")
    means = [float(frame_rows[index]["mean_thumb_frame_error_deg"]) for index in range(2)]
    lower = [
        means[index] - float(frame_rows[index]["mean_thumb_frame_error_ci95_low_deg"])
        for index in range(2)
    ]
    upper = [
        float(frame_rows[index]["mean_thumb_frame_error_ci95_high_deg"]) - means[index]
        for index in range(2)
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.5),
        gridspec_kw={"width_ratios": (0.9, 1.0, 1.0)},
    )
    axes[0].bar(["w/o frame", "Full"], means, color=["#D55E00", "#0072B2"], width=0.62)
    axes[0].errorbar(
        np.arange(2),
        means,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.0,
    )
    axes[0].set_ylabel("Mean thumb-frame error (deg)")
    axes[0].grid(axis="y", alpha=0.2)
    for axis, method, label in (
        (axes[1], "no_frame", "w/o frame"),
        (axes[2], "full", "Full"),
    ):
        axis.imshow(images[method])
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(f"Same opposition input frame: {frame_index}", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_dir / "frame_ablation_with_postures.png", dpi=300)
    figure.savefig(output_dir / "frame_ablation_with_postures.pdf")
    plt.close(figure)
    print(f"Saved opposition snapshots and composite figure to {output_dir}")


if __name__ == "__main__":
    main()
