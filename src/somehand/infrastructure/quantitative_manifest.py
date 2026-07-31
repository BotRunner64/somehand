"""Configuration loading for the quantitative experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class QuantitativeHandSpec:
    config_path: Path
    main_table: bool = False

    @property
    def name(self) -> str:
        return self.config_path.stem


@dataclass(frozen=True, slots=True)
class QuantitativeManifest:
    experiment_id: str
    recording_path: Path
    annotation_path: Path | None
    output_dir: Path
    sample_rate_hz: int
    warmup_frames: int
    contact_threshold_m: float
    human_pinch_threshold_m: float
    minimum_contact_duration_s: float
    reference_hand: str
    hands: tuple[QuantitativeHandSpec, ...]

    def planned_runs(self) -> tuple[tuple[QuantitativeHandSpec, str], ...]:
        runs: list[tuple[QuantitativeHandSpec, str]] = [(hand, "full") for hand in self.hands]
        reference = next((hand for hand in self.hands if hand.name == self.reference_hand), None)
        if reference is None:
            raise ValueError(f"reference_hand {self.reference_hand!r} is not present in hands")
        runs.extend(((reference, "no_distance"), (reference, "no_frame")))
        return tuple(runs)


def _resolve_project_path(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_quantitative_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> QuantitativeManifest:
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError("Quantitative manifest root must be a mapping")

    root = Path(project_root).resolve()
    hand_specs: list[QuantitativeHandSpec] = []
    for item in payload.get("hands", []):
        if not isinstance(item, dict) or not item.get("config"):
            raise ValueError("Each quantitative hand entry must define config")
        hand_specs.append(
            QuantitativeHandSpec(
                config_path=_resolve_project_path(root, item["config"]),
                main_table=bool(item.get("main_table", False)),
            )
        )
    if not hand_specs:
        raise ValueError("Quantitative manifest must define at least one hand")

    annotation_value = payload.get("annotations")
    manifest = QuantitativeManifest(
        experiment_id=str(payload.get("experiment_id", "")).strip(),
        recording_path=_resolve_project_path(root, payload["recording"]),
        annotation_path=(
            None
            if annotation_value is None or not str(annotation_value).strip()
            else _resolve_project_path(root, annotation_value)
        ),
        output_dir=_resolve_project_path(root, payload["output_dir"]),
        sample_rate_hz=int(payload.get("sample_rate_hz", 80)),
        warmup_frames=int(payload.get("warmup_frames", 80)),
        contact_threshold_m=float(payload.get("contact_threshold_m", 0.001)),
        human_pinch_threshold_m=float(payload.get("human_pinch_threshold_m", 0.020)),
        minimum_contact_duration_s=float(payload.get("minimum_contact_duration_s", 0.1)),
        reference_hand=str(payload.get("reference_hand", "")).strip(),
        hands=tuple(hand_specs),
    )
    if not manifest.experiment_id:
        raise ValueError("experiment_id must not be empty")
    if manifest.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if manifest.warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    if manifest.contact_threshold_m < 0.0:
        raise ValueError("contact_threshold_m must be >= 0")
    if manifest.human_pinch_threshold_m <= 0.0:
        raise ValueError("human_pinch_threshold_m must be > 0")
    if manifest.minimum_contact_duration_s < 0.0:
        raise ValueError("minimum_contact_duration_s must be >= 0")
    manifest.planned_runs()
    return manifest
