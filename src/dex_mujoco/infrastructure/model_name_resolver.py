"""Resolve semantic config names to concrete MuJoCo object names."""

from __future__ import annotations

import mujoco


_SIDE_PREFIXES = ("lh", "rh")
_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "thumb_dip": ("thumb_ip",),
    "thumb_ip": ("thumb_dip",),
    "middle_dip": ("middle_distal",),
    "middle_distal": ("middle_dip",),
}


def _strip_side_prefix(name: str) -> str:
    for prefix in _SIDE_PREFIXES:
        token = f"{prefix}_"
        if name.startswith(token):
            return name[len(token):]
    return name


class ModelNameResolver:
    """Maps semantic body/site/joint names onto a specific MuJoCo model."""

    def __init__(self, model: mujoco.MjModel, *, hand_side: str):
        self.model = model
        self.hand_side = hand_side
        self._preferred_prefix = "lh" if hand_side == "left" else "rh"
        self._body_names = self._collect_names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        self._site_names = self._collect_names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
        self._joint_names = self._collect_names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)

    def _collect_names(self, obj_type, count: int) -> set[str]:
        names: set[str] = set()
        for index in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, index)
            if name:
                names.add(name)
        return names

    def _candidate_names(self, semantic_name: str) -> list[str]:
        semantic = _strip_side_prefix(semantic_name)
        semantic_variants = [semantic_name, semantic]
        semantic_variants.extend(_SEMANTIC_ALIASES.get(semantic, ()))

        candidates: list[str] = []
        for variant in semantic_variants:
            stripped_variant = _strip_side_prefix(variant)
            for candidate in (variant, f"{self._preferred_prefix}_{stripped_variant}", stripped_variant):
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def resolve(self, semantic_name: str, *, obj_type, role: str) -> str:
        names = {
            mujoco.mjtObj.mjOBJ_BODY: self._body_names,
            mujoco.mjtObj.mjOBJ_SITE: self._site_names,
            mujoco.mjtObj.mjOBJ_JOINT: self._joint_names,
        }[obj_type]
        for candidate in self._candidate_names(semantic_name):
            if candidate in names:
                return candidate
        raise ValueError(
            f"{role} semantic name '{semantic_name}' could not be resolved for {self.hand_side} hand model"
        )
