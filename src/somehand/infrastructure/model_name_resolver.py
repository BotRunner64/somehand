"""Resolve semantic config names to concrete MuJoCo object names."""

from __future__ import annotations

import mujoco
import re


_SIDE_PREFIXES = ("lh", "rh", "left", "right", "l", "r", "L", "R")
_PREFERRED_SIDE_PREFIXES: dict[str, tuple[str, ...]] = {
    "left": ("lh", "left", "l", "L"),
    "right": ("rh", "right", "r", "R"),
}
_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "thumb_dip": ("thumb_ip",),
    "thumb_ip": ("thumb_dip",),
    "middle_dip": ("middle_distal",),
    "middle_distal": ("middle_dip",),
    "middle_base": ("middle_metacarpals", "middle_proximal", "middle_proximal_link", "f_link3_1", "finger3_link1"),
}
_FINGER_NUMBERS = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "pinky": 5,
    "little": 5,
}
_FINGER_ABBREVIATIONS = {
    "thumb": "th",
    "index": "if",
    "middle": "mf",
    "ring": "rf",
    "pinky": "lf",
}
_GENERIC_SEMANTIC_PATTERN = re.compile(r"^(thumb|index|middle|ring|pinky|little)_(.+)$")
_DEX5_PATTERN = re.compile(r"^(link)_(\d+)(.*)$")
_UPPERCASE_SEGMENTS = {
    "_cmc_vl": "_CMC_VL",
    "_mcp_vl": "_MCP_VL",
    "_mc_": "_MC_",
    "_pp": "_PP",
    "_mp": "_MP",
    "_dp": "_DP",
}


def _finger_role_candidates(finger: str, role: str) -> tuple[str, ...]:
    canonical_finger = "pinky" if finger == "little" else finger
    finger_labels = (canonical_finger, "little") if canonical_finger == "pinky" else (canonical_finger,)
    number = _FINGER_NUMBERS[canonical_finger]
    abbreviation = _FINGER_ABBREVIATIONS[canonical_finger]
    if canonical_finger == "thumb":
        body_roles = {
            "base": (
                "thumb_metacarpals",
                "thumb_metacarpal_link",
                "thumb_metacarpals_base2",
                "thumb_abad",
                "thumb_abad_link",
                "thumb_mc",
                "thumb_mcp_vl",
                "thumb_proximal_base",
                "thumb_proximal",
                "thumb_2",
                "f_link1_2",
                "finger1_link2",
                "link_12",
                f"{abbreviation}_root_link",
                f"{abbreviation}_proximal_link",
            ),
            "mid": (
                "thumb_proximal",
                "thumb_proximal_link",
                "thumb_distal",
                "thumb_pp",
                "thumb_pip_link",
                "thumb_intermediate",
                "thumb_3",
                "f_link1_3",
                "finger1_link3",
                "link_13",
                f"{abbreviation}_slider_link",
                f"{abbreviation}_connecting_link",
            ),
            "distal": (
                "thumb_distal",
                "thumb_distal_link",
                "thumb_dp",
                "thumb_dip_link",
                "thumb_4",
                "f_link1_4",
                "finger1_link4",
                "link_14",
                f"{abbreviation}_distal_link",
            ),
            "tip": (
                "thumb_distal_tip",
                "thumb_distal_link_tip",
                "thumb_dip_link_tip",
                "thumb_dp_tip",
                "thumb_4_tip",
                "f_link1_4_tip",
                "finger1_link4_tip",
                "link_14_tip",
                f"{abbreviation}_distal_link_tip",
                f"{abbreviation}_connecting_link_tip",
            ),
        }
        joint_roles = {
            "proximal_flex": ("thumb_mcp", "thumb_proximal_joint", "f_joint1_3", "finger1_joint3"),
            "distal_flex": ("thumb_dip", "thumb_ip", "thumb_distal_joint", "f_joint1_4", "finger1_joint4"),
        }
        return body_roles.get(role, joint_roles.get(role, ()))
    base_names = tuple(
        name
        for label in finger_labels
        for name in (
            f"{label}_metacarpals",
            f"{label}_proximal",
            f"{label}_proximal_link",
            f"{label}_pp",
            f"{label}_mcp_vl",
            f"{label}_pip_link",
            f"{label}_abad_link",
            f"{label}_1",
        )
    )
    mid_names = tuple(
        name
        for label in finger_labels
        for name in (
            f"{label}_middle",
            f"{label}_distal",
            f"{label}_distal_link",
            f"{label}_dip_link",
            f"{label}_intermediate",
            f"{label}_mp",
            f"{label}_pip_link",
            f"{label}_2",
            f"{label}_3",
        )
    )
    distal_names = tuple(
        name
        for label in finger_labels
        for name in (
            f"{label}_distal",
            f"{label}_distal_link",
            f"{label}_intermediate",
            f"{label}_dp",
            f"{label}_dip_link",
            f"{label}_2",
            f"{label}_3",
            f"{label}_4",
        )
    )
    tip_names = tuple(
        name
        for label in finger_labels
        for name in (
            f"{label}_distal_tip",
            f"{label}_middle_tip",
            f"{label}_distal_link_tip",
            f"{label}_dip_link_tip",
            f"{label}_intermediate_tip",
            f"{label}_dp_tip",
            f"{label}_2_tip",
            f"{label}_4_tip",
        )
    )
    body_roles = {
        "base": (
            *base_names,
            f"f_link{number}_1",
            f"finger{number}_link1",
            f"link_{number}2",
            f"link_{number}1",
            f"{abbreviation}_slider_link",
            f"{abbreviation}_proximal_link",
        ),
        "mid": (
            *mid_names,
            f"f_link{number}_2",
            f"finger{number}_link2",
            f"link_{number}3",
            f"link_{number}2",
            f"{abbreviation}_distal_link",
            f"{abbreviation}_connecting_link",
        ),
        "distal": (
            *distal_names,
            f"f_link{number}_3",
            f"finger{number}_link3",
            f"link_{number}3",
            f"link_{number}4",
            f"{abbreviation}_connecting_link",
            f"{abbreviation}_distal_link",
        ),
        "tip": (
            *tip_names,
            f"f_link{number}_4_tip",
            f"finger{number}_link4_tip",
            f"link_{number}4_tip",
            f"{abbreviation}_connecting_link_tip",
            f"{abbreviation}_distal_link_tip",
        ),
    }
    joint_roles = {
        "base_flex": (
            f"{finger}_mcp_pitch",
            f"{finger}_base_pitch",
            f"{finger}_proximal_joint",
            f"f_joint{number}_1",
            f"finger{number}_joint1",
        ),
        "proximal_flex": (
            f"{finger}_pip",
            f"{finger}_middle_joint",
            f"f_joint{number}_2",
            f"finger{number}_joint2",
        ),
        "distal_flex": (
            f"{finger}_dip",
            f"{finger}_distal_joint",
            f"f_joint{number}_3",
            f"finger{number}_joint3",
        ),
    }
    return body_roles.get(role, joint_roles.get(role, ()))


def _strip_side_prefix(name: str) -> str:
    for prefix in _SIDE_PREFIXES:
        token = f"{prefix}_"
        if name.startswith(token):
            return name[len(token):]
    return name


def _dex5_side_variants(candidate: str, hand_side: str) -> tuple[str, ...]:
    match = _DEX5_PATTERN.match(candidate)
    if not match:
        return ()
    _, digits, suffix = match.groups()
    side_letter = "L" if hand_side == "left" else "R"
    return (f"Link_{digits}{side_letter}{suffix}",)


def _case_variants(candidate: str) -> tuple[str, ...]:
    variants = [candidate]
    for source, target in _UPPERCASE_SEGMENTS.items():
        if source in candidate:
            transformed = candidate.replace(source, target)
            if transformed not in variants:
                variants.append(transformed)
    if candidate.endswith("_mc"):
        transformed = candidate[:-3] + "_MC"
        if transformed not in variants:
            variants.append(transformed)
    return tuple(variants)


class ModelNameResolver:
    """Maps semantic body/site/joint names onto a specific MuJoCo model."""

    def __init__(self, model: mujoco.MjModel, *, hand_side: str):
        self.model = model
        self.hand_side = hand_side
        self._preferred_prefixes = _PREFERRED_SIDE_PREFIXES[hand_side]
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
        if semantic.startswith("pinky_"):
            semantic_variants.append("little_" + semantic[len("pinky_"):])
        elif semantic.startswith("little_"):
            semantic_variants.append("pinky_" + semantic[len("little_"):])
        semantic_variants.extend(_SEMANTIC_ALIASES.get(semantic, ()))
        generic_match = _GENERIC_SEMANTIC_PATTERN.match(semantic)
        if generic_match:
            semantic_variants.extend(_finger_role_candidates(generic_match.group(1), generic_match.group(2)))

        candidates: list[str] = []
        for variant in semantic_variants:
            stripped_variant = _strip_side_prefix(variant)
            prefixed_candidates = tuple(
                f"{prefix}_{stripped_variant}" for prefix in self._preferred_prefixes
            )
            dex5_candidates = _dex5_side_variants(stripped_variant, self.hand_side)
            for raw_candidate in (variant, *prefixed_candidates, stripped_variant, *dex5_candidates):
                for candidate in _case_variants(raw_candidate):
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

    def resolve_optional(self, semantic_name: str, *, obj_type, role: str) -> str | None:
        try:
            return self.resolve(semantic_name, obj_type=obj_type, role=role)
        except ValueError:
            return None
