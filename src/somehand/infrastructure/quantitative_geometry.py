"""Robot semantic geometry used by quantitative experiment metrics."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from somehand.domain.config import FrameConstraint, RetargetingConfig
from somehand.domain.quantitative_metrics import orthonormal_frame

from .hand_model import HandModel
from .model_name_resolver import ModelNameResolver


FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True, slots=True)
class ResolvedSemanticPoint:
    semantic_name: str
    model_name: str
    point_id: int
    is_site: bool


class RobotEvaluationGeometry:
    """Resolve and query a model-independent set of robot-hand semantics."""

    def __init__(self, hand_model: HandModel, full_config: RetargetingConfig):
        self.hand_model = hand_model
        self.model = hand_model.model
        self.data = hand_model.data
        self.config = full_config
        self._resolver = ModelNameResolver(self.model, hand_side=full_config.hand.side)

        self._middle_base = self._resolve_point("middle_base", "body", role="Evaluation palm scale")
        self._finger_points: dict[str, tuple[ResolvedSemanticPoint, ResolvedSemanticPoint, ResolvedSemanticPoint]] = {}
        self._tip_collision_geoms: dict[str, tuple[int, ...]] = {}
        for finger in FINGER_NAMES:
            base = self._resolve_point(f"{finger}_base", "body", role="Evaluation finger base")
            distal = self._resolve_body_with_fallback(
                (f"{finger}_distal", f"{finger}_mid"),
                role="Evaluation finger distal",
            )
            tip = self._resolve_point(f"{finger}_tip", "site", role="Evaluation fingertip")
            self._finger_points[finger] = (base, distal, tip)
            self._tip_collision_geoms[finger] = self._collision_geoms_for_site(tip)

        self._frame_constraint: FrameConstraint | None = None
        self._frame_origin: ResolvedSemanticPoint | None = None
        self._frame_local_primary: np.ndarray | None = None
        self._frame_local_secondary: np.ndarray | None = None
        if full_config.frame_constraints:
            self._initialize_frame(full_config.frame_constraints[0])

    def _resolve_body_with_fallback(
        self,
        semantic_names: tuple[str, ...],
        *,
        role: str,
    ) -> ResolvedSemanticPoint:
        for semantic_name in semantic_names:
            resolved = self._resolver.resolve_optional(
                semantic_name,
                obj_type=mujoco.mjtObj.mjOBJ_BODY,
                role=role,
            )
            if resolved is None:
                continue
            return ResolvedSemanticPoint(
                semantic_name=semantic_name,
                model_name=resolved,
                point_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, resolved),
                is_site=False,
            )
        raise ValueError(f"{role} could not resolve any of: {', '.join(semantic_names)}")

    def _resolve_point(
        self,
        semantic_name: str,
        point_type: str,
        *,
        role: str,
    ) -> ResolvedSemanticPoint:
        is_site = point_type == "site"
        obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
        resolved = self._resolver.resolve(semantic_name, obj_type=obj_type, role=role)
        return ResolvedSemanticPoint(
            semantic_name=semantic_name,
            model_name=resolved,
            point_id=mujoco.mj_name2id(self.model, obj_type, resolved),
            is_site=is_site,
        )

    def _position(self, point: ResolvedSemanticPoint) -> np.ndarray:
        if point.is_site:
            return self.data.site_xpos[point.point_id].copy()
        return self.data.xpos[point.point_id].copy()

    def _rotation(self, point: ResolvedSemanticPoint) -> np.ndarray:
        if point.is_site:
            return self.data.site_xmat[point.point_id].reshape(3, 3).copy()
        return self.data.xmat[point.point_id].reshape(3, 3).copy()

    def _collision_geoms_for_site(self, site: ResolvedSemanticPoint) -> tuple[int, ...]:
        body_id = int(self.model.site_bodyid[site.point_id])
        collidable = tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == body_id
            and (int(self.model.geom_contype[geom_id]) != 0 or int(self.model.geom_conaffinity[geom_id]) != 0)
        )
        if not collidable:
            raise ValueError(f"Evaluation fingertip '{site.model_name}' has no collision geometry")
        return collidable

    def _initialize_frame(self, constraint: FrameConstraint) -> None:
        origin = self._resolve_point(
            constraint.robot_origin,
            constraint.robot_types[0],
            role=f"Evaluation frame origin ({constraint.name})",
        )
        primary = self._resolve_point(
            constraint.robot_primary,
            constraint.robot_types[1],
            role=f"Evaluation frame primary ({constraint.name})",
        )
        secondary = self._resolve_point(
            constraint.robot_secondary,
            constraint.robot_types[2],
            role=f"Evaluation frame secondary ({constraint.name})",
        )
        origin_rotation = self._rotation(origin)
        local_primary = origin_rotation.T @ (self._position(primary) - self._position(origin))
        local_secondary = origin_rotation.T @ (self._position(secondary) - self._position(origin))
        local_frame = orthonormal_frame(local_primary, local_secondary)
        if local_frame is None:
            raise ValueError(f"Evaluation frame '{constraint.name}' is degenerate")

        self._frame_constraint = constraint
        self._frame_origin = origin
        self._frame_local_primary = local_frame[:, 0]
        self._frame_local_secondary = local_frame[:, 1]

    def palm_origin(self) -> np.ndarray:
        """Return the model world origin, used as the robot palm-base origin."""
        return self.data.xpos[0].copy()

    def palm_scale(self) -> float:
        return float(np.linalg.norm(self._position(self._middle_base) - self.palm_origin()))

    def fingertip_positions(self) -> np.ndarray:
        return np.asarray(
            [self._position(self._finger_points[finger][2]) for finger in FINGER_NAMES],
            dtype=np.float64,
        )

    def common_direction_vectors(self) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for finger in FINGER_NAMES:
            base, _, tip = self._finger_points[finger]
            base_position = self._position(base)
            tip_position = self._position(tip)
            vectors.append(tip_position - base_position)
        return np.asarray(vectors, dtype=np.float64)

    def fingertip_site_distances(self) -> np.ndarray:
        positions = self.fingertip_positions()
        return np.linalg.norm(positions[1:] - positions[0], axis=1)

    def fingertip_surface_distances(self) -> np.ndarray:
        thumb_geoms = self._tip_collision_geoms["thumb"]
        distances = np.empty(4, dtype=np.float64)
        for index, finger in enumerate(FINGER_NAMES[1:]):
            distances[index] = min(
                float(mujoco.mj_geomDistance(self.model, self.data, thumb_geom, finger_geom, 10.0, None))
                for thumb_geom in thumb_geoms
                for finger_geom in self._tip_collision_geoms[finger]
            )
        return distances

    @property
    def has_frame(self) -> bool:
        return self._frame_constraint is not None

    def human_frame_rotation(self, landmarks: np.ndarray) -> np.ndarray | None:
        if self._frame_constraint is None:
            return None
        constraint = self._frame_constraint
        origin = landmarks[constraint.human_origin]
        return orthonormal_frame(
            landmarks[constraint.human_primary] - origin,
            landmarks[constraint.human_secondary] - origin,
        )

    def robot_frame_rotation(self) -> np.ndarray | None:
        if (
            self._frame_origin is None
            or self._frame_local_primary is None
            or self._frame_local_secondary is None
        ):
            return None
        origin_rotation = self._rotation(self._frame_origin)
        return orthonormal_frame(
            origin_rotation @ self._frame_local_primary,
            origin_rotation @ self._frame_local_secondary,
        )

    def describe(self) -> dict[str, object]:
        finger_points = {
            finger: {
                "base": self._describe_point(self._finger_points[finger][0]),
                "distal": self._describe_point(self._finger_points[finger][1]),
                "tip": self._describe_point(self._finger_points[finger][2]),
                "collision_geom_ids": list(self._tip_collision_geoms[finger]),
            }
            for finger in FINGER_NAMES
        }
        return {
            "palm_origin": "world",
            "palm_scale_point": self._describe_point(self._middle_base),
            "fingers": finger_points,
            "frame_name": None if self._frame_constraint is None else self._frame_constraint.name,
        }

    @staticmethod
    def _describe_point(point: ResolvedSemanticPoint) -> dict[str, object]:
        return {
            "semantic_name": point.semantic_name,
            "model_name": point.model_name,
            "point_id": point.point_id,
            "point_type": "site" if point.is_site else "body",
        }
