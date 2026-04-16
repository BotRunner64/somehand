"""MuJoCo/SciPy-backed vector retargeting solver."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize

from somehand.domain import RetargetingConfig

from .hand_model import HandModel, mimic_joint_derivative
from .model_name_resolver import ModelNameResolver
from .vector_solver_objective import accumulate_direction_loss, compute_loss, compute_loss_and_grad, rotation_jacobian_to_axis_jacobian
from .vector_solver_primitives import TemporalFilter
from .vector_solver_targets import build_target_state, dist_activation, human_distance_scale, orthonormalize_frame_axes


class VectorRetargeter:
    """Optimizes robot joint angles to match human finger vector directions."""

    def __init__(self, hand_model: HandModel, config: RetargetingConfig):
        self.hand_model = hand_model
        self.config = config
        self.model = hand_model.model
        self.data = hand_model.data
        self._mimic_joints = hand_model.mimic_joints
        self._name_resolver = ModelNameResolver(self.model, hand_side=config.hand.side)

        self.landmark_filter = TemporalFilter(alpha=config.preprocess.temporal_filter_alpha)

        self._norm_delta = config.solver.norm_delta
        self._max_iterations = config.solver.max_iterations
        self._output_alpha = config.solver.output_alpha
        self._vector_loss_type = config.vector_loss.type
        self._vector_huber_delta = config.vector_loss.huber_delta

        self._target_directions: np.ndarray | None = None
        self._target_vectors: np.ndarray | None = None
        self._target_frame_primary_directions: np.ndarray | None = None
        self._target_frame_secondary_directions: np.ndarray | None = None
        self._target_angles: np.ndarray | None = None
        self._target_distances: np.ndarray | None = None
        self._raw_human_distances: np.ndarray | None = None
        self._last_qpos: np.ndarray | None = None
        self._vector_scale_landmark_idx = config.vector_loss.scale_landmarks[1]
        self._robot_vector_scale = 0.0
        self._robot_distance_scale = 0.0

        vector_scale_ids: list[tuple[int, bool]] = []
        for index, name in enumerate(config.vector_loss.scale_bodies):
            is_site = config.vector_loss.scale_body_types[index] == "site"
            object_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            resolved_name = self._name_resolver.resolve(name, obj_type=object_type, role="Vector scale body")
            body_id = mujoco.mj_name2id(self.model, object_type, resolved_name)
            if body_id < 0:
                raise ValueError(f"Vector scale body '{name}' not found")
            vector_scale_ids.append((body_id, is_site))

        self._forward()
        vector_scale_p0 = self._get_pos(vector_scale_ids[0][0], vector_scale_ids[0][1])
        vector_scale_p1 = self._get_pos(vector_scale_ids[1][0], vector_scale_ids[1][1])
        self._robot_vector_scale = (
            float(np.linalg.norm(vector_scale_p1 - vector_scale_p0)) * config.vector_loss.scaling
        )
        self._robot_distance_scale = self._robot_vector_scale
        middle_chain_points: list[tuple[int, bool]] = []
        for name, point_type in (
            ("middle_base", "body"),
            ("middle_mid", "body"),
            ("middle_distal", "body"),
            ("middle_tip", "site"),
        ):
            point = self._resolve_named_point(name, point_type, role="Distance scale", optional=True)
            if point is None:
                continue
            if middle_chain_points and middle_chain_points[-1] == point:
                continue
            middle_chain_points.append(point)
        if len(middle_chain_points) >= 2:
            self._robot_distance_scale = float(
                sum(
                    np.linalg.norm(
                        self._get_pos(b[0], b[1]) - self._get_pos(a[0], a[1])
                    )
                    for a, b in zip(middle_chain_points, middle_chain_points[1:])
                )
            )

        resolved_vector_constraints = []
        self.origin_ids: list[int] = []
        self.origin_is_site: list[bool] = []
        self.task_ids: list[int] = []
        self.task_is_site: list[bool] = []
        for constraint in config.vector_constraints:
            origin = self._resolve_named_point(
                constraint.robot[0],
                constraint.robot_types[0],
                role="Origin link",
                optional=constraint.optional,
            )
            task = self._resolve_named_point(
                constraint.robot[1],
                constraint.robot_types[1],
                role="Task link",
                optional=constraint.optional,
            )
            if origin is None or task is None:
                continue
            if origin == task:
                continue
            self.origin_ids.append(origin[0])
            self.origin_is_site.append(origin[1])
            self.task_ids.append(task[0])
            self.task_is_site.append(task[1])
            resolved_vector_constraints.append(constraint)
        self.config.vector_constraints = resolved_vector_constraints
        self.human_vector_pairs = [(pair[0], pair[1]) for pair in config.human_vector_pairs]
        self.origin_link_names = config.origin_link_names
        self.task_link_names = config.task_link_names
        self._weights = np.array(config.vector_weights, dtype=np.float64)
        self._per_vector_loss_types = [constraint.loss_type for constraint in config.vector_constraints]
        self._per_vector_loss_scales = [constraint.loss_scale for constraint in config.vector_constraints]
        if not self.human_vector_pairs:
            raise ValueError(f"retargeting config '{config.hand.name}' resolved zero vector constraints")

        self._angle_landmarks: list[tuple[int, int, int]] = []
        self._angle_qpos_ids: list[int] = []
        self._angle_dof_ids: list[int] = []
        self._angle_joint_ranges: list[tuple[float, float]] = []
        self._angle_weights: list[float] = []
        self._angle_scales: list[float] = []
        self._angle_inverts: list[bool] = []
        resolved_angle_constraints = []
        for constraint in config.angle_constraints:
            resolved_joint = self._name_resolver.resolve_optional(
                constraint.joint,
                obj_type=mujoco.mjtObj.mjOBJ_JOINT,
                role="Angle constraint joint",
            )
            if resolved_joint is None:
                if constraint.optional:
                    continue
                raise ValueError(f"Angle constraint joint '{constraint.joint}' not found in model")
            self._angle_landmarks.append(tuple(constraint.landmarks))
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, resolved_joint)
            if joint_id < 0:
                raise ValueError(f"Angle constraint joint '{constraint.joint}' not found in model")
            self._angle_qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self._angle_dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
            low, high = self.model.jnt_range[joint_id]
            self._angle_joint_ranges.append((float(low), float(high)))
            self._angle_weights.append(constraint.weight)
            self._angle_scales.append(float(constraint.scale))
            self._angle_inverts.append(bool(constraint.invert))
            resolved_angle_constraints.append(constraint)
        self.config.angle_constraints = resolved_angle_constraints

        self._dist_human_pairs: list[tuple[int, int]] = []
        self._dist_site_ids: list[tuple[int, bool, int, bool]] = []
        self._dist_weights: list[float] = []
        self._dist_scales: list[float] = []
        self._dist_thresholds: list[float] = []
        self._dist_activation_types: list[str] = []
        self._dist_scale_modes: list[str] = []
        self._prev_activations: np.ndarray | None = None
        self._smoothed_activations: np.ndarray | None = None
        self._activation_alpha: float = config.solver.activation_alpha
        resolved_distance_constraints = []
        for constraint in config.distance_constraints:
            first = self._resolve_named_point(
                constraint.robot[0],
                constraint.robot_types[0],
                role="Distance constraint",
                optional=constraint.optional,
            )
            second = self._resolve_named_point(
                constraint.robot[1],
                constraint.robot_types[1],
                role="Distance constraint",
                optional=constraint.optional,
            )
            if first is None or second is None:
                continue
            if first == second:
                continue
            self._dist_human_pairs.append((constraint.human[0], constraint.human[1]))
            self._dist_weights.append(constraint.weight)
            self._dist_scales.append(constraint.scale)
            self._dist_thresholds.append(constraint.threshold)
            self._dist_activation_types.append(constraint.activation_type)
            self._dist_scale_modes.append(constraint.scale_mode)
            self._dist_site_ids.append((first[0], first[1], second[0], second[1]))
            resolved_distance_constraints.append(constraint)
        self.config.distance_constraints = resolved_distance_constraints

        self._frame_names: list[str] = []
        self._frame_human_indices: list[tuple[int, int, int]] = []
        self._frame_primary_weights: list[float] = []
        self._frame_secondary_weights: list[float] = []
        self._frame_origin_ids: list[int] = []
        self._frame_origin_is_site: list[bool] = []
        self._frame_local_primary_axes: list[np.ndarray] = []
        self._frame_local_secondary_axes: list[np.ndarray] = []
        resolved_frame_constraints = []
        for constraint in config.frame_constraints:
            origin_id, origin_is_site = self._resolve_frame_point(
                constraint.robot_origin,
                constraint.robot_types[0],
                role=f"Frame origin ({constraint.name or 'unnamed'})",
                optional=constraint.optional,
            )
            if origin_id is None:
                continue
            primary_id, primary_is_site = self._resolve_frame_point(
                constraint.robot_primary,
                constraint.robot_types[1],
                role=f"Frame primary ({constraint.name or 'unnamed'})",
                optional=constraint.optional,
            )
            if primary_id is None:
                continue
            secondary_id, secondary_is_site = self._resolve_frame_point(
                constraint.robot_secondary,
                constraint.robot_types[2],
                role=f"Frame secondary ({constraint.name or 'unnamed'})",
                optional=constraint.optional,
            )
            if secondary_id is None:
                continue
            self._frame_names.append(constraint.name)
            self._frame_human_indices.append(
                (constraint.human_origin, constraint.human_primary, constraint.human_secondary)
            )
            self._frame_primary_weights.append(constraint.primary_weight)
            self._frame_secondary_weights.append(constraint.secondary_weight)
            self._frame_origin_ids.append(origin_id)
            self._frame_origin_is_site.append(origin_is_site)
            origin_position = self._get_pos(origin_id, origin_is_site)
            primary_position = self._get_pos(primary_id, primary_is_site)
            secondary_position = self._get_pos(secondary_id, secondary_is_site)
            origin_rotation = self._get_rot(origin_id, origin_is_site)
            local_primary_axis, local_secondary_axis = self._build_local_frame_axes(
                origin_rotation=origin_rotation,
                origin_position=origin_position,
                primary_position=primary_position,
                secondary_position=secondary_position,
                frame_name=constraint.name or "unnamed",
            )
            self._frame_local_primary_axes.append(local_primary_axis)
            self._frame_local_secondary_axes.append(local_secondary_axis)
            resolved_frame_constraints.append(constraint)
        self.config.frame_constraints = resolved_frame_constraints

        self._bounds: list[tuple[float | None, float | None]] = []
        for joint_index in range(self.model.nq):
            low, high = self.model.jnt_range[joint_index]
            if low < high:
                self._bounds.append((float(low), float(high)))
            else:
                self._bounds.append((None, None))

        self._mimic_qpos_indices = {int(item["qpos_id"]) for item in self._mimic_joints}
        self._independent_qpos_indices = [
            qpos_index for qpos_index in range(self.model.nq) if qpos_index not in self._mimic_qpos_indices
        ]
        self._reduced_bounds = [self._bounds[index] for index in self._independent_qpos_indices]
        self._mimic_by_source_qpos: dict[int, list[dict[str, object]]] = {}
        for mimic in self._mimic_joints:
            source_qpos_id = int(mimic["source_qpos_id"])
            self._mimic_by_source_qpos.setdefault(source_qpos_id, []).append(mimic)

        for joint_index in range(self.model.nq):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if joint_name and "thumb" in joint_name and "cmc" in joint_name:
                low, high = self.model.jnt_range[joint_index]
                if low < high:
                    self.data.qpos[joint_index] = (low + high) / 2.0

        self._forward()

    def _resolve_named_point(
        self,
        name: str,
        point_type: str,
        *,
        role: str,
        optional: bool,
    ) -> tuple[int, bool] | None:
        is_site = point_type == "site"
        obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
        resolved_name = (
            self._name_resolver.resolve_optional(name, obj_type=obj_type, role=role)
            if optional
            else self._name_resolver.resolve(name, obj_type=obj_type, role=role)
        )
        if resolved_name is None:
            return None
        point_id = mujoco.mj_name2id(self.model, obj_type, resolved_name)
        if point_id < 0:
            raise ValueError(f"{role} '{name}' not found in model")
        return point_id, is_site

    def _resolve_frame_point(
        self,
        name: str,
        point_type: str,
        *,
        role: str,
        optional: bool,
    ) -> tuple[int | None, bool]:
        resolved = self._resolve_named_point(name, point_type, role=role, optional=optional)
        if resolved is None:
            return None, point_type == "site"
        return resolved

    def _expand_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if qpos.shape[0] == self.model.nq:
            full_qpos = qpos.copy()
        else:
            full_qpos = self.data.qpos.copy()
            for reduced_index, qpos_index in enumerate(self._independent_qpos_indices):
                full_qpos[qpos_index] = qpos[reduced_index]
        return self.hand_model.apply_mimic_constraints(full_qpos)

    def _reduce_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.asarray([qpos[index] for index in self._independent_qpos_indices], dtype=np.float64)

    def _reduce_grad(self, grad: np.ndarray) -> np.ndarray:
        reduced_grad = np.asarray([grad[index] for index in self._independent_qpos_indices], dtype=np.float64)
        for reduced_index, qpos_index in enumerate(self._independent_qpos_indices):
            for mimic in self._mimic_by_source_qpos.get(qpos_index, ()):
                source_value = float(self.data.qpos[qpos_index])
                derivative = mimic_joint_derivative(mimic, source_value)
                reduced_grad[reduced_index] += derivative * grad[int(mimic["qpos_id"])]
        return reduced_grad

    def _forward(self, qpos: np.ndarray | None = None) -> None:
        if qpos is not None:
            self.data.qpos[:] = self._expand_qpos(qpos)
        mujoco.mj_fwdPosition(self.model, self.data)

    def _get_pos(self, index: int, is_site: bool) -> np.ndarray:
        if is_site:
            return self.data.site_xpos[index].copy()
        return self.data.xpos[index].copy()

    def _dist_activation(self, index: int, raw_dist: float) -> float:
        return dist_activation(self._dist_activation_types[index], self._dist_thresholds[index], raw_dist)

    @staticmethod
    def _human_distance_scale(landmarks: np.ndarray) -> float:
        return human_distance_scale(landmarks)

    def _get_rot(self, index: int, is_site: bool) -> np.ndarray:
        if is_site:
            return self.data.site_xmat[index].reshape(3, 3).copy()
        return self.data.xmat[index].reshape(3, 3).copy()

    @staticmethod
    def _orthonormalize_frame_axes(
        primary_vector: np.ndarray,
        secondary_vector: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return orthonormalize_frame_axes(primary_vector, secondary_vector)

    def _build_local_frame_axes(
        self,
        *,
        origin_rotation: np.ndarray,
        origin_position: np.ndarray,
        primary_position: np.ndarray,
        secondary_position: np.ndarray,
        frame_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_primary = origin_rotation.T @ (primary_position - origin_position)
        local_secondary = origin_rotation.T @ (secondary_position - origin_position)
        primary_axis, secondary_axis = self._orthonormalize_frame_axes(local_primary, local_secondary)
        if primary_axis is None or secondary_axis is None:
            raise ValueError(f"Frame constraint '{frame_name}' cannot build a valid local coordinate frame")
        return primary_axis, secondary_axis

    def _get_robot_vectors(self) -> np.ndarray:
        vectors = np.empty((len(self.origin_ids), 3))
        for index in range(len(self.origin_ids)):
            origin = self._get_pos(self.origin_ids[index], self.origin_is_site[index])
            task = self._get_pos(self.task_ids[index], self.task_is_site[index])
            vectors[index] = task - origin
        return vectors

    def _get_robot_frame_axes(self) -> tuple[np.ndarray, np.ndarray]:
        primary_axes = np.empty((len(self._frame_origin_ids), 3), dtype=np.float64)
        secondary_axes = np.empty((len(self._frame_origin_ids), 3), dtype=np.float64)
        for index in range(len(self._frame_origin_ids)):
            rotation = self._get_rot(self._frame_origin_ids[index], self._frame_origin_is_site[index])
            primary_axes[index] = rotation @ self._frame_local_primary_axes[index]
            secondary_axes[index] = rotation @ self._frame_local_secondary_axes[index]
        return primary_axes, secondary_axes

    def _get_robot_frame_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_robot_frame_axes()

    @staticmethod
    def _rotation_jacobian_to_axis_jacobian(jac_rot: np.ndarray, axis: np.ndarray) -> np.ndarray:
        return rotation_jacobian_to_axis_jacobian(jac_rot, axis)

    def _accumulate_direction_loss(
        self,
        vector: np.ndarray,
        target_dir: np.ndarray,
        weight: float,
        jac_diff: np.ndarray | None = None,
        grad: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray | None]:
        return accumulate_direction_loss(vector, target_dir, weight, jac_diff=jac_diff, grad=grad)

    def _get_effective_weight(self, index: int) -> float:
        return self._weights[index]

    def _get_loss_type(self, index: int) -> str:
        override = self._per_vector_loss_types[index]
        return override if override else self._vector_loss_type

    def _compute_loss(self, qpos: np.ndarray) -> float:
        return compute_loss(self, qpos)

    def _compute_loss_and_grad(self, qpos: np.ndarray) -> tuple[float, np.ndarray]:
        return compute_loss_and_grad(self, qpos)

    def update_targets(
        self,
        landmarks_3d: np.ndarray,
        hand_side: str = "right",
    ) -> None:
        build_target_state(self, landmarks_3d, hand_side=hand_side)

    def solve(self) -> np.ndarray:
        if self._target_directions is None:
            return self.hand_model.apply_mimic_constraints(self.data.qpos.copy())

        x0 = self._reduce_qpos(self.data.qpos.copy())
        previous_qpos = None if self._last_qpos is None else self._last_qpos.copy()

        result = minimize(
            fun=self._compute_loss_and_grad,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=self._reduced_bounds,
            options={
                "maxiter": self._max_iterations,
                "ftol": 1e-6,
            },
        )

        qpos = self._expand_qpos(result.x.copy())
        if previous_qpos is not None and self._output_alpha < 1.0:
            qpos = previous_qpos + self._output_alpha * (qpos - previous_qpos)
            qpos = self.hand_model.apply_mimic_constraints(qpos)

        self._last_qpos = qpos.copy()
        self._forward(qpos)
        return qpos

    def compute_error(self) -> float:
        self._forward()
        if self._target_directions is None:
            return 0.0
        return self._compute_loss(self._reduce_qpos(self.data.qpos.copy()))

    def get_target_directions(self) -> np.ndarray | None:
        if self._target_directions is None:
            return None
        return self._target_directions.copy()

    def get_frame_target_directions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        primary = None if self._target_frame_primary_directions is None else self._target_frame_primary_directions.copy()
        secondary = (
            None if self._target_frame_secondary_directions is None else self._target_frame_secondary_directions.copy()
        )
        return primary, secondary

    def get_robot_scale(self) -> float:
        return float(self._robot_distance_scale)
