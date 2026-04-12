"""MuJoCo/SciPy-backed vector retargeting solver."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize

from somehand.domain import RetargetingConfig, preprocess_landmarks

from .hand_model import HandModel
from .model_name_resolver import ModelNameResolver


def _huber_loss(distance: float, delta: float) -> float:
    if distance <= delta:
        return 0.5 * distance * distance
    return delta * (distance - 0.5 * delta)


def _huber_grad(distance: float, delta: float) -> float:
    if distance <= delta:
        return distance
    return delta


class TemporalFilter:
    """Exponential moving average filter for smooth landmark tracking."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def filter(self, value: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value.copy()
            return value
        self._prev = self.alpha * value + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self) -> None:
        self._prev = None


class VectorRetargeter:
    """Optimizes robot joint angles to match human finger vector directions."""

    def __init__(self, hand_model: HandModel, config: RetargetingConfig):
        self.hand_model = hand_model
        self.config = config
        self.model = hand_model.model
        self.data = hand_model.data
        self._mimic_joints = hand_model.mimic_joints
        self._name_resolver = ModelNameResolver(self.model, hand_side=config.hand.side)

        self.human_vector_pairs = [(pair[0], pair[1]) for pair in config.human_vector_pairs]
        self.origin_link_names = config.origin_link_names
        self.task_link_names = config.task_link_names

        self.origin_ids: list[int] = []
        self.origin_is_site: list[bool] = []
        for index, name in enumerate(self.origin_link_names):
            is_site = config.origin_link_types[index] == "site"
            self.origin_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            resolved_name = self._name_resolver.resolve(name, obj_type=obj_type, role="Origin link")
            link_id = mujoco.mj_name2id(self.model, obj_type, resolved_name)
            if link_id < 0:
                raise ValueError(f"Origin link '{name}' not found in model")
            self.origin_ids.append(link_id)

        self.task_ids: list[int] = []
        self.task_is_site: list[bool] = []
        for index, name in enumerate(self.task_link_names):
            is_site = config.task_link_types[index] == "site"
            self.task_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            resolved_name = self._name_resolver.resolve(name, obj_type=obj_type, role="Task link")
            link_id = mujoco.mj_name2id(self.model, obj_type, resolved_name)
            if link_id < 0:
                raise ValueError(f"Task link '{name}' not found in model")
            self.task_ids.append(link_id)

        self.landmark_filter = TemporalFilter(alpha=config.preprocess.temporal_filter_alpha)

        self._norm_delta = config.solver.norm_delta
        self._max_iterations = config.solver.max_iterations
        self._output_alpha = config.solver.output_alpha
        self._weights = np.array(config.vector_weights, dtype=np.float64)
        self._vector_loss_type = config.vector_loss.type
        self._vector_huber_delta = config.vector_loss.huber_delta

        self._target_directions: np.ndarray | None = None
        self._target_vectors: np.ndarray | None = None
        self._target_frame_primary_directions: np.ndarray | None = None
        self._target_frame_secondary_directions: np.ndarray | None = None
        self._target_angles: np.ndarray | None = None
        self._last_qpos: np.ndarray | None = None
        self._vector_scale_landmark_idx = config.vector_loss.scale_landmarks[1]
        self._robot_vector_scale = 0.0

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

        self._angle_landmarks: list[tuple[int, int, int]] = []
        self._angle_qpos_ids: list[int] = []
        self._angle_dof_ids: list[int] = []
        self._angle_joint_ranges: list[tuple[float, float]] = []
        self._angle_weights: list[float] = []
        self._angle_scales: list[float] = []
        self._angle_inverts: list[bool] = []
        for constraint in config.angle_constraints:
            self._angle_landmarks.append(tuple(constraint.landmarks))
            resolved_joint = self._name_resolver.resolve(
                constraint.joint, obj_type=mujoco.mjtObj.mjOBJ_JOINT, role="Angle constraint joint"
            )
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

        self._frame_names: list[str] = []
        self._frame_human_indices: list[tuple[int, int, int]] = []
        self._frame_primary_weights: list[float] = []
        self._frame_secondary_weights: list[float] = []
        self._frame_origin_ids: list[int] = []
        self._frame_origin_is_site: list[bool] = []
        self._frame_primary_ids: list[int] = []
        self._frame_primary_is_site: list[bool] = []
        self._frame_secondary_ids: list[int] = []
        self._frame_secondary_is_site: list[bool] = []
        for constraint in config.frame_constraints:
            self._frame_names.append(constraint.name)
            self._frame_human_indices.append(
                (constraint.human_origin, constraint.human_primary, constraint.human_secondary)
            )
            self._frame_primary_weights.append(constraint.primary_weight)
            self._frame_secondary_weights.append(constraint.secondary_weight)
            origin_id, origin_is_site = self._resolve_frame_point(
                constraint.robot_origin,
                constraint.robot_types[0],
                role=f"Frame origin ({constraint.name or 'unnamed'})",
            )
            primary_id, primary_is_site = self._resolve_frame_point(
                constraint.robot_primary,
                constraint.robot_types[1],
                role=f"Frame primary ({constraint.name or 'unnamed'})",
            )
            secondary_id, secondary_is_site = self._resolve_frame_point(
                constraint.robot_secondary,
                constraint.robot_types[2],
                role=f"Frame secondary ({constraint.name or 'unnamed'})",
            )
            self._frame_origin_ids.append(origin_id)
            self._frame_origin_is_site.append(origin_is_site)
            self._frame_primary_ids.append(primary_id)
            self._frame_primary_is_site.append(primary_is_site)
            self._frame_secondary_ids.append(secondary_id)
            self._frame_secondary_is_site.append(secondary_is_site)

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
        self._mimic_by_source_qpos: dict[int, list[dict[str, float | int]]] = {}
        for mimic in self._mimic_joints:
            source_qpos_id = int(mimic["source_qpos_id"])
            self._mimic_by_source_qpos.setdefault(source_qpos_id, []).append(mimic)

        for joint_index in range(self.model.nq):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if joint_name and "thumb" in joint_name and "cmc" in joint_name:
                low, high = self.model.jnt_range[joint_index]
                if low < high:
                    self.data.qpos[joint_index] = (low + high) / 2.0
                break

        self._forward()

    def _resolve_frame_point(self, name: str, point_type: str, *, role: str) -> tuple[int, bool]:
        is_site = point_type == "site"
        obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
        resolved_name = self._name_resolver.resolve(name, obj_type=obj_type, role=role)
        point_id = mujoco.mj_name2id(self.model, obj_type, resolved_name)
        if point_id < 0:
            raise ValueError(f"{role} '{name}' not found in model")
        return point_id, is_site

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
                reduced_grad[reduced_index] += float(mimic["multiplier"]) * grad[int(mimic["qpos_id"])]
        return reduced_grad

    def _forward(self, qpos: np.ndarray | None = None) -> None:
        if qpos is not None:
            self.data.qpos[:] = self._expand_qpos(qpos)
        mujoco.mj_fwdPosition(self.model, self.data)

    def _get_pos(self, index: int, is_site: bool) -> np.ndarray:
        if is_site:
            return self.data.site_xpos[index].copy()
        return self.data.xpos[index].copy()

    def _get_robot_vectors(self) -> np.ndarray:
        vectors = np.empty((len(self.origin_ids), 3))
        for index in range(len(self.origin_ids)):
            origin = self._get_pos(self.origin_ids[index], self.origin_is_site[index])
            task = self._get_pos(self.task_ids[index], self.task_is_site[index])
            vectors[index] = task - origin
        return vectors

    def _get_robot_frame_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        primary_vectors = np.empty((len(self._frame_origin_ids), 3), dtype=np.float64)
        secondary_vectors = np.empty((len(self._frame_origin_ids), 3), dtype=np.float64)
        for index in range(len(self._frame_origin_ids)):
            origin = self._get_pos(self._frame_origin_ids[index], self._frame_origin_is_site[index])
            primary = self._get_pos(self._frame_primary_ids[index], self._frame_primary_is_site[index])
            secondary = self._get_pos(self._frame_secondary_ids[index], self._frame_secondary_is_site[index])
            primary_vectors[index] = primary - origin
            secondary_vectors[index] = secondary - origin
        return primary_vectors, secondary_vectors

    def _accumulate_direction_loss(
        self,
        vector: np.ndarray,
        target_dir: np.ndarray,
        weight: float,
        jac_diff: np.ndarray | None = None,
        grad: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray | None]:
        if weight <= 0.0:
            return 0.0, grad
        robot_norm = np.linalg.norm(vector)
        if robot_norm < 1e-8:
            return weight, grad
        robot_dir = vector / robot_norm
        cos_sim = float(np.dot(robot_dir, target_dir))
        loss = weight * (1.0 - cos_sim)
        if grad is None or jac_diff is None:
            return loss, grad
        grad_vec = -(target_dir - cos_sim * robot_dir) / robot_norm
        grad += weight * (grad_vec @ jac_diff)
        return loss, grad

    def _get_effective_weight(self, index: int) -> float:
        return self._weights[index]

    def _compute_loss(self, qpos: np.ndarray) -> float:
        full_qpos = self._expand_qpos(qpos)
        self._forward(full_qpos)
        robot_vecs = self._get_robot_vectors()
        loss = 0.0
        for index in range(len(robot_vecs)):
            weight = self._get_effective_weight(index)
            if self._vector_loss_type == "residual":
                diff = robot_vecs[index] - self._target_vectors[index]
                dist = float(np.linalg.norm(diff))
                loss += weight * _huber_loss(dist, self._vector_huber_delta)
            else:
                direction_loss, _ = self._accumulate_direction_loss(
                    robot_vecs[index],
                    self._target_directions[index],
                    weight,
                )
                loss += direction_loss
        if self._target_frame_primary_directions is not None:
            primary_vectors, secondary_vectors = self._get_robot_frame_vectors()
            for index in range(len(primary_vectors)):
                primary_loss, _ = self._accumulate_direction_loss(
                    primary_vectors[index],
                    self._target_frame_primary_directions[index],
                    self._frame_primary_weights[index],
                )
                secondary_loss, _ = self._accumulate_direction_loss(
                    secondary_vectors[index],
                    self._target_frame_secondary_directions[index],
                    self._frame_secondary_weights[index],
                )
                loss += primary_loss + secondary_loss
        if self._last_qpos is not None:
            loss += self._norm_delta * np.sum((full_qpos - self._last_qpos) ** 2)
        if self._target_angles is not None:
            for index in range(len(self._angle_qpos_ids)):
                qpos_id = self._angle_qpos_ids[index]
                diff = full_qpos[qpos_id] - self._target_angles[index]
                loss += self._angle_weights[index] * diff * diff
        return loss

    def _compute_loss_and_grad(self, qpos: np.ndarray) -> tuple[float, np.ndarray]:
        full_qpos = self._expand_qpos(qpos)
        self._forward(full_qpos)
        robot_vecs = self._get_robot_vectors()
        num_velocities = self.model.nv
        grad = np.zeros(num_velocities)
        loss = 0.0

        for index in range(len(self.origin_ids)):
            robot_vec = robot_vecs[index]
            weight = self._get_effective_weight(index)
            jac_task = np.zeros((3, num_velocities))
            jac_origin = np.zeros((3, num_velocities))

            if self.task_is_site[index]:
                mujoco.mj_jacSite(self.model, self.data, jac_task, None, self.task_ids[index])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_task, None, self.task_ids[index])

            if self.origin_is_site[index]:
                mujoco.mj_jacSite(self.model, self.data, jac_origin, None, self.origin_ids[index])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_origin, None, self.origin_ids[index])

            jac_diff = jac_task - jac_origin
            if self._vector_loss_type == "residual":
                diff = robot_vec - self._target_vectors[index]
                dist = float(np.linalg.norm(diff))
                loss += weight * _huber_loss(dist, self._vector_huber_delta)
                if dist > 1e-8:
                    grad_coeff = _huber_grad(dist, self._vector_huber_delta) / dist
                    grad += weight * grad_coeff * (diff @ jac_diff)
            else:
                direction_loss, grad = self._accumulate_direction_loss(
                    robot_vec,
                    self._target_directions[index],
                    weight,
                    jac_diff=jac_diff,
                    grad=grad,
                )
                loss += direction_loss

        if self._target_frame_primary_directions is not None:
            for index in range(len(self._frame_origin_ids)):
                jac_origin = np.zeros((3, num_velocities))
                jac_primary = np.zeros((3, num_velocities))
                jac_secondary = np.zeros((3, num_velocities))

                if self._frame_origin_is_site[index]:
                    mujoco.mj_jacSite(self.model, self.data, jac_origin, None, self._frame_origin_ids[index])
                else:
                    mujoco.mj_jacBody(self.model, self.data, jac_origin, None, self._frame_origin_ids[index])

                if self._frame_primary_is_site[index]:
                    mujoco.mj_jacSite(self.model, self.data, jac_primary, None, self._frame_primary_ids[index])
                else:
                    mujoco.mj_jacBody(self.model, self.data, jac_primary, None, self._frame_primary_ids[index])

                if self._frame_secondary_is_site[index]:
                    mujoco.mj_jacSite(self.model, self.data, jac_secondary, None, self._frame_secondary_ids[index])
                else:
                    mujoco.mj_jacBody(self.model, self.data, jac_secondary, None, self._frame_secondary_ids[index])

                origin = self._get_pos(self._frame_origin_ids[index], self._frame_origin_is_site[index])
                primary = self._get_pos(self._frame_primary_ids[index], self._frame_primary_is_site[index])
                secondary = self._get_pos(self._frame_secondary_ids[index], self._frame_secondary_is_site[index])
                primary_vec = primary - origin
                secondary_vec = secondary - origin
                primary_loss, grad = self._accumulate_direction_loss(
                    primary_vec,
                    self._target_frame_primary_directions[index],
                    self._frame_primary_weights[index],
                    jac_diff=jac_primary - jac_origin,
                    grad=grad,
                )
                secondary_loss, grad = self._accumulate_direction_loss(
                    secondary_vec,
                    self._target_frame_secondary_directions[index],
                    self._frame_secondary_weights[index],
                    jac_diff=jac_secondary - jac_origin,
                    grad=grad,
                )
                loss += primary_loss + secondary_loss

        if self._last_qpos is not None:
            delta_q = full_qpos - self._last_qpos
            loss += self._norm_delta * np.sum(delta_q**2)
            grad += 2.0 * self._norm_delta * delta_q

        if self._target_angles is not None:
            for index in range(len(self._angle_qpos_ids)):
                qpos_id = self._angle_qpos_ids[index]
                dof_id = self._angle_dof_ids[index]
                target = self._target_angles[index]
                weight = self._angle_weights[index]
                diff = full_qpos[qpos_id] - target
                loss += weight * diff * diff
                grad[dof_id] += 2.0 * weight * diff

        return loss, self._reduce_grad(grad)

    def update_targets(
        self,
        landmarks_3d: np.ndarray,
        hand_side: str = "right",
    ) -> None:
        landmarks = preprocess_landmarks(
            landmarks_3d,
            hand_side=hand_side,
        )
        landmarks = self.landmark_filter.filter(landmarks)

        directions = np.empty((len(self.human_vector_pairs), 3), dtype=np.float64)
        target_vectors = np.empty((len(self.human_vector_pairs), 3), dtype=np.float64)
        vector_scale = self._robot_vector_scale / max(
            float(np.linalg.norm(landmarks[self._vector_scale_landmark_idx])),
            1e-6,
        )
        for index, (origin_idx, target_idx) in enumerate(self.human_vector_pairs):
            vector = landmarks[target_idx] - landmarks[origin_idx]
            norm = np.linalg.norm(vector)
            target_vectors[index] = vector_scale * vector
            if norm < 1e-8:
                directions[index] = 0.0
            else:
                directions[index] = vector / norm
        self._target_directions = directions
        self._target_vectors = target_vectors
        if self._frame_human_indices:
            frame_primary = np.empty((len(self._frame_human_indices), 3), dtype=np.float64)
            frame_secondary = np.empty((len(self._frame_human_indices), 3), dtype=np.float64)
            for index, (origin_idx, primary_idx, secondary_idx) in enumerate(self._frame_human_indices):
                primary_vector = landmarks[primary_idx] - landmarks[origin_idx]
                secondary_vector = landmarks[secondary_idx] - landmarks[origin_idx]
                primary_norm = np.linalg.norm(primary_vector)
                secondary_norm = np.linalg.norm(secondary_vector)
                frame_primary[index] = 0.0 if primary_norm < 1e-8 else primary_vector / primary_norm
                frame_secondary[index] = 0.0 if secondary_norm < 1e-8 else secondary_vector / secondary_norm
            self._target_frame_primary_directions = frame_primary
            self._target_frame_secondary_directions = frame_secondary
        else:
            self._target_frame_primary_directions = None
            self._target_frame_secondary_directions = None

        if self._angle_landmarks:
            target_angles = np.zeros(len(self._angle_landmarks))
            for index, (a, b, c) in enumerate(self._angle_landmarks):
                v_ba = landmarks[a] - landmarks[b]
                v_bc = landmarks[c] - landmarks[b]
                norm_ba = np.linalg.norm(v_ba)
                norm_bc = np.linalg.norm(v_bc)
                if norm_ba < 1e-8 or norm_bc < 1e-8:
                    flexion = 0.0
                else:
                    cos_angle = np.clip(np.dot(v_ba, v_bc) / (norm_ba * norm_bc), -1.0, 1.0)
                    flexion = np.pi - np.arccos(cos_angle)
                low, high = self._angle_joint_ranges[index]
                normalized = flexion / np.pi
                if self._angle_inverts[index]:
                    normalized = 1.0 - normalized
                normalized = np.clip(normalized * self._angle_scales[index], 0.0, 1.0)
                target_angles[index] = low + normalized * (high - low)
            self._target_angles = target_angles

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
