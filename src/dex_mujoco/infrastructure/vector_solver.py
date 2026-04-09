"""MuJoCo/SciPy-backed vector retargeting solver."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize

from dex_mujoco.domain import RetargetingConfig, preprocess_landmarks

from .hand_model import HandModel
from .model_name_resolver import ModelNameResolver

_THUMB_TIP_IDX = 4
_FINGER_TIP_INDICES = [8, 12, 16, 20]
_THUMB_LANDMARKS = {0, 1, 2, 3, 4}


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
        self._target_angles: np.ndarray | None = None
        self._last_qpos: np.ndarray | None = None

        self._pinch_enabled = config.pinch.enabled
        self._pinch_alphas = np.zeros(4)
        self._pinch_alpha_thumb = 0.0
        self._pinch_d1 = config.pinch.d1
        self._pinch_d2 = config.pinch.d2
        self._pinch_weight = config.pinch.weight
        self._thumb_weight_boost = config.pinch.thumb_weight_boost
        self._thumb_vector_indices: set[int] = set()
        self._thumb_site_id = -1
        self._finger_site_ids: list[int] = []

        if self._pinch_enabled:
            for index, (origin_idx, target_idx) in enumerate(self.human_vector_pairs):
                if origin_idx in _THUMB_LANDMARKS or target_idx in _THUMB_LANDMARKS:
                    self._thumb_vector_indices.add(index)

            sites = config.pinch.fingertip_sites
            if len(sites) >= 5:
                thumb_site = self._name_resolver.resolve(
                    sites[0], obj_type=mujoco.mjtObj.mjOBJ_SITE, role="Thumb site"
                )
                self._thumb_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, thumb_site)
                for name in sites[1:5]:
                    resolved_name = self._name_resolver.resolve(
                        name, obj_type=mujoco.mjtObj.mjOBJ_SITE, role="Finger site"
                    )
                    site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, resolved_name)
                    self._finger_site_ids.append(site_id)
                if self._thumb_site_id < 0:
                    raise ValueError(f"Thumb site '{sites[0]}' not found in model")
                for index, site_id in enumerate(self._finger_site_ids):
                    if site_id < 0:
                        raise ValueError(f"Finger site '{sites[index + 1]}' not found in model")

        self._pos_enabled = config.position.enabled
        self._pos_weight = config.position.weight
        self._pos_landmark_indices: list[int] = []
        self._pos_body_ids: list[int] = []
        self._pos_body_is_site: list[bool] = []
        self._pos_per_weights: list[float] = []
        self._position_targets: np.ndarray | None = None
        self._robot_palm_size = 0.0
        self._scale_landmark_idx = 0
        self._wrist_body_id = 0
        self._wrist_is_site = False
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

        if self._pos_enabled:
            position_config = config.position
            self._scale_landmark_idx = position_config.scale_landmarks[1]
            scale_ids: list[tuple[int, bool]] = []
            for index, name in enumerate(position_config.scale_bodies):
                is_site = position_config.scale_body_types[index] == "site"
                object_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
                resolved_name = self._name_resolver.resolve(name, obj_type=object_type, role="Scale body")
                body_id = mujoco.mj_name2id(self.model, object_type, resolved_name)
                if body_id < 0:
                    raise ValueError(f"Scale body '{name}' not found")
                scale_ids.append((body_id, is_site))

            self._wrist_body_id = scale_ids[0][0]
            self._wrist_is_site = scale_ids[0][1]

            self._forward()
            p0 = self._get_pos(scale_ids[0][0], scale_ids[0][1])
            p1 = self._get_pos(scale_ids[1][0], scale_ids[1][1])
            self._robot_palm_size = float(np.linalg.norm(p1 - p0))

            for constraint in position_config.constraints:
                self._pos_landmark_indices.append(constraint.landmark)
                is_site = constraint.body_type == "site"
                self._pos_body_is_site.append(is_site)
                object_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
                resolved_name = self._name_resolver.resolve(
                    constraint.body, obj_type=object_type, role="Position constraint body"
                )
                body_id = mujoco.mj_name2id(self.model, object_type, resolved_name)
                if body_id < 0:
                    raise ValueError(f"Position constraint body '{constraint.body}' not found")
                self._pos_body_ids.append(body_id)
                self._pos_per_weights.append(constraint.weight)

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

        self._bounds: list[tuple[float | None, float | None]] = []
        for joint_index in range(self.model.nq):
            low, high = self.model.jnt_range[joint_index]
            if low < high:
                self._bounds.append((float(low), float(high)))
            else:
                self._bounds.append((None, None))

        for joint_index in range(self.model.nq):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if joint_name and "thumb" in joint_name and "cmc" in joint_name:
                low, high = self.model.jnt_range[joint_index]
                if low < high:
                    self.data.qpos[joint_index] = (low + high) / 2.0
                break

        self._forward()

    def _forward(self, qpos: np.ndarray | None = None) -> None:
        if qpos is not None:
            self.data.qpos[:] = qpos
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

    def _get_effective_weight(self, index: int) -> float:
        weight = self._weights[index]
        if self._pinch_enabled and self._pinch_alpha_thumb > 0 and index in self._thumb_vector_indices:
            weight *= 1.0 + self._pinch_alpha_thumb * self._thumb_weight_boost
        return weight

    def _compute_pinch_loss(self) -> float:
        if not self._pinch_enabled or self._pinch_alpha_thumb < 1e-6:
            return 0.0
        thumb_pos = self.data.site_xpos[self._thumb_site_id]
        loss = 0.0
        for index in range(4):
            alpha = self._pinch_alphas[index]
            if alpha < 1e-6:
                continue
            finger_pos = self.data.site_xpos[self._finger_site_ids[index]]
            diff = thumb_pos - finger_pos
            loss += alpha * self._pinch_weight * np.dot(diff, diff)
        return loss

    def _compute_position_loss(self) -> float:
        if not self._pos_enabled or self._position_targets is None:
            return 0.0
        wrist_pos = self._get_pos(self._wrist_body_id, self._wrist_is_site)
        loss = 0.0
        for index in range(len(self._pos_body_ids)):
            body_pos = self._get_pos(self._pos_body_ids[index], self._pos_body_is_site[index])
            robot_rel = body_pos - wrist_pos
            diff = robot_rel - self._position_targets[index]
            weight = self._pos_weight * self._pos_per_weights[index]
            loss += weight * np.dot(diff, diff)
        return loss

    def _compute_loss(self, qpos: np.ndarray) -> float:
        self._forward(qpos)
        robot_vecs = self._get_robot_vectors()
        loss = 0.0
        for index in range(len(robot_vecs)):
            weight = self._get_effective_weight(index)
            if self._vector_loss_type == "residual":
                diff = robot_vecs[index] - self._target_vectors[index]
                dist = float(np.linalg.norm(diff))
                loss += weight * _huber_loss(dist, self._vector_huber_delta)
            else:
                robot_norm = np.linalg.norm(robot_vecs[index])
                if robot_norm < 1e-8:
                    loss += weight
                    continue
                cos_sim = np.dot(robot_vecs[index] / robot_norm, self._target_directions[index])
                loss += weight * (1.0 - cos_sim)
        if self._last_qpos is not None:
            loss += self._norm_delta * np.sum((qpos - self._last_qpos) ** 2)
        if self._target_angles is not None:
            for index in range(len(self._angle_qpos_ids)):
                qpos_id = self._angle_qpos_ids[index]
                diff = qpos[qpos_id] - self._target_angles[index]
                loss += self._angle_weights[index] * diff * diff
        loss += self._compute_pinch_loss()
        loss += self._compute_position_loss()
        return loss

    def _compute_loss_and_grad(self, qpos: np.ndarray) -> tuple[float, np.ndarray]:
        self._forward(qpos)
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
                robot_norm = np.linalg.norm(robot_vec)
                if robot_norm < 1e-8:
                    loss += weight
                    continue

                robot_dir = robot_vec / robot_norm
                target_dir = self._target_directions[index]
                cos_sim = np.dot(robot_dir, target_dir)
                loss += weight * (1.0 - cos_sim)

                grad_vec = -(target_dir - cos_sim * robot_dir) / robot_norm
                grad += weight * (grad_vec @ jac_diff)

        if self._last_qpos is not None:
            delta_q = qpos - self._last_qpos
            loss += self._norm_delta * np.sum(delta_q**2)
            grad += 2.0 * self._norm_delta * delta_q

        if self._target_angles is not None:
            for index in range(len(self._angle_qpos_ids)):
                qpos_id = self._angle_qpos_ids[index]
                dof_id = self._angle_dof_ids[index]
                target = self._target_angles[index]
                weight = self._angle_weights[index]
                diff = qpos[qpos_id] - target
                loss += weight * diff * diff
                grad[dof_id] += 2.0 * weight * diff

        if self._pinch_enabled and self._pinch_alpha_thumb > 1e-6:
            thumb_pos = self.data.site_xpos[self._thumb_site_id]
            jac_thumb = np.zeros((3, num_velocities))
            mujoco.mj_jacSite(self.model, self.data, jac_thumb, None, self._thumb_site_id)
            for index in range(4):
                alpha = self._pinch_alphas[index]
                if alpha < 1e-6:
                    continue
                finger_pos = self.data.site_xpos[self._finger_site_ids[index]]
                diff = thumb_pos - finger_pos
                coeff = alpha * self._pinch_weight
                loss += coeff * np.dot(diff, diff)
                jac_finger = np.zeros((3, num_velocities))
                mujoco.mj_jacSite(self.model, self.data, jac_finger, None, self._finger_site_ids[index])
                grad += 2.0 * coeff * (diff @ (jac_thumb - jac_finger))

        if self._pos_enabled and self._position_targets is not None:
            wrist_pos = self._get_pos(self._wrist_body_id, self._wrist_is_site)
            jac_wrist = np.zeros((3, num_velocities))
            if self._wrist_is_site:
                mujoco.mj_jacSite(self.model, self.data, jac_wrist, None, self._wrist_body_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_wrist, None, self._wrist_body_id)
            for index in range(len(self._pos_body_ids)):
                body_pos = self._get_pos(self._pos_body_ids[index], self._pos_body_is_site[index])
                robot_rel = body_pos - wrist_pos
                diff = robot_rel - self._position_targets[index]
                weight = self._pos_weight * self._pos_per_weights[index]
                loss += weight * np.dot(diff, diff)
                jac_body = np.zeros((3, num_velocities))
                if self._pos_body_is_site[index]:
                    mujoco.mj_jacSite(self.model, self.data, jac_body, None, self._pos_body_ids[index])
                else:
                    mujoco.mj_jacBody(self.model, self.data, jac_body, None, self._pos_body_ids[index])
                grad += 2.0 * weight * (diff @ (jac_body - jac_wrist))

        return loss, grad

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

        if self._pos_enabled:
            human_palm_size = np.linalg.norm(landmarks[self._scale_landmark_idx])
            scale = self._robot_palm_size / max(human_palm_size, 1e-6)
            targets = np.empty((len(self._pos_landmark_indices), 3), dtype=np.float64)
            for index, landmark_index in enumerate(self._pos_landmark_indices):
                targets[index] = scale * landmarks[landmark_index]
            self._position_targets = targets

        if self._pinch_enabled:
            thumb_pos = landmarks[_THUMB_TIP_IDX]
            for index, tip_index in enumerate(_FINGER_TIP_INDICES):
                distance = np.linalg.norm(landmarks[tip_index] - thumb_pos)
                self._pinch_alphas[index] = np.clip(
                    (self._pinch_d2 - distance) / (self._pinch_d2 - self._pinch_d1 + 1e-8),
                    0.0,
                    1.0,
                )
            self._pinch_alpha_thumb = float(np.max(self._pinch_alphas))

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
            return self.data.qpos.copy()

        x0 = self.data.qpos.copy()
        previous_qpos = None if self._last_qpos is None else self._last_qpos.copy()

        result = minimize(
            fun=self._compute_loss_and_grad,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=self._bounds,
            options={
                "maxiter": self._max_iterations,
                "ftol": 1e-6,
            },
        )

        qpos = result.x.copy()
        if previous_qpos is not None and self._output_alpha < 1.0:
            qpos = previous_qpos + self._output_alpha * (qpos - previous_qpos)

        self._last_qpos = qpos.copy()
        self._forward(qpos)
        return qpos

    def compute_error(self) -> float:
        self._forward()
        if self._target_directions is None:
            return 0.0
        return self._compute_loss(self.data.qpos.copy())

    def get_target_directions(self) -> np.ndarray | None:
        if self._target_directions is None:
            return None
        return self._target_directions.copy()
