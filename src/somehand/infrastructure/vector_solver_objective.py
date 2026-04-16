"""Loss and gradient helpers for vector retargeting."""

from __future__ import annotations

import mujoco
import numpy as np

from .vector_solver_primitives import huber_grad, huber_loss


def rotation_jacobian_to_axis_jacobian(jac_rot: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.cross(jac_rot.T, axis).T


def accumulate_direction_loss(
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


def compute_loss(retargeter, qpos: np.ndarray) -> float:
    full_qpos = retargeter._expand_qpos(qpos)
    retargeter._forward(full_qpos)
    robot_vecs = retargeter._get_robot_vectors()
    loss = 0.0
    for index in range(len(robot_vecs)):
        weight = retargeter._get_effective_weight(index)
        if retargeter._get_loss_type(index) == "residual":
            diff = robot_vecs[index] - retargeter._target_vectors[index]
            dist = float(np.linalg.norm(diff))
            loss += weight * huber_loss(dist, retargeter._vector_huber_delta)
        else:
            direction_loss, _ = accumulate_direction_loss(
                robot_vecs[index],
                retargeter._target_directions[index],
                weight,
            )
            loss += direction_loss
    if retargeter._target_frame_primary_directions is not None:
        primary_axes, secondary_axes = retargeter._get_robot_frame_axes()
        for index in range(len(primary_axes)):
            primary_loss, _ = accumulate_direction_loss(
                primary_axes[index],
                retargeter._target_frame_primary_directions[index],
                retargeter._frame_primary_weights[index],
            )
            secondary_loss, _ = accumulate_direction_loss(
                secondary_axes[index],
                retargeter._target_frame_secondary_directions[index],
                retargeter._frame_secondary_weights[index],
            )
            loss += primary_loss + secondary_loss
    if retargeter._last_qpos is not None:
        loss += retargeter._norm_delta * np.sum((full_qpos - retargeter._last_qpos) ** 2)
    if retargeter._target_angles is not None:
        for index in range(len(retargeter._angle_qpos_ids)):
            qpos_id = retargeter._angle_qpos_ids[index]
            diff = full_qpos[qpos_id] - retargeter._target_angles[index]
            loss += retargeter._angle_weights[index] * diff * diff
    if retargeter._target_distances is not None:
        for index in range(len(retargeter._dist_site_ids)):
            activation = retargeter._smoothed_activations[index]
            if activation < 1e-4:
                continue
            id_a, is_site_a, id_b, is_site_b = retargeter._dist_site_ids[index]
            pos_a = retargeter._get_pos(id_a, is_site_a)
            pos_b = retargeter._get_pos(id_b, is_site_b)
            robot_dist = float(np.linalg.norm(pos_b - pos_a))
            diff = robot_dist - retargeter._target_distances[index]
            if diff > 0.0:
                loss += retargeter._dist_weights[index] * activation * diff * diff
    return loss


def compute_loss_and_grad(retargeter, qpos: np.ndarray) -> tuple[float, np.ndarray]:
    full_qpos = retargeter._expand_qpos(qpos)
    retargeter._forward(full_qpos)
    robot_vecs = retargeter._get_robot_vectors()
    num_velocities = retargeter.model.nv
    grad = np.zeros(num_velocities)
    loss = 0.0

    for index in range(len(retargeter.origin_ids)):
        robot_vec = robot_vecs[index]
        weight = retargeter._get_effective_weight(index)
        jac_task = np.zeros((3, num_velocities))
        jac_origin = np.zeros((3, num_velocities))

        if retargeter.task_is_site[index]:
            mujoco.mj_jacSite(retargeter.model, retargeter.data, jac_task, None, retargeter.task_ids[index])
        else:
            mujoco.mj_jacBody(retargeter.model, retargeter.data, jac_task, None, retargeter.task_ids[index])

        if retargeter.origin_is_site[index]:
            mujoco.mj_jacSite(retargeter.model, retargeter.data, jac_origin, None, retargeter.origin_ids[index])
        else:
            mujoco.mj_jacBody(retargeter.model, retargeter.data, jac_origin, None, retargeter.origin_ids[index])

        jac_diff = jac_task - jac_origin
        if retargeter._get_loss_type(index) == "residual":
            diff = robot_vec - retargeter._target_vectors[index]
            dist = float(np.linalg.norm(diff))
            loss += weight * huber_loss(dist, retargeter._vector_huber_delta)
            if dist > 1e-8:
                grad_coeff = huber_grad(dist, retargeter._vector_huber_delta) / dist
                grad += weight * grad_coeff * (diff @ jac_diff)
        else:
            direction_loss, grad = accumulate_direction_loss(
                robot_vec,
                retargeter._target_directions[index],
                weight,
                jac_diff=jac_diff,
                grad=grad,
            )
            loss += direction_loss

    if retargeter._target_frame_primary_directions is not None:
        for index in range(len(retargeter._frame_origin_ids)):
            jac_origin_rot = np.zeros((3, num_velocities))

            if retargeter._frame_origin_is_site[index]:
                mujoco.mj_jacSite(retargeter.model, retargeter.data, None, jac_origin_rot, retargeter._frame_origin_ids[index])
            else:
                mujoco.mj_jacBody(retargeter.model, retargeter.data, None, jac_origin_rot, retargeter._frame_origin_ids[index])

            origin_rotation = retargeter._get_rot(retargeter._frame_origin_ids[index], retargeter._frame_origin_is_site[index])
            primary_axis = origin_rotation @ retargeter._frame_local_primary_axes[index]
            secondary_axis = origin_rotation @ retargeter._frame_local_secondary_axes[index]
            primary_jac = rotation_jacobian_to_axis_jacobian(jac_origin_rot, primary_axis)
            secondary_jac = rotation_jacobian_to_axis_jacobian(jac_origin_rot, secondary_axis)
            primary_loss, grad = accumulate_direction_loss(
                primary_axis,
                retargeter._target_frame_primary_directions[index],
                retargeter._frame_primary_weights[index],
                jac_diff=primary_jac,
                grad=grad,
            )
            secondary_loss, grad = accumulate_direction_loss(
                secondary_axis,
                retargeter._target_frame_secondary_directions[index],
                retargeter._frame_secondary_weights[index],
                jac_diff=secondary_jac,
                grad=grad,
            )
            loss += primary_loss + secondary_loss

    if retargeter._last_qpos is not None:
        delta_q = full_qpos - retargeter._last_qpos
        loss += retargeter._norm_delta * np.sum(delta_q**2)
        grad += 2.0 * retargeter._norm_delta * delta_q

    if retargeter._target_angles is not None:
        for index in range(len(retargeter._angle_qpos_ids)):
            qpos_id = retargeter._angle_qpos_ids[index]
            dof_id = retargeter._angle_dof_ids[index]
            target = retargeter._target_angles[index]
            weight = retargeter._angle_weights[index]
            diff = full_qpos[qpos_id] - target
            loss += weight * diff * diff
            grad[dof_id] += 2.0 * weight * diff

    if retargeter._target_distances is not None:
        for index in range(len(retargeter._dist_site_ids)):
            activation = retargeter._smoothed_activations[index]
            if activation < 1e-4:
                continue
            id_a, is_site_a, id_b, is_site_b = retargeter._dist_site_ids[index]
            pos_a = retargeter._get_pos(id_a, is_site_a)
            pos_b = retargeter._get_pos(id_b, is_site_b)
            vec_ab = pos_b - pos_a
            robot_dist = float(np.linalg.norm(vec_ab))
            if robot_dist < 1e-8:
                continue
            diff = robot_dist - retargeter._target_distances[index]
            if diff <= 0.0:
                continue
            weight = retargeter._dist_weights[index] * activation
            loss += weight * diff * diff
            jac_a = np.zeros((3, num_velocities))
            jac_b = np.zeros((3, num_velocities))
            if is_site_a:
                mujoco.mj_jacSite(retargeter.model, retargeter.data, jac_a, None, id_a)
            else:
                mujoco.mj_jacBody(retargeter.model, retargeter.data, jac_a, None, id_a)
            if is_site_b:
                mujoco.mj_jacSite(retargeter.model, retargeter.data, jac_b, None, id_b)
            else:
                mujoco.mj_jacBody(retargeter.model, retargeter.data, jac_b, None, id_b)
            direction = vec_ab / robot_dist
            grad += 2.0 * weight * diff * (direction @ (jac_b - jac_a))

    return loss, retargeter._reduce_grad(grad)
