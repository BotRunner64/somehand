"""MuJoCo hand-model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import mink
import mujoco
import numpy as np


def _joint_type_value(value) -> int:
    return int(value.value) if hasattr(value, "value") else int(value)


def mimic_polycoef(mimic: Mapping[str, object]) -> list[float]:
    polycoef = mimic.get("polycoef")
    if polycoef is not None:
        return [float(value) for value in polycoef]
    return [
        float(mimic.get("offset", 0.0)),
        float(mimic.get("multiplier", 1.0)),
    ]


def evaluate_mimic_joint(mimic: Mapping[str, object], source_value: float) -> float:
    polycoef = mimic_polycoef(mimic)
    return sum(coefficient * (source_value ** power) for power, coefficient in enumerate(polycoef))


def mimic_joint_derivative(mimic: Mapping[str, object], source_value: float) -> float:
    polycoef = mimic_polycoef(mimic)
    return sum(
        power * coefficient * (source_value ** (power - 1))
        for power, coefficient in enumerate(polycoef[1:], start=1)
    )


class HandModel:
    """Wraps a MuJoCo hand model and provides kinematic queries via Mink."""

    def __init__(self, mjcf_path: str):
        self.mjcf_path = str(Path(mjcf_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        self.mimic_joints = self._collect_mimic_joints()
        self.apply_mimic_constraints(self.data.qpos)

    def _collect_mimic_joints(self) -> list[dict[str, object]]:
        mimic_joints: list[dict[str, object]] = []
        joint_eq_type = _joint_type_value(mujoco.mjtEq.mjEQ_JOINT)
        for equality_index in range(self.model.neq):
            if int(self.model.eq_type[equality_index]) != joint_eq_type:
                continue
            mimic_joint_id = int(self.model.eq_obj1id[equality_index])
            source_joint_id = int(self.model.eq_obj2id[equality_index])
            if mimic_joint_id < 0 or source_joint_id < 0:
                continue
            mimic_qpos_id = int(self.model.jnt_qposadr[mimic_joint_id])
            source_qpos_id = int(self.model.jnt_qposadr[source_joint_id])
            mimic_dof_id = int(self.model.jnt_dofadr[mimic_joint_id])
            source_dof_id = int(self.model.jnt_dofadr[source_joint_id])
            coefficients = self.model.eq_data[equality_index]
            mimic_joints.append(
                {
                    "joint_id": mimic_joint_id,
                    "source_joint_id": source_joint_id,
                    "qpos_id": mimic_qpos_id,
                    "source_qpos_id": source_qpos_id,
                    "dof_id": mimic_dof_id,
                    "source_dof_id": source_dof_id,
                    "polycoef": [float(value) for value in coefficients[:5]],
                }
            )
        return mimic_joints

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    def get_body_position(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_site_position(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()

    def get_joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
        ]

    def get_body_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(1, self.model.nbody)
        ]

    def get_site_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            for site_id in range(self.model.nsite)
        ]

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_joint_name_to_qpos_index(self) -> dict[str, int]:
        return {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): int(self.model.jnt_qposadr[joint_id])
            for joint_id in range(self.model.njnt)
        }

    def get_actuator_qpos_indices(self) -> np.ndarray:
        indices: list[int] = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int32)

    def apply_mimic_constraints(self, qpos: np.ndarray) -> np.ndarray:
        for mimic in self.mimic_joints:
            qpos_id = int(mimic["qpos_id"])
            source_qpos_id = int(mimic["source_qpos_id"])
            source_value = float(qpos[source_qpos_id])
            qpos[qpos_id] = evaluate_mimic_joint(mimic, source_value)
        return qpos

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.apply_mimic_constraints(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.apply_mimic_constraints(self.data.qpos)
        self.configuration.update()
