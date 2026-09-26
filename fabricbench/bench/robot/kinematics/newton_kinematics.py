"""Newton-backed robot kinematics.

This module provides a ``BaseKinematics`` implementation backed by the Newton
model that is already used for simulation.  The class is parameterized by joint
indices and end-effector frames, so it can be used by teleop, planners, and task
code without depending directly on ``DeformableEnv``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from fabricbench.bench.robot.kinematics.base_kinematics import BaseKinematics, ConstraintType


@dataclass(frozen=True)
class NewtonKinematicsFrame:
    """A named TCP frame represented by a Newton body plus a local offset."""

    body: int | str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


class NewtonKinematics(BaseKinematics):
    """FK/IK adapter around ``newton.ik.IKSolver`` for one robot chain."""

    def __init__(
        self,
        model: newton.Model,
        *,
        arm_joint_indices: Sequence[int],
        ee_body: int | str,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        full_q_getter: Callable[[], np.ndarray] | None = None,
        frame_map: Mapping[str, NewtonKinematicsFrame | tuple[int | str, Sequence[float]]] | None = None,
        base_link: str = "",
        ee_link: str = "tcp_link",
        urdf_path: str = "",
        iterations: int = 24,
        position_tolerance: float = 2.5e-3,
        rotation_tolerance: float = math.radians(8.0),
        joint_limit_weight: float = 10.0,
        lambda_initial: float = 0.1,
        ground_check_enabled: bool = False,
        ground_z_threshold: float = 0.0,
        ground_check_body_indices: Sequence[int | str] | None = None,
    ) -> None:
        super().__init__(urdf_path=urdf_path, base_link=base_link, ee_link=ee_link)

        self.model = model
        self.arm_indices = np.asarray(arm_joint_indices, dtype=np.int32)
        if self.arm_indices.ndim != 1 or len(self.arm_indices) == 0:
            raise ValueError("arm_joint_indices must be a non-empty 1D sequence")

        self.full_q_getter = full_q_getter
        self.iterations = int(iterations)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.ground_check_enabled = bool(ground_check_enabled)
        self.ground_z_threshold = float(ground_z_threshold)

        self.default_frame = NewtonKinematicsFrame(self._resolve_body(ee_body), self._vec3_tuple(ee_offset))
        self.frame_map: dict[str, NewtonKinematicsFrame] = {
            ee_link: self.default_frame,
            "": self.default_frame,
        }
        if frame_map:
            for name, frame in frame_map.items():
                self.frame_map[str(name)] = self._normalize_frame(frame)

        self.num_joints = len(self.arm_indices)
        self.joint_limits = self._joint_limits_for_arm()
        self._ground_check_body_indices = self._resolve_ground_check_bodies(ground_check_body_indices)

        self._fk_state = self.model.state()
        self._fk_q = wp.array(self._current_full_q().astype(np.float32), dtype=wp.float32, device=self.model.device)
        self._fk_qd = wp.zeros_like(self.model.joint_qd)

        initial_pose = self.solve_fk(self._current_arm_q(), ee_link=ee_link)
        target_pos = wp.vec3(float(initial_pose[0]), float(initial_pose[1]), float(initial_pose[2]))
        target_quat_xyzw = wp.vec4(
            float(initial_pose[4]),
            float(initial_pose[5]),
            float(initial_pose[6]),
            float(initial_pose[3]),
        )

        self.pos_obj = ik.IKObjectivePosition(
            link_index=int(self.default_frame.body),
            link_offset=wp.vec3(*self.default_frame.offset),
            target_positions=wp.array([target_pos], dtype=wp.vec3, device=self.model.device),
            weight=1.0,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=int(self.default_frame.body),
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([target_quat_xyzw], dtype=wp.vec4, device=self.model.device),
            weight=1.0,
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=float(joint_limit_weight),
        )
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj],
            lambda_initial=float(lambda_initial),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._ik_q = wp.array(
            self._current_full_q()[None, :].astype(np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )

    @classmethod
    def from_deformable_env(cls, env, arm: str, **kwargs) -> "NewtonKinematics":
        """Build a Newton kinematics backend from ``DeformableEnv`` metadata."""

        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if not getattr(env, "has_robot", False):
            raise ValueError("NewtonKinematics.from_deformable_env requires a robot scene")

        arm_indices = env.idx_l_arm if arm == "left" else env.idx_r_arm
        ee_body = env.eef_l_body if arm == "left" else env.eef_r_body
        ee_offset = (0.0, 0.0, float(env.robot.eef_offset_z))

        def full_q_getter() -> np.ndarray:
            if hasattr(env, "_actual_q"):
                return np.asarray(env._actual_q(), dtype=float).copy()
            if getattr(env, "state_0", None) is not None and env.state_0.joint_q is not None:
                return np.asarray(env.state_0.joint_q.numpy(), dtype=float).copy()
            return np.asarray(env.model.joint_q.numpy(), dtype=float).copy()

        frame_name = kwargs.pop("ee_link", f"{arm}_tcp")
        frame_map = dict(kwargs.pop("frame_map", {}) or {})
        frame_map.setdefault(frame_name, NewtonKinematicsFrame(int(ee_body), ee_offset))
        frame_map.setdefault("tcp_link", NewtonKinematicsFrame(int(ee_body), ee_offset))

        return cls(
            env.model,
            arm_joint_indices=arm_indices,
            ee_body=int(ee_body),
            ee_offset=ee_offset,
            full_q_getter=full_q_getter,
            frame_map=frame_map,
            ee_link=frame_name,
            **kwargs,
        )

    @staticmethod
    def _vec3_tuple(value: Sequence[float]) -> tuple[float, float, float]:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"expected a 3D offset, got shape {arr.shape}")
        return float(arr[0]), float(arr[1]), float(arr[2])

    @staticmethod
    def _quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y, z, w = q_xyzw
        u = np.array([x, y, z], dtype=float)
        return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)

    @staticmethod
    def _quat_angle(q_a_xyzw: np.ndarray, q_b_xyzw: np.ndarray) -> float:
        qa = q_a_xyzw / max(np.linalg.norm(q_a_xyzw), 1.0e-12)
        qb = q_b_xyzw / max(np.linalg.norm(q_b_xyzw), 1.0e-12)
        dot = abs(float(np.dot(qa, qb)))
        dot = min(1.0, max(-1.0, dot))
        return 2.0 * math.acos(dot)

    @staticmethod
    def _constraint_enables_rotation(pose_constraint: Optional[ConstraintType]) -> bool:
        if pose_constraint is None:
            return True
        if isinstance(pose_constraint, dict):
            values: list[float] = []
            for item in pose_constraint.values():
                arr = np.asarray(item, dtype=float).reshape(-1)
                if arr.size >= 6:
                    values.extend(arr[3:6].tolist())
            return any(abs(v) > 0.0 for v in values)
        arr = np.asarray(pose_constraint, dtype=float).reshape(-1)
        if arr.size < 6:
            return True
        return bool(np.any(np.abs(arr[3:6]) > 0.0))

    def _resolve_body(self, body: int | str) -> int:
        if isinstance(body, int):
            if body < 0 or body >= int(self.model.body_count):
                raise IndexError(f"body index {body} out of range")
            return int(body)

        labels = getattr(self.model, "body_label", None)
        if labels is None:
            raise KeyError(f"model does not expose body labels, cannot resolve {body!r}")
        for index, label in enumerate(labels):
            if str(label) == str(body):
                return index
        raise KeyError(f"Newton body label {body!r} not found")

    def _normalize_frame(
        self,
        frame: NewtonKinematicsFrame | tuple[int | str, Sequence[float]],
    ) -> NewtonKinematicsFrame:
        if isinstance(frame, NewtonKinematicsFrame):
            return NewtonKinematicsFrame(self._resolve_body(frame.body), self._vec3_tuple(frame.offset))
        body, offset = frame
        return NewtonKinematicsFrame(self._resolve_body(body), self._vec3_tuple(offset))

    def _frame(self, ee_link: str | None) -> NewtonKinematicsFrame:
        key = self.ee_link if ee_link is None else str(ee_link)
        if key not in self.frame_map:
            raise KeyError(f"Newton kinematics frame {key!r} is not configured")
        return self.frame_map[key]

    def _current_full_q(self) -> np.ndarray:
        if self.full_q_getter is not None:
            q = self.full_q_getter()
        else:
            q = self.model.joint_q.numpy()
        return np.asarray(q, dtype=float).copy()

    def _current_arm_q(self) -> np.ndarray:
        return self._current_full_q()[self.arm_indices].copy()

    def _full_q_with_arm(self, q_arm, *, seed_full_q: Sequence[float] | None = None) -> np.ndarray:
        q = self._current_full_q() if seed_full_q is None else np.asarray(seed_full_q, dtype=float).copy()
        q[self.arm_indices] = np.asarray(q_arm, dtype=float)[: len(self.arm_indices)]
        return q

    def _joint_limits_for_arm(self) -> np.ndarray:
        lower = self.model.joint_limit_lower.numpy()[self.arm_indices]
        upper = self.model.joint_limit_upper.numpy()[self.arm_indices]
        return np.stack([lower, upper], axis=1)

    def _resolve_ground_check_bodies(self, body_indices: Sequence[int | str] | None) -> list[int]:
        if body_indices is not None:
            return [self._resolve_body(body) for body in body_indices]
        return [int(self.default_frame.body)]

    def _check_ground_penetration(self, full_q: Sequence[float]) -> tuple[bool, int | None, float]:
        self._fk_q.assign(np.asarray(full_q, dtype=np.float32))
        self._fk_qd.zero_()
        newton.eval_fk(self.model, self._fk_q, self._fk_qd, self._fk_state)
        body_q = self._fk_state.body_q.numpy()
        min_body = None
        min_z = float("inf")
        for body in self._ground_check_body_indices:
            z = float(body_q[int(body), 2])
            if z < min_z:
                min_z = z
                min_body = int(body)
        return min_z < self.ground_z_threshold, min_body, min_z

    def solve_fk(self, joint_angles: Union[list[float], np.ndarray], ee_link: Optional[str] = None) -> np.ndarray:
        """Return ``[x, y, z, qw, qx, qy, qz]`` for the selected TCP frame."""

        frame = self._frame(ee_link)
        q = self._full_q_with_arm(joint_angles)
        self._fk_q.assign(q.astype(np.float32))
        self._fk_qd.zero_()
        newton.eval_fk(self.model, self._fk_q, self._fk_qd, self._fk_state)

        body_pose = self._fk_state.body_q.numpy()[int(frame.body)]
        q_xyzw = body_pose[3:7].astype(float)
        pos = body_pose[:3].astype(float) + self._quat_rotate(q_xyzw, np.asarray(frame.offset, dtype=float))
        return np.array([pos[0], pos[1], pos[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

    def solve_ik(
        self,
        target_pose: Union[np.ndarray, list[float]],
        seed_joints: Optional[np.ndarray] = None,
        threshold: float = 0.005,
        ee_link: Optional[str] = None,
        pose_constraint: Optional[ConstraintType] = None,
        constraint_frame: str = "world",
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Solve IK and return this chain's joint coordinates.

        ``pose_constraint`` is currently supported as a coarse position/rotation
        switch. Per-axis weighting needs custom Newton IK objectives.
        """

        del constraint_frame

        frame = self._frame(ee_link)
        target = np.asarray(target_pose, dtype=float)
        if target.shape[0] < 7:
            raise ValueError("target_pose must be [x, y, z, qw, qx, qy, qz]")

        target_pos = target[:3]
        target_q_xyzw = np.array([target[4], target[5], target[6], target[3]], dtype=float)
        target_q_xyzw /= max(np.linalg.norm(target_q_xyzw), 1.0e-12)
        use_rotation = self._constraint_enables_rotation(pose_constraint)

        self.pos_obj.link_index = int(frame.body)
        self.pos_obj.link_offset = wp.vec3(*frame.offset)
        self.pos_obj.weight = 1.0
        self.pos_obj.set_target_position(0, wp.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])))

        self.rot_obj.link_index = int(frame.body)
        self.rot_obj.weight = 1.0 if use_rotation else 0.0
        self.rot_obj.set_target_rotation(
            0,
            wp.vec4(
                float(target_q_xyzw[0]),
                float(target_q_xyzw[1]),
                float(target_q_xyzw[2]),
                float(target_q_xyzw[3]),
            ),
        )

        seed_full = kwargs.pop("seed_full_q", None)
        if seed_full is None:
            seed_full = self._current_full_q()
        else:
            seed_full = np.asarray(seed_full, dtype=float).copy()
        if seed_joints is not None:
            seed_full[self.arm_indices] = np.asarray(seed_joints, dtype=float)[: len(self.arm_indices)]

        self._ik_q.assign(np.asarray(seed_full, dtype=np.float32)[None, :])
        self.solver.step(self._ik_q, self._ik_q, iterations=self.iterations)

        solved_full = self._ik_q.numpy()[0].astype(float)
        candidate = solved_full[self.arm_indices].copy()
        candidate = np.clip(candidate, self.joint_limits[:, 0], self.joint_limits[:, 1])

        fk = self.solve_fk(candidate, ee_link=ee_link)
        pos_err = float(np.linalg.norm(fk[:3] - target_pos))
        if pos_err > max(float(threshold), self.position_tolerance):
            return None
        if use_rotation:
            rot_err = self._quat_angle(np.array([fk[4], fk[5], fk[6], fk[3]], dtype=float), target_q_xyzw)
            if rot_err > self.rotation_tolerance:
                return None

        if self.ground_check_enabled:
            full_q = self._full_q_with_arm(candidate, seed_full_q=solved_full)
            penetrates, _body, _min_z = self._check_ground_penetration(full_q)
            if penetrates:
                return None

        return candidate


NewtonTeleopKinematics = NewtonKinematics
