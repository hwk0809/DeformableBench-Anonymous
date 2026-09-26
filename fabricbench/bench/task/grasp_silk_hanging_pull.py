"""Single-arm hanging-silk corner grasp + pull task.

Sequence: reset silk into a vertical hanging sheet -> kinematically pin the top
edge -> move one gripper to the lower corner -> close -> pull back toward the
robot. This is intentionally close to ``grasp_silk_bimanual`` but uses the
env-level particle pin primitive to model the fixed upper edge.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from fabricbench.bench.common.success_checker import SuccessChecker
from fabricbench.bench.common.types import ExecutionStep, ObservationMessage

_POS_ONLY = [1, 1, 1, 0, 0, 0]


def _obs(env) -> ObservationMessage:
    return ObservationMessage(
        joint_left=env.left_qpos,
        joint_right=env.right_qpos,
        eef_pose_left=env.left_eef_pose,
        eef_pose_right=env.right_eef_pose,
        gripper_left=env.left_gripper_val,
        gripper_right=env.right_gripper_val,
    )


class SilkHangingCornerPullTask:
    """Open-loop single-arm gripper test on a vertically pinned silk sheet."""

    def __init__(self, spec):
        self.spec = spec
        p = spec.task_params

        self.arm = str(p.get("arm", "right")).lower()
        if self.arm not in ("left", "right"):
            raise ValueError(f"task_params.arm must be 'left' or 'right', got {self.arm!r}")

        self.open_val = float(p.get("grasp_open_val", 1.0))
        self.close_val = float(p.get("grasp_close_val", 0.03))

        # Hanging cloth placement. The existing builder creates a horizontal sheet;
        # the task rigidly remaps it into an XZ plane at hang_x.
        self.hang_x = p.get("hang_x", None)
        self.hang_y = p.get("hang_y", None)
        # World-Z yaw in degrees. Positive follows right-hand rule; negative is
        # clockwise when viewed from +Z.
        self.hang_yaw_deg = float(p.get("hang_yaw_deg", 0.0))
        self.top_z = float(p.get("top_z", 0.58))
        self.top_edge_width = float(p.get("top_edge_width", 0.012))
        self.min_top_pins = int(p.get("min_top_pins", 8))
        self.hanging_settle_frames = int(p.get("hanging_settle_frames", 60))

        # Grasp lower corner and pull back toward the robot side (-x).
        self.approach_x_offset = float(p.get("approach_x_offset", 0.10))
        self.grasp_x_offset = float(p.get("grasp_x_offset", -0.002))
        self.grasp_z_offset = float(p.get("grasp_z_offset", 0.0))
        # Move the selected lower corner inward toward the current cloth center
        # so the gripper pinches fabric area instead of the exact corner vertex.
        self.grasp_center_inset = float(p.get("grasp_center_inset", p.get("grasp_edge_inset", 0.015)))
        self.pull_distance = float(p.get("pull_distance", 0.18))
        self.pull_lift = float(p.get("pull_lift", 0.02))

        self.close_duration = float(p.get("close_duration", 0.6))
        self.pull_settle = float(p.get("pull_settle", 0.8))

        self.success_grasp_dist = float(p.get("success_grasp_dist", 0.055))
        self.success_pull_delta = float(p.get("success_pull_delta", 0.08))

        self._top_pin_positions: dict[int, np.ndarray] = {}
        self._grasp_idx: int | None = None
        self._grasp_start_x: float | None = None

    # ------------------------------------------------------------------
    # Cloth setup
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_cloth_state(env, q: np.ndarray) -> None:
        env.state_0.particle_q.assign(q)
        env.state_1.particle_q.assign(q)
        try:
            env.state_0.particle_qd.zero_()
            env.state_1.particle_qd.zero_()
        except Exception:
            z = np.zeros_like(q)
            env.state_0.particle_qd.assign(z)
            env.state_1.particle_qd.assign(z)

    def _pin_top_edge(self, env, q: np.ndarray, top_idx: np.ndarray) -> None:
        self._top_pin_positions = {int(i): q[int(i)].copy() for i in top_idx}
        for idx, pos in self._top_pin_positions.items():
            env.set_particle_pin(idx, pos)

    def _refresh_top_pins(self, env) -> None:
        for idx, pos in self._top_pin_positions.items():
            env.set_particle_pin(idx, pos)

    def setup_hanging_cloth(self, env) -> tuple[np.ndarray, np.ndarray]:
        """Place the current cloth as a vertical sheet and pin its top edge.

        The input silk is roughly square in XY. We keep the current Y coordinate
        as horizontal span, map the current X coordinate to vertical Z, and put all
        particles in one X plane. The edge with maximum original X becomes the
        fixed top edge; the opposite edge becomes the lower graspable edge.
        """

        env.clear_particle_pins()
        q0 = env.get_sim_vertices().copy()
        local_x = q0[:, 0]
        local_y = q0[:, 1]

        x_max = float(local_x.max())
        y_mid = float(local_y.mean())
        hang_x = float(q0[:, 0].mean() if self.hang_x is None else self.hang_x)
        hang_y = float(q0[:, 1].mean() if self.hang_y is None else self.hang_y)

        q = q0.copy()
        q[:, 0] = hang_x
        q[:, 1] = hang_y + (local_y - y_mid)
        q[:, 2] = self.top_z - (x_max - local_x)
        if abs(self.hang_yaw_deg) > 1e-9:
            theta = np.deg2rad(self.hang_yaw_deg)
            c, s = np.cos(theta), np.sin(theta)
            xy = q[:, :2] - np.array([hang_x, hang_y])
            q[:, 0] = hang_x + c * xy[:, 0] - s * xy[:, 1]
            q[:, 1] = hang_y + s * xy[:, 0] + c * xy[:, 1]

        top_idx = np.flatnonzero(local_x >= x_max - self.top_edge_width)
        if len(top_idx) < self.min_top_pins:
            order = np.argsort(x_max - local_x)
            top_idx = order[: self.min_top_pins]

        self._assign_cloth_state(env, q)
        self._pin_top_edge(env, q, top_idx)

        hold = (env.left_qpos, env.right_qpos, env.left_gripper_val, env.right_gripper_val)
        for _ in range(max(0, self.hanging_settle_frames)):
            self._refresh_top_pins(env)
            env.step(*hold)

        q = env.get_sim_vertices().copy()
        logger.info(
            f"[grasp_silk_hanging_pull] hanging cloth set: "
            f"top_pins={len(top_idx)} center=({hang_x:.3f},{hang_y:.3f}) "
            f"yaw={self.hang_yaw_deg:.1f}deg "
            f"z[min/max]={q[:, 2].min():.3f}/{q[:, 2].max():.3f}"
        )
        return q, top_idx

    # ------------------------------------------------------------------
    # Grasp geometry
    # ------------------------------------------------------------------
    def lower_corner(self, cloth: np.ndarray) -> tuple[int, np.ndarray]:
        z_min = float(cloth[:, 2].min())
        bottom = np.flatnonzero(cloth[:, 2] <= z_min + self.top_edge_width)
        if len(bottom) == 0:
            bottom = np.array([int(np.argmin(cloth[:, 2]))])

        if self.arm == "left":
            idx = int(bottom[np.argmax(cloth[bottom, 1])])
        else:
            idx = int(bottom[np.argmin(cloth[bottom, 1])])

        target = cloth[idx].copy()
        center = cloth.mean(axis=0)
        inward = center - target
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm > 1e-8 and self.grasp_center_inset > 0.0:
            target += self.grasp_center_inset * inward / inward_norm

        target[2] += self.grasp_z_offset
        target[0] += self.grasp_x_offset
        return idx, target

    def _active_base(self, env) -> np.ndarray:
        return env.left_base if self.arm == "left" else env.right_base

    def _active_eef(self, env) -> np.ndarray:
        return env.left_eef_pose if self.arm == "left" else env.right_eef_pose

    def _pose(self, skills, env, xyz) -> np.ndarray:
        pos = np.asarray(xyz, dtype=float).copy()
        quat = skills.calculate_base_to_target_horizontal_quat(self._active_base(env), pos)
        return np.concatenate([pos, quat])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, env, skills, executor) -> bool:
        cloth, _top_idx = self.setup_hanging_cloth(env)
        grasp_idx, grasp_xyz = self.lower_corner(cloth)
        self._grasp_idx = grasp_idx
        self._grasp_start_x = float(cloth[grasp_idx, 0])

        grasp_pose = self._pose(skills, env, grasp_xyz)
        approach_pose = grasp_pose.copy()
        approach_pose[0] -= self.approach_x_offset
        pull_pose = grasp_pose.copy()
        pull_pose[0] -= self.pull_distance
        pull_pose[2] += self.pull_lift

        logger.info(
            f"[grasp_silk_hanging_pull] arm={self.arm} lower_corner_idx={grasp_idx} "
            f"grasp={grasp_xyz} pull={pull_pose[:3]}"
        )

        def move(pose, grip, settle, desc):
            skills.sync_state(_obs(env))
            plan = skills.move_to(
                pose,
                arm=self.arm,
                goal_type="pose",
                ee_link="tcp_link_bottom",
                pose_constraint=_POS_ONLY,
                pose_constraint_frame="local",
                gripper=grip,
                other_arm="hold",
            )
            if not plan.success:
                logger.error(f"[grasp_silk_hanging_pull:{desc}] plan failed")
                return False
            return executor.execute_plan(ExecutionStep(plan=plan, settle_time=settle, description=desc))

        def close():
            skills.sync_state(_obs(env))
            plan = skills.gripper_action(
                self.close_val,
                arm=self.arm,
                duration=self.close_duration,
            )
            if not plan.success:
                logger.error("[grasp_silk_hanging_pull:close] plan failed")
                return False
            return executor.execute_plan(ExecutionStep(plan=plan, settle_time=0.3, description="close"))

        ok = (
            move(approach_pose, self.open_val, 0.2, "approach")
            and move(grasp_pose, self.open_val, 0.2, "enter_corner")
            and close()
            and move(pull_pose, self.close_val, self.pull_settle, "pull_back")
        )
        if not ok:
            logger.warning("[grasp_silk_hanging_pull] a motion phase failed")

        for _ in range(int(self.spec.eval.get("settle_after_frames", 30))):
            self._refresh_top_pins(env)
            env.step(env.left_qpos, env.right_qpos, env.left_gripper_val, env.right_gripper_val)

        cfg = self.spec.success_check_cfg
        checker = SuccessChecker(cfg, env, custom_check=lambda: self._check_pull_success(env))
        success = checker.check()
        cloth = env.get_sim_vertices()
        logger.info(
            f"[grasp_silk_hanging_pull] success={success} "
            f"corner_x={cloth[grasp_idx, 0]:.3f} start_x={self._grasp_start_x:.3f} "
            f"ee_dist={np.linalg.norm(cloth[grasp_idx] - self._active_eef(env)[:3]):.3f}"
        )
        return bool(success)

    def _check_pull_success(self, env) -> bool:
        if self._grasp_idx is None or self._grasp_start_x is None:
            return False
        cloth = env.get_sim_vertices()
        corner = cloth[self._grasp_idx]
        ee = self._active_eef(env)[:3]
        pulled = (self._grasp_start_x - float(corner[0])) >= self.success_pull_delta
        close = float(np.linalg.norm(corner - ee)) <= self.success_grasp_dist
        return bool(pulled and close)
