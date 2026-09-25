"""Dual-arm grasp-silk-and-lift task (closed-loop robust grasp).

Sequence: settle cloth -> pick the +y / -y near cloth corners -> robust_bimanual_grasp
(direct approach to the grasp pose at velocity_scale, close, test-lift verify, retry
on slip) -> dual-arm lift -> evaluate ``bimanual_hold`` + ``mean_z``.

Aligned with the reference SilkGraspBimanualTask (velocity_scale=0.15, gripper 0.015,
grasp_depth -0.001, [1,1,1,1,0,1] local). Earlier this was a full-speed open-loop
placeholder whose recorded demos differed from the reference data (2x faster, no
retry) — the source of the trained-policy SR gap.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from deformablebench.bench.common import checks
from deformablebench.bench.common.types import DualPlanResult, ExecutionStep, ObservationMessage

_POS_ONLY = [1, 1, 1, 0, 0, 0]
# Grasp constraint mirrors the reference task: lock position + roll/yaw, free pitch.
_GRASP_CONSTRAINT = [1, 1, 1, 1, 0, 1]


def _obs(env) -> ObservationMessage:
    return ObservationMessage(
        joint_left=env.left_qpos, joint_right=env.right_qpos,
        eef_pose_left=env.left_eef_pose, eef_pose_right=env.right_eef_pose,
        gripper_left=env.left_gripper_val, gripper_right=env.right_gripper_val,
    )


class SilkGraspBimanualTask:
    """Open-loop bimanual grasp + lift driven by a TaskSpec.

    Also owns this task's evaluation (RoboTwin/RoboDojo style): ``observe`` each
    poll, then ``check_success`` / ``get_score`` — both composed from
    :mod:`common.checks`. The driver (eval_sim / this run()) only calls them.
    """

    # This task's criteria (single source of truth; supersedes the yaml
    # eval.success_check for grasp). Each hand "holds" if the nearest cloth
    # particle is within GRASP_DIST of the EE and above LIFT_H; the whole cloth
    # is "lifted" if mean height > MEAN_Z.
    GRASP_DIST = 0.05
    LIFT_H = 0.10
    MEAN_Z = 0.05

    def __init__(self, spec):
        self.spec = spec
        self._hist: list[dict] = []      # per-poll atom record for scoring
        p = spec.task_params
        self.grasp_close = float(p.get("grasp_close_val", 0.0))  # 0 = closed (open fraction)
        # Move speed: the open-loop moves pass no velocity_scale -> toppra default
        # 1.0 (full 5 rad/s), ~2x the per-step deltas of the high-SR robust data
        # (velocity_scale=0.15). Match it so the recorded trajectory profile aligns.
        self.move_vel_scale = float(p.get("move_vel_scale", 0.15))
        self.approach_h = float(p.get("approach_height", 0.12))
        self.lift_h = float(p.get("lift_height", 0.4))
        self.lift_xy = p.get("lift_xy", [0.2, 0.17])  # |y| for each arm at lift
        # Grasp depth: fingertip (tcp_link_bottom) target z relative to the cloth
        # surface. Slightly into the cloth, like the reference's GRASP_Z_OFFSET.
        self.grasp_z_offset = float(p.get("grasp_z_offset", -0.001))
        # Move the chosen corner grasp points inward along the cloth plane so the
        # gripper pinches fabric area instead of only the sharp corner vertex.
        self.grasp_xy_inset = float(p.get("grasp_xy_inset", 0.0))
        self.grasp_point_mode = str(p.get("grasp_point_mode", "asset_keypoints"))
        if self.grasp_point_mode not in (
            "asset_keypoints", "asset_keypoints_random"
        ):
            raise ValueError(
                "grasp_point_mode must be 'asset_keypoints', "
                "or 'asset_keypoints_random', "
                f"got {self.grasp_point_mode!r}"
            )
        # Semantic corners are bound from env.cloth_keypoints in run().  Vertex
        # ids belong to the selected asset and must never live in a task/config.
        self._asset_corner_indices: tuple[int, ...] | None = None
        self.inward_jitter_radius = float(p.get("inward_jitter_radius", 0.025))
        # arm base positions resolved at run() from env.left_base/right_base

    def _corner_indices(self, cloth: np.ndarray) -> list[int]:
        """Return the four semantic corners declared by the selected asset."""
        if self._asset_corner_indices is None:
            raise RuntimeError("asset cloth keypoints have not been bound")
        if max(self._asset_corner_indices) >= len(cloth):
            raise IndexError(
                f"asset corner keypoint exceeds cloth vertex count {len(cloth)}"
            )
        return list(self._asset_corner_indices)

    def _jitter_grasp_index(self, idx: int, cloth: np.ndarray) -> int:
        """Match deformable_bench v9's inward-jitter selection exactly."""
        origin = cloth[idx]
        dists = np.linalg.norm(cloth - origin, axis=1)
        nearby = np.where(dists <= self.inward_jitter_radius)[0]
        nearby = nearby[nearby != idx]
        if len(nearby) == 0:
            return idx
        center = cloth.mean(axis=0)
        origin_r = np.linalg.norm(origin[:2] - center[:2])
        inward = nearby[
            np.linalg.norm(cloth[nearby, :2] - center[:2], axis=1) < origin_r
        ]
        if len(inward) == 0:
            return idx
        chosen = int(np.random.choice(inward))
        logger.info(
            f"[grasp_silk_bimanual] inward jitter {idx}->{chosen} "
            f"delta={np.linalg.norm(cloth[chosen] - origin) * 1000:.1f}mm "
            f"radius={self.inward_jitter_radius:.3f}m candidates={len(inward)}"
        )
        return chosen

    def grasp_corners(self, cloth: np.ndarray):
        """Pick the two NEAR cloth corners (closest to the robot at x=0), one per arm.

        Find the 4 rectangle corners as the extrema along the diagonal directions,
        keep the 2 with smallest x (the edge facing the robot), then assign the +y
        one to the left arm and the -y one to the right arm. Optionally inset each
        point toward the current cloth center in XY."""
        xy = cloth[:, :2]
        l_idx, r_idx = self._near_corner_indices(cloth)
        left, right = cloth[l_idx].copy(), cloth[r_idx].copy()

        if self.grasp_xy_inset > 0.0:
            center_xy = xy.mean(axis=0)

            def inset(point: np.ndarray) -> np.ndarray:
                delta = center_xy - point[:2]
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    point[:2] += self.grasp_xy_inset * delta / norm
                return point

            left = inset(left)
            right = inset(right)

        return left, right

    def _near_corner_indices(self, cloth: np.ndarray):
        """Same near-corner pick as grasp_corners, but returns the two cloth
        VERTEX INDICES (l_idx=+y, r_idx=-y) so robust_bimanual_grasp can re-lock
        the held arm on retry."""
        corner_idx = self._corner_indices(cloth)
        corners = cloth[corner_idx]
        near = np.argsort(corners[:, 0])[:2]          # 2 smallest-x corners (near edge)
        a, b = corner_idx[near[0]], corner_idx[near[1]]
        left, right = (a, b) if cloth[a, 1] >= cloth[b, 1] else (b, a)
        if self.grasp_point_mode == "asset_keypoints_random":
            left = self._jitter_grasp_index(left, cloth)
            right = self._jitter_grasp_index(right, cloth)
        return left, right

    def _grasp_pose(self, skills, target_xyz, base, z_offset):
        """Grasp pose at target with the fingertip z offset and a base->target
        horizontal-approach orientation (mirrors the reference task)."""
        pos = np.array(target_xyz, float).copy()
        pos[2] = max(pos[2] + z_offset, 0.0)
        quat = skills.calculate_base_to_target_horizontal_quat(base, pos)
        return np.concatenate([pos, quat])

    # --- evaluation (composed from common.checks) --------------------------
    def reset_eval(self) -> None:
        self._hist = []

    def observe(self, env) -> None:
        """Record this poll's atoms (called every check interval by the driver)."""
        c = env.get_cloth_position()
        if c is None:
            self._hist.append({"left": False, "right": False, "lifted": False,
                               "mean_z": 0.0, "max_z": 0.0})
            return
        self._hist.append({
            "left": checks.arm_hold(c, env.left_eef_pose, self.GRASP_DIST, self.LIFT_H),
            "right": checks.arm_hold(c, env.right_eef_pose, self.GRASP_DIST, self.LIFT_H),
            "lifted": checks.mean_z_ok(c, self.MEAN_Z),
            "mean_z": float(c[:, 2].mean()), "max_z": float(c[:, 2].max()),
        })

    def check_success(self) -> bool:
        """SR: both hands holding + whole cloth lifted, sustained ~1s."""
        return checks.sustained([h["left"] and h["right"] and h["lifted"] for h in self._hist])

    def get_score(self) -> float:
        """Score 0/25/50/75/100: per hand end-held=50 / grabbed-then-slipped=25 /
        never=0, summed. mean_z lives in SR only."""
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        hand = lambda k: 50 if checks.end_held(ser(k)) else 25 if checks.ever(ser(k)) else 0  # noqa: E731
        return float(hand("left") + hand("right"))

    def eval_metrics(self) -> dict:
        """Raw per-episode facts; Score is one derivation of these (recomposable)."""
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        return {
            "left_ever": checks.ever(ser("left")), "left_held": checks.end_held(ser("left")),
            "right_ever": checks.ever(ser("right")), "right_held": checks.end_held(ser("right")),
            "lifted_ever": checks.ever(ser("lifted")), "lifted_end": checks.end_held(ser("lifted")),
            "max_z": round(max((h["max_z"] for h in self._hist), default=0.0), 4),
            "mean_z_end": round(self._hist[-1]["mean_z"], 4) if self._hist else 0.0,
        }

    def run(self, env, skills, executor) -> bool:
        # Closed-loop robust grasp, aligned with the reference SilkGraspBimanualTask
        # (direct approach to the grasp pose, velocity_scale, retry + test-lift
        # verify) — replaces the old open-loop placeholder. The driver hands us a
        # plain RobotSkills, so build a ClosedLoopSkills exactly like BaseTask does.
        from deformablebench.bench.robot.skills.closed_loop_skills import ClosedLoopSkills
        self.left_base, self.right_base = env.left_base, env.right_base
        keypoints = env.cloth_keypoints
        corner_names = ("corner_LU", "corner_RU", "corner_RD", "corner_LD")
        missing = [name for name in corner_names if name not in keypoints]
        if missing:
            raise ValueError(f"cloth asset is missing grasp keypoints: {missing}")
        self._asset_corner_indices = tuple(int(keypoints[name]) for name in corner_names)
        logger.info(
            f"[grasp_silk_bimanual] asset keypoints "
            f"{dict(zip(corner_names, self._asset_corner_indices))}"
        )
        cl = ClosedLoopSkills(skills.planner_manager, env.get_observation)
        cl.sync_state(env.get_observation(include_visuals=False))

        def get_targets(obs, specific_indices=None, z_offset=None):
            cloth = obs.pcd_cloth_truth
            if specific_indices is not None and None not in specific_indices:
                l_idx, r_idx = specific_indices          # retry: keep held-arm index
            else:
                l_idx, r_idx = self._near_corner_indices(cloth)
            zoff = self.grasp_z_offset if z_offset is None else z_offset
            pose_l = self._grasp_pose(cl, cloth[l_idx], self.left_base, zoff)
            pose_r = self._grasp_pose(cl, cloth[r_idx], self.right_base, zoff)
            logger.info(f"[grasp_silk_bimanual] L={l_idx} {pose_l[:3]}  R={r_idx} {pose_r[:3]}")
            return pose_l, pose_r, int(l_idx), int(r_idx)

        # robust_bimanual_grasp is a generator (yields ExecutionSteps, returns the
        # bool success via StopIteration.value); drive it through the executor.
        gen = cl.robust_bimanual_grasp(
            get_targets_func=get_targets,
            max_retries=3,
            close_gripper_value=self.grasp_close,
            grasp_depth=self.grasp_z_offset,
            retry_depth_step=-0.005,
            pose_constraint=_GRASP_CONSTRAINT,
            pose_constraint_frame="local",
            test_lift_height=0.15,
            check_threshold_height=0.08,
            ee_link="tcp_link_bottom",
            grasp_settle_time=0.6,
            velocity_scale=self.move_vel_scale,
        )
        grasped = False
        try:
            step = next(gen)
            while True:
                executor.execute_plan(step)
                step = next(gen)
        except StopIteration as e:
            grasped = bool(e.value)

        if grasped:
            cl.sync_state(env.get_observation(include_visuals=False))
            lift = cl.move_dual_arm_to(
                left_target_pose=[self.lift_xy[0], self.lift_xy[1], self.lift_h],
                right_target_pose=[self.lift_xy[0], -self.lift_xy[1], self.lift_h],
                pose_constraint=_POS_ONLY, velocity_scale=0.5)
            executor.execute_plan(ExecutionStep(plan=lift, settle_time=0.5, description="dual_arm_lift"))
        else:
            logger.warning("[grasp_silk_bimanual] robust grasp failed to secure the cloth")

        # settle, then evaluate (observe each frame so check_success sees ~1s)
        self.reset_eval()
        for _ in range(int(self.spec.eval.get("settle_after_frames", 30))):
            env.step()
            self.observe(env)

        success = self.check_success()
        logger.info(f"[grasp_silk_bimanual] success={success} score={self.get_score()} "
                    f"metrics={self.eval_metrics()}")
        return success
