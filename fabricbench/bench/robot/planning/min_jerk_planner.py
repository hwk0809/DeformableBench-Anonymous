"""Minimum Jerk trajectory planner.

Produces bell-shaped velocity profiles that closely match human arm movement
characteristics (Flash & Hogan 1985), making it suitable for sim-to-real
co-training scenarios.

Path construction (waypoints -> IK -> joint path) is identical to ToppraPlanner.
Only the time parameterization differs:
  ToppRA  -> time-optimal (trapezoidal velocity, discontinuous acceleration)
  MinJerk -> bell-shaped velocity, continuous jerk, v=0 & a=0 at endpoints

Key parameter:
  avg_speed (rad/s) — average joint-space velocity along the path.
  Peak velocity = 1.875 * avg_speed (from the min-jerk profile).
  Use velocity_scale in kwargs to scale per-call (same semantics as ToppRA).
"""

import numpy as np
import os
import sys
from typing import List, Optional, Union
from loguru import logger
from scipy.interpolate import interp1d

try:
    from spatialmath import SE3, UnitQuaternion
except ImportError as e:
    logger.error(f"[MinJerkPlanner] Missing spatialmath: {e}")
    raise

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fabricbench.bench.robot.planning.base_planner import BasePlanner
from fabricbench.bench.robot.kinematics.base_kinematics import BaseKinematics

try:
    from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
except ImportError as e:
    logger.error(f"[MinJerkPlanner] Kinematics not found: {e}")
    raise

from fabricbench.bench.common.types import PlanResult
from fabricbench.bench.common.errors import PlannerErrorCode


class MinJerkPlanner(BasePlanner):
    """Minimum jerk time parameterization planner.

    Same path construction as ToppraPlanner (Cartesian/joint interpolation + IK),
    but replaces ToppRA's time-optimal parameterization with a 5th-order
    polynomial profile: s(t) = 10t^3 - 15t^4 + 6t^5.
    """

    def __init__(
        self,
        urdf_path: str,
        active_joint_names: List[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        kinematics: Union[BaseKinematics, None] = None,
        base_link: str = "base_link",
        ee_link: str = "tcp_link",
        avg_speed: float = 1.0,
        trajectory_frequency: int = 50,
        **kwargs
    ):
        super().__init__(urdf_path, active_joint_names, **kwargs)

        if kinematics is None:
            kinematics = PinocchioKinematics(
                urdf_path=urdf_path,
                base_link=base_link,
                ee_link=ee_link,
                base_height=kwargs.get("base_height", 0.0),
            )
        self.kinematics = kinematics
        self.base_link = base_link
        self.ee_link = ee_link

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.dof = len(active_joint_names)
        self.avg_speed = avg_speed
        self.trajectory_frequency = trajectory_frequency

        if "velocity_limits" in kwargs or "acceleration_limits" in kwargs:
            logger.info(
                f"[MinJerkPlanner] velocity_limits/acceleration_limits ignored — "
                f"using avg_speed={avg_speed} rad/s"
            )

    # =========================================================================
    # Core: Minimum Jerk time parameterization
    # =========================================================================

    def _run_min_jerk(
        self,
        waypoints: np.ndarray,
        scaling_factor: float = 1.0,
        acceleration_scaling_factor: Optional[float] = None,
        ee_link: Optional[str] = None,
        keyframe_indices: Optional[List[int]] = None,
    ) -> PlanResult:
        """Minimum jerk time parameterization on joint-space waypoints.

        Interface mirrors ToppraPlanner._run_toppra for drop-in compatibility.
        acceleration_scaling_factor is accepted but not used (the min-jerk
        profile determines acceleration intrinsically).
        """
        if len(waypoints) < 2:
            return PlanResult.make_error(
                PlannerErrorCode.INVALID_INPUT, "Need >= 2 waypoints"
            )

        try:
            # 1. Arc-length parameterization in joint space
            diffs = np.diff(waypoints, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
            total_len = cum_len[-1]

            # Near-zero motion — return short hold
            if total_len < 1e-6:
                return self._make_hold_result(waypoints[0], 0.1, ee_link)

            # 2. Estimate total duration from avg_speed
            effective_speed = self.avg_speed * scaling_factor
            if effective_speed < 1e-6:
                effective_speed = 0.1
            T = total_len / effective_speed
            T = max(T, 0.2)

            # 3. Normalized arc-length for each waypoint
            s_wp = cum_len / total_len  # [0, 1]

            # 4. Minimum jerk profile: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            num_samples = max(10, int(T * self.trajectory_frequency))
            ts = np.linspace(0, T, num_samples)
            tau = ts / T
            s_t = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            # 5. Interpolate waypoints at min-jerk arc-length positions
            kind = "cubic" if len(waypoints) >= 4 else "linear"
            f_path = interp1d(s_wp, waypoints, axis=0, kind=kind)
            qs = f_path(np.clip(s_t, 0.0, 1.0))

            # 6. Velocity & acceleration via numerical gradient
            qds = np.gradient(qs, ts, axis=0)
            qdds = np.gradient(qds, ts, axis=0)

            # 7. Forward kinematics
            N = len(qs)
            ee_positions = np.zeros((N, 3))
            ee_quaternions = np.zeros((N, 4))
            for i, q in enumerate(qs):
                ee_pose = self.kinematics.solve_fk(q, ee_link=ee_link)
                ee_positions[i] = ee_pose[:3]
                ee_quaternions[i] = ee_pose[3:]

            res = PlanResult.make_success(
                time=ts, pos=qs, vel=qds, acc=qdds,
                joint_names=self.active_joint_names,
                ee_position=ee_positions,
                ee_quaternion=ee_quaternions,
            )

            # 8. Keyframe timing (same logic as ToppRA)
            if keyframe_indices is not None:
                key_times = []
                last_search_idx = 0
                for idx in keyframe_indices:
                    if idx >= len(waypoints):
                        continue
                    target_q = waypoints[idx]
                    search_slice = qs[last_search_idx:]
                    if len(search_slice) == 0:
                        key_times.append(ts[-1])
                        continue
                    dists = np.linalg.norm(search_slice - target_q, axis=1)
                    local_min_idx = np.argmin(dists)
                    global_idx = last_search_idx + local_min_idx
                    key_times.append(ts[global_idx])
                    last_search_idx = global_idx
                res.waypoint_times = key_times

            return res

        except Exception as e:
            logger.exception(f"[MinJerkPlanner] Exception in _run_min_jerk: {e}")
            return PlanResult.make_error(PlannerErrorCode.TIME_PARAM_FAILED, str(e))

    def _make_hold_result(self, joint_config: np.ndarray, duration: float,
                          ee_link: Optional[str] = None) -> PlanResult:
        """Generate a short static trajectory (near-zero motion)."""
        n = max(2, int(duration * self.trajectory_frequency))
        ts = np.linspace(0, duration, n)
        qs = np.tile(joint_config, (n, 1))
        qds = np.zeros_like(qs)
        qdds = np.zeros_like(qs)

        ee_positions = np.zeros((n, 3))
        ee_quaternions = np.zeros((n, 4))
        for i in range(n):
            p = self.kinematics.solve_fk(joint_config, ee_link=ee_link)
            ee_positions[i] = p[:3]
            ee_quaternions[i] = p[3:]

        return PlanResult.make_success(
            time=ts, pos=qs, vel=qds, acc=qdds,
            joint_names=self.active_joint_names,
            ee_position=ee_positions,
            ee_quaternion=ee_quaternions,
        )

    # =========================================================================
    # Planning interface (path construction identical to ToppraPlanner)
    # =========================================================================

    def plan_to_joint_config(
        self,
        start_joints: np.ndarray,
        goal_joints: np.ndarray,
        **kwargs,
    ) -> PlanResult:
        """Joint space planning: linear interpolation + min-jerk time param."""
        velocity_scale = kwargs.get("velocity_scale", 1.0)
        num_interp = 20

        diff = goal_joints - start_joints
        waypoints = start_joints + np.outer(np.linspace(0, 1, num_interp), diff)

        return self._run_min_jerk(waypoints, scaling_factor=velocity_scale)

    def plan_to_pose(
        self,
        start_joints: np.ndarray,
        goal_pose: np.ndarray,
        pose_constraint: Optional[Union[np.ndarray, List[float]]] = None,
        constraint_frame: str = "world",
        **kwargs,
    ) -> PlanResult:
        """Cartesian space planning: IK + min-jerk time param."""
        constraint_arg = [pose_constraint] if pose_constraint is not None else None
        return self.plan_waypoints(
            start_joints, [goal_pose],
            pose_constraint=constraint_arg,
            constraint_frame=constraint_frame,
            **kwargs,
        )

    def plan_waypoints(
        self,
        start_joints: np.ndarray,
        waypoints: List[np.ndarray],
        pose_constraint: Optional[Union[np.ndarray, List[float], List[List[float]]]] = None,
        constraint_frame: str = "world",
        interpolation_mode: Union[str, List[str]] = "joint",
        **kwargs,
    ) -> PlanResult:
        """Cartesian path planning (Waypoint Following) with min-jerk timing.

        Args:
            start_joints: Starting joint configuration
            waypoints: List of target poses [x,y,z,qw,qx,qy,qz]
            pose_constraint: [x,y,z,rx,ry,rz] weights (0-1)
            constraint_frame: "world" or "local"
            interpolation_mode: "cartesian" or "joint" per segment
            **kwargs:
                velocity_scale: float, scales avg_speed
                steps_per_segment: int, interpolation density for cartesian mode
                ee_link: str, end-effector link override
        """
        velocity_scale = kwargs.get("velocity_scale", 1.0)
        steps_per_segment = kwargs.get("steps_per_segment", 10)
        ee_link = kwargs.get("ee_link", None)

        num_wps = len(waypoints)
        if num_wps == 0:
            return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "No waypoints")
        if self.kinematics is None:
            return PlanResult.make_error(PlannerErrorCode.BACKEND_UNAVAILABLE, "No kinematics")

        # --- 1. Standardize inputs ---
        modes = self._ensure_list_input(interpolation_mode, num_wps, "interpolation_mode", default="joint")
        constraints = self._ensure_list_input(pose_constraint, num_wps, "pose_constraint", default=[1] * 6)
        steps_list = self._ensure_list_input(steps_per_segment, num_wps, "steps", default=10)

        # --- 2. Global Skeleton IK ---
        skeleton_joints = [start_joints]
        curr_seed = start_joints

        for i, wp_pose in enumerate(waypoints):
            sol = self.kinematics.solve_ik(
                target_pose=wp_pose,
                seed_joints=curr_seed,
                pose_constraint=constraints[i],
                constraint_frame=constraint_frame,
                ee_link=ee_link,
            )
            if sol is None:
                msg = f"IK Failed at waypoint index {i} (Pose: {np.round(wp_pose[:3], 3)})"
                logger.warning(f"[MinJerkPlanner] {msg}")
                return PlanResult.make_error(PlannerErrorCode.IK_FAILED_GLOBAL, msg)

            skeleton_joints.append(sol)
            curr_seed = sol

        # --- 3. Construct trajectory segment by segment ---
        full_path_joints = [skeleton_joints[0]]
        waypoint_indices = [0]

        for i in range(num_wps):
            mode = modes[i]
            target_joints_knot = skeleton_joints[i + 1]
            target_pose_knot = waypoints[i]
            current_seg_steps = steps_list[i]
            current_constraint = constraints[i]
            segment_start_joints = full_path_joints[-1]

            if mode == "joint":
                full_path_joints.append(target_joints_knot)
                waypoint_indices.append(len(full_path_joints) - 1)

            elif mode == "cartesian":
                seg_start_pose = self.kinematics.solve_fk(segment_start_joints, ee_link=ee_link)
                cartesian_path_joints = self._cartesian_interpolate(
                    start_joints=segment_start_joints,
                    start_pose=seg_start_pose,
                    target_pose=target_pose_knot,
                    steps=current_seg_steps,
                    constraint=current_constraint,
                    constraint_frame=constraint_frame,
                    ee_link=ee_link,
                )
                if cartesian_path_joints is None:
                    return PlanResult.make_error(
                        PlannerErrorCode.IK_FAILED_GLOBAL,
                        f"Cartesian interpolation failed in segment {i}",
                    )
                cartesian_path_joints[-1] = target_joints_knot
                full_path_joints.extend(cartesian_path_joints[1:])
                waypoint_indices.extend(
                    range(len(full_path_joints) - len(cartesian_path_joints) + 1, len(full_path_joints))
                )
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, f"Unknown mode: {mode}")

        full_path_joints = np.array(full_path_joints)

        # --- 4. Min-jerk time parameterization ---
        return self._run_min_jerk(
            full_path_joints,
            scaling_factor=velocity_scale,
            ee_link=ee_link,
            keyframe_indices=waypoint_indices,
        )

    # =========================================================================
    # Utility methods (shared with ToppraPlanner)
    # =========================================================================

    def _cartesian_interpolate(
        self,
        start_joints: np.ndarray,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        steps: int,
        constraint: List[float],
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Cartesian space linear interpolation (SE3) with IK at each step."""
        def to_se3(pose):
            return SE3(pose[:3]) * UnitQuaternion(pose[3:]).SE3()

        T_start = to_se3(start_pose)
        T_target = to_se3(target_pose)
        s_vals = np.linspace(0, 1, steps)
        T_sequence = [T_start.interp(T_target, s) for s in s_vals]

        joint_traj = [start_joints]
        curr_joints = start_joints

        for i, T in enumerate(T_sequence):
            if i == 0:
                continue
            pos = T.t
            quat = UnitQuaternion(T.R).A
            target_vec = np.concatenate([pos, quat])

            sol = self.kinematics.solve_ik(
                target_pose=target_vec,
                seed_joints=curr_joints,
                pose_constraint=constraint,
                constraint_frame=constraint_frame,
                ee_link=ee_link,
            )
            if sol is None:
                logger.warning(f"IK failed at interpolation step {i}/{steps}")
                return None

            joint_traj.append(sol)
            curr_joints = sol

        return np.array(joint_traj)

    def _ensure_list_input(self, input_val, length, name, default=None):
        """Helper to ensure input is a list of correct length."""
        if input_val is None:
            return [default] * length
        if isinstance(input_val, (list, np.ndarray)):
            if name == "pose_constraint":
                if len(input_val) > 0 and isinstance(input_val[0], (int, float)):
                    return [input_val] * length
            if len(input_val) == length:
                return input_val
            if len(input_val) != length and name != "pose_constraint":
                logger.warning(
                    f"[{name}] Input list length {len(input_val)} != waypoints length {length}."
                )
                return input_val
        return [input_val] * length


if __name__ == "__main__":
    import time
    from fabricbench.bench.robot.utils.visualize_trajectory import plot_plan_trajectory

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_path = os.path.join(
        project_root,
        "assets/robot/piper/urdf/piper_with_gripper.urdf",
    )

    start_joints = np.array([0.0, 0.2, -0.2, 0.0, 0.0, 0.0])
    goal_joints = np.array([0.5, 0.3, -0.2, 0.0, 0.4, 0.1])
    target_pose = np.array([0.3, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

    print("=" * 60)
    print("Testing MinJerkPlanner")
    print("=" * 60)

    planner = MinJerkPlanner(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="tcp_link",
        active_joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        avg_speed=1.0,
        trajectory_frequency=50,
    )

    # --- Test 1: Joint Planning ---
    print("\n[Test 1] Plan to Joint Config...")
    t_start = time.time()
    res = planner.plan_to_joint_config(start_joints, goal_joints)

    print(f"  Time: {(time.time() - t_start)*1000:.2f} ms")
    print(f"  Status: {res.status}")
    print(f"  Success: {res.success}")

    if res.success:
        print(f"  Trajectory points: {len(res.position)}")
        print(f"  Duration: {res.time[-1]:.2f} s")
        print(f"  Peak joint vel: {np.max(np.abs(res.velocity)):.3f} rad/s")

    # --- Test 2: Pose Planning ---
    print("\n[Test 2] Plan to Pose...")
    t_start = time.time()
    res = planner.plan_to_pose(start_joints, target_pose)

    print(f"  Time: {(time.time() - t_start)*1000:.2f} ms")
    print(f"  Status: {res.status}")

    if res.success:
        print(f"  Trajectory points: {len(res.position)}")
        print(f"  Duration: {res.time[-1]:.2f} s")
    else:
        print(f"  Error: {res.error_code} — {res.error_msg}")

    # --- Test 3: Waypoints ---
    print("\n[Test 3] Plan Waypoints Sequence...")
    waypoints = [
        np.array([0.3, 0.1, 0.2, 1, 0, 0, 0]),
        np.array([0.3, -0.1, 0.2, 1, 0, 0, 0]),
        np.array([0.3, 0.0, 0.3, 1, 0, 0, 0]),
    ]
    res = planner.plan_waypoints(start_joints, waypoints, velocity_scale=0.8)

    print(f"  Status: {res.status}")
    if res.success:
        print(f"  Duration: {res.time[-1]:.2f} s")

    # --- Test 4: Cartesian interpolation ---
    print("\n[Test 4] Plan Waypoints (Cartesian Interpolation)...")
    waypoints_easy = [
        np.array([0.3, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0]),
        np.array([0.3, 0.1, 0.2, 0.7071, 0.0, 0.7071, 0.0]),
    ]
    res = planner.plan_waypoints(
        start_joints,
        waypoints_easy,
        velocity_scale=0.8,
        interpolation_mode="cartesian",
        steps_per_segment=20,
    )

    print(f"  Status: {res.status}")
    if res.success:
        print(f"  Duration: {res.time[-1]:.2f} s")
        print(f"  Peak joint vel: {np.max(np.abs(res.velocity)):.3f} rad/s")

    # --- Test 5: velocity_scale comparison ---
    print("\n[Test 5] velocity_scale effect...")
    for vs in [0.5, 1.0, 2.0]:
        res = planner.plan_to_joint_config(start_joints, goal_joints, velocity_scale=vs)
        if res.success:
            print(f"  velocity_scale={vs}: duration={res.time[-1]:.2f}s, "
                  f"peak_vel={np.max(np.abs(res.velocity)):.3f} rad/s")

    print("\nAll tests passed!")
