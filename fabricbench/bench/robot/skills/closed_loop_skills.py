"""
Closed-loop skills with observation feedback.

Extends RobotSkills with methods that observe-plan-verify in a loop:
  - robust_bimanual_grasp: 双臂闭环抓取 (approach→close→test_lift→verify→retry)
  - robust_single_grasp:   单臂闭环抓取
  - robust_place_single:   单臂闭环放置 + 黏连检测
  - robust_place_bimanual:  双臂闭环放置 + 黏连检测
  - check_single_arm_grasp: 抓取验证 helper
  - check_cloth_released:   释放验证 helper

Usage:
    self.skills = ClosedLoopSkills(planner_manager, env.get_observation, robot_cfg)
    # All RobotSkills methods remain: move_to, gripper_action, fling, etc.
    # Plus new closed-loop methods:
    success = yield from self.skills.robust_bimanual_grasp(get_targets_func=..., ...)
"""

import numpy as np
from typing import Optional, Union, Callable, Tuple, Generator, Literal, List
from loguru import logger
from omegaconf import DictConfig

from fabricbench.bench.robot.skills.primitive_skills import RobotSkills
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.common.types import (
    ArmSide, ExecutionStep, DualPlanResult, ObservationMessage,
)


def _pre_grasp_along_approach(target_7d: np.ndarray, distance: float) -> np.ndarray:
    """沿 grasp orientation 的 -TCP_z (approach 反向) 后退 `distance` 米.

    target_7d: [x, y, z, qw, qx, qy, qz] (world/base frame).
    返回: 同 orientation, 位置沿 -TCP_z 退 distance 米的新 7D pose.
    用途: slow_approach 的 pre-grasp / via waypoint. camera 视角与 grasp 完全一致,
    只是 zoom out; 可达性优于 grasp (离 home seed 更近).
    """
    pos = np.asarray(target_7d[:3], dtype=np.float64)
    qw, qx, qy, qz = target_7d[3], target_7d[4], target_7d[5], target_7d[6]
    # TCP-z (local [0,0,1]) rotated to world by quat (w,x,y,z):
    tcp_z_world = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ])
    new_pos = pos - distance * tcp_z_world      # 退 = 沿 -TCP_z
    return np.concatenate([new_pos, target_7d[3:7]])


class ClosedLoopSkills(RobotSkills):
    """Closed-loop skills with observation feedback.

    Extends RobotSkills with methods that observe-plan-verify in a loop.
    All existing RobotSkills methods (move_to, gripper_action, etc.)
    remain available via inheritance.
    """

    def __init__(
        self,
        planner_manager: PlannerManager,
        observe_func: Callable[..., ObservationMessage],
        robot_cfg: Optional[Union[dict, DictConfig]] = None,
    ):
        """
        Args:
            planner_manager: Motion planning manager.
            observe_func: Callback that returns ObservationMessage.
                          Must accept keyword arg ``include_visuals: bool``.
                          Typically ``env.get_observation``.
            robot_cfg: Optional robot configuration.
        """
        super().__init__(planner_manager, robot_cfg)
        self._observe = observe_func

    # =========================================================
    # Verification Helpers
    # =========================================================

    def check_single_arm_grasp(
        self,
        arm: ArmSide,
        threshold_dist: float = 0.01,
        threshold_height: float = 0.05,
        obs: Optional["ObservationMessage"] = None,
    ) -> bool:
        """Check if a single arm successfully grasped and lifted cloth.

        Criteria: nearest cloth point is close to EE AND above threshold height.
        If obs is not provided, observe internally.
        """
        if obs is None:
            obs = self._observe(include_visuals=False)
        cloth_pos = obs.pcd_cloth_truth

        if cloth_pos is None or len(cloth_pos) == 0:
            logger.warning(f"[{arm}] No cloth data for check.")
            return False

        ee_pos = obs.eef_pose_left[:3] if arm == ArmSide.LEFT else obs.eef_pose_right[:3]

        dists = np.linalg.norm(cloth_pos - ee_pos, axis=1)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        nearest_point = cloth_pos[min_dist_idx]

        is_close = min_dist < threshold_dist
        is_lifted = nearest_point[2] > threshold_height

        logger.info(
            f"[{arm} Check] Min Dist: {min_dist:.4f}m, "
            f"Point Height: {nearest_point[2]:.4f}m -> "
            f"{'OK' if is_close and is_lifted else 'FAIL'}"
        )
        return is_close and is_lifted

    def check_cloth_released(
        self,
        arm: ArmSide,
        release_dist_threshold: float = 0.03,
        table_height_threshold: float = 0.08,
    ) -> bool:
        """Check if cloth was successfully released (not stuck to gripper).

        Opposite logic of check_single_arm_grasp:
        Released = cloth far from EE OR cloth at table height.

        Args:
            arm: Which arm to check.
            release_dist_threshold: Distance above which cloth is considered released.
            table_height_threshold: Height below which cloth is at table.
        Returns:
            True = released, False = stuck/penetration.
        """
        obs = self._observe(include_visuals=False)
        cloth_pos = obs.pcd_cloth_truth

        if cloth_pos is None or len(cloth_pos) == 0:
            logger.warning(f"[{arm} Release] No cloth data.")
            return False

        ee_pos = obs.eef_pose_left[:3] if arm == ArmSide.LEFT else obs.eef_pose_right[:3]

        dists = np.linalg.norm(cloth_pos - ee_pos, axis=1)
        min_dist = np.min(dists)
        nearest_z = cloth_pos[np.argmin(dists), 2]

        not_stuck = min_dist > release_dist_threshold
        at_table = nearest_z < table_height_threshold

        released = not_stuck or at_table
        logger.info(
            f"[{arm} Release] dist_to_ee={min_dist:.4f}m "
            f"(thresh={release_dist_threshold}), "
            f"nearest_z={nearest_z:.4f}m "
            f"(thresh={table_height_threshold}) -> "
            f"{'RELEASED' if released else 'STUCK/PENETRATION'}"
        )
        return released

    def check_bimanual_grasp(
        self,
        obs: Optional["ObservationMessage"] = None,
        threshold_dist: float = 0.04,
        threshold_height: float = 0.05,
    ) -> Tuple[bool, bool]:
        """Check both arms' grasp status. Returns (l_ok, r_ok).

        Convenience wrapper that observes once (or uses provided obs)
        and checks both arms with consistent thresholds.
        """
        if obs is None:
            obs = self._observe(include_visuals=False)
        l_ok = self.check_single_arm_grasp(
            ArmSide.LEFT, threshold_dist=threshold_dist,
            threshold_height=threshold_height, obs=obs,
        )
        r_ok = self.check_single_arm_grasp(
            ArmSide.RIGHT, threshold_dist=threshold_dist,
            threshold_height=threshold_height, obs=obs,
        )
        return l_ok, r_ok

    def get_actual_grasp_indices(
        self,
        obs: Optional["ObservationMessage"] = None,
    ) -> Tuple[int, int]:
        """Return the cloth vertex indices closest to each EE.

        Useful for querying what was actually grasped (may differ from
        the selected vertex indices due to gripper slip / cloth deformation).
        Returns (left_idx, right_idx).
        """
        if obs is None:
            obs = self._observe(include_visuals=False)
        cloth_pos = obs.pcd_cloth_truth
        if cloth_pos is None or len(cloth_pos) == 0:
            logger.warning("[get_actual_grasp_indices] No cloth data.")
            return -1, -1
        ee_l = obs.eef_pose_left[:3]
        ee_r = obs.eef_pose_right[:3]
        l_idx = int(np.argmin(np.linalg.norm(cloth_pos - ee_l, axis=1)))
        r_idx = int(np.argmin(np.linalg.norm(cloth_pos - ee_r, axis=1)))
        return l_idx, r_idx

    # =========================================================
    # Closed-Loop Grasp Skills (Generator pattern)
    # =========================================================

    def robust_bimanual_grasp(
        self,
        get_targets_func: Callable,
        max_retries: int = 3,
        open_gripper_value: float = 0.7,
        close_gripper_value: float = 0.0,
        # --- Test lift parameters ---
        test_lift_height: float = 0.05,
        check_threshold_height: float = 0.01,
        check_threshold_dist: float = 0.01,
        # --- Approach constraint parameters ---
        pose_constraint: Optional[List] = None,
        pose_constraint_frame: str = "world",
        retry_pose_constraint: Optional[List] = None,
        retry_constraint_frame: str = "world",
        ee_link: str = "tcp_link_bottom",
        # --- Adaptive z_offset ---
        grasp_depth: float = 0.0,   # below the item when grasp  TODO：set a value is better
        retry_depth_step: float = 0.0,   # retry reduce height
        retry_inward_step: float = 0.0,  # per-retry XY shift toward cloth centroid
        # --- Lift behavior ---
        selective_lift: bool = True,
        y_retract_offset: float = 0.0,
        # --- Gripper close duration ---
        grasp_settle_time: float = 0.8,
        # --- Async close (打破双臂 state shortcut) ---
        # True + delay > 0: approach 用 synchronize=False, close 错开 delay 秒
        # delay 值由 task 层随机采样后传入; leader 由内部随机选 L/R.
        async_close: bool = False,
        async_close_delay: float = 0.0,
        # --- Slow approach via-point (approach 末段慢速 + via 扰动) ---
        # True: approach 分两段: home→via→grasp; via_distance 由 task 层采样传入.
        # 第一段速度 = planner_kwargs["velocity_scale"] (默认 0.15),
        # 第二段速度 = slow_approach_final_vel_scale (默认 0.08, 慢约 2x).
        # via 构造: 沿 grasp orientation 的 -TCP_z (approach 反向) 退 via_distance 米.
        #   → camera 视角和 grasp 时完全一致, 只是 zoom out
        #   → 可达性: pre_grasp 离 home 更近, 比 grasp 本身更易 IK
        slow_approach: bool = False,
        slow_approach_via_distance: float = 0.0,
        slow_approach_final_vel_scale: float = 0.08,
        # --- Planner backend kwargs (velocity_scale, etc.) ---
        **planner_kwargs,
    ) -> Generator[ExecutionStep, None, bool]:
        """Bimanual closed-loop grasp with retry.

        Flow per attempt:
          1. Observe + sync
          2. Get targets via callback, for retry to get target
          3. Approach (both arms, or retry failed arm only)
          4. Close grippers
          5. Test lift (optional selective: only lift untested arm)
          6. Verify with check_single_arm_grasp
          7. On failure: retry with locked indices + deeper z_offset

        Args:
            get_targets_func: Callback ``(obs, specific_indices, z_offset=...)``
                -> ``(l_pose_7d, r_pose_7d, l_idx, r_idx)``.
                First call: ``specific_indices=None`` (full decision).
                Retry: ``specific_indices=(l_idx, r_idx)`` (lock indices, update coords).
            max_retries: Maximum retry count (total attempts = max_retries + 1).
            open_gripper_value: Gripper value for open state.
            close_gripper_value: Gripper value for closed state.
            test_lift_height: How high to lift for grasp verification (m).
            check_threshold_height: Min height for cloth to be considered lifted (m).
            pose_constraint: Pose constraint for first approach. Default [1,1,1,0,0,0].
            pose_constraint_frame: Frame for initial constraint.
            retry_pose_constraint: Pose constraint for retry approach. Default [1,1,1,1,0,1].
            retry_constraint_frame: Frame for retry constraint.
            ee_link: End-effector link name.
            grasp_depth: Initial grasp depth below target (m), passed as z_offset to callback.
            retry_depth_step: Per-retry depth increment (m). 0 = no adaptive offset.
            retry_inward_step: Per-retry XY shift (m) toward cloth centroid. 0 = disabled.
            selective_lift: If True, only lift the failing arm on retry (successful arm joint-locks).
            y_retract_offset: Y-axis retract during test lift (m). Left arm y-=, right arm y+=.
            grasp_settle_time: Duration to hold gripper closed (s).
            **planner_kwargs: Passed to move_dual_arm_to (e.g. velocity_scale).

        Returns:
            bool: True if both arms grasped successfully.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]
        if retry_pose_constraint is None:
            retry_pose_constraint = pose_constraint

        left_grasped = False
        right_grasped = False
        current_indices = None

        # Detect whether callback accepts l_ok/r_ok kwargs (backward compat)
        import inspect
        sig_params = inspect.signature(get_targets_func).parameters
        accepts_ok_flags = ('l_ok' in sig_params and 'r_ok' in sig_params) or \
                           any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())

        for attempt in range(max_retries + 1):
            logger.info(
                f"--- [Robust Grasp] Attempt {attempt}/{max_retries} "
                f"(L:{left_grasped}, R:{right_grasped}) ---"
            )

            # Retry 前先开失败臂夹爪, 让 cloth 回落, 避免脏观察.
            # 不加这段 observe 会在失败臂还夹着布时进行, cloth_pos 里失败点
            # 周围 vtx 都被拽到 gripper 高度, 污染 candidate 选择.
            if attempt > 0:
                l_grip = close_gripper_value if left_grasped else open_gripper_value
                r_grip = close_gripper_value if right_grasped else open_gripper_value
                yield ExecutionStep(
                    plan=self.gripper_bimanual_action(
                        l_grip, r_grip, duration=0.3),
                    settle_time=0.3,
                    description=f"retry_release_failed_{attempt}",
                )

            # 1. Observe + sync
            obs = self._observe(include_visuals=False)
            self.sync_state(obs)

            # 2. Compute adaptive z_offset
            z_offset = grasp_depth + (attempt * retry_depth_step) if retry_depth_step != 0.0 else None

            # 3. Get targets via callback
            try:
                kwargs = {}
                if z_offset is not None:
                    kwargs["z_offset"] = z_offset
                if accepts_ok_flags:
                    kwargs["l_ok"] = left_grasped
                    kwargs["r_ok"] = right_grasped
                l_target, r_target, l_idx, r_idx = get_targets_func(
                    obs, current_indices, **kwargs
                )
                current_indices = (l_idx, r_idx)
            except Exception as e:
                logger.error(f"Target calculation failed: {e}")
                return False

            # Retry inward shift: on each retry move targets toward cloth centroid.
            if retry_inward_step > 0.0 and attempt > 0:
                inward_dist = attempt * retry_inward_step
                centroid_xy = np.mean(obs.pcd_cloth_truth[:, :2], axis=0)
                def _shift(pose_7d, dist, cxy):
                    p = np.array(pose_7d, dtype=float)
                    v = cxy - p[:2]
                    n = float(np.linalg.norm(v))
                    if n > 1e-4:
                        p[:2] += dist * v / n
                    return p
                l_target = _shift(l_target, inward_dist, centroid_xy)
                r_target = _shift(r_target, inward_dist, centroid_xy)
                logger.info(
                    f"[Robust Grasp] inward shift attempt={attempt}: "
                    f"dist={inward_dist:.3f}m, centroid={centroid_xy}"
                )

            # =========================================
            # A. Approach
            # =========================================
            plan_approach = None
            async_active = (
                async_close
                and async_close_delay > 1e-6
                and not left_grasped
                and not right_grasped
            )

            if async_active:
                # 方案 C: approach synchronize=False (两臂自然错开到位),
                #         close 段并发执行但 follower 前缀 hold `async_close_delay` 秒,
                #         总时长 ≈ grasp_settle_time + async_close_delay (不翻倍串行).
                # Leader 内部随机选 L/R (不由调用方指定).
                leader = "left" if np.random.random() < 0.5 else "right"

                # --- 双臂 approach (synchronize=False → 自然错开到位) ---
                kwargs_no_sync = dict(planner_kwargs)
                kwargs_no_sync["synchronize"] = False
                if slow_approach and slow_approach_via_distance > 1e-6:
                    logger.success(
                        f"[slow_approach+async] via_distance={slow_approach_via_distance:.3f}m "
                        f"(沿 grasp -TCP_z 后退), final_vel={slow_approach_final_vel_scale}, "
                        f"leader={leader}, close_delay={async_close_delay:.2f}s, debug_ik=True"
                    )
                    # via = grasp 沿 -TCP_z 后退 via_distance 米, orientation 不变 → camera
                    # 视角与 grasp 一致, 可达性优于 grasp. via pose_constraint=[1,1,1,0,0,1]
                    # (local): xyz + local rz 锁, rx/ry 自由. end_first_seed=True: grasp 先解,
                    # via 用 grasp_sol 作 seed → via rx/ry 自然贴近 grasp.
                    via_l = _pre_grasp_along_approach(np.asarray(l_target, dtype=np.float64),
                                                      slow_approach_via_distance)
                    via_r = _pre_grasp_along_approach(np.asarray(r_target, dtype=np.float64),
                                                      slow_approach_via_distance)
                    via_constraint = [1, 1, 1, 0, 0, 1]
                    first_vel = kwargs_no_sync.pop("velocity_scale", 0.3)
                    plan_approach_async = self.move_dual_arm_to(
                        left_target_pose=[via_l, np.asarray(l_target)],
                        right_target_pose=[via_r, np.asarray(r_target)],
                        goal_type="waypoints",
                        left_gripper=open_gripper_value,
                        right_gripper=open_gripper_value,
                        pose_constraint=[via_constraint, pose_constraint],
                        pose_constraint_frame="local",
                        ee_link=ee_link,
                        velocity_scale=[first_vel, slow_approach_final_vel_scale],
                        end_first_seed=True,
                        debug_ik=True,
                        **kwargs_no_sync,
                    )
                else:
                    plan_approach_async = self.move_dual_arm_to(
                        left_target_pose=l_target,
                        right_target_pose=r_target,
                        left_gripper=open_gripper_value,
                        right_gripper=open_gripper_value,
                        pose_constraint=pose_constraint,
                        pose_constraint_frame=pose_constraint_frame,
                        ee_link=ee_link,
                        **kwargs_no_sync,
                    )
                yield ExecutionStep(
                    plan=plan_approach_async, settle_time=0.1,
                    description=f"async_approach_{attempt}",
                )

                # --- 两臂并发 close, follower 前缀 hold async_close_delay 秒 ---
                yield ExecutionStep(
                    plan=self.gripper_bimanual_action(
                        close_gripper_value, close_gripper_value,
                        duration=grasp_settle_time,
                        stagger_leader=leader,
                        stagger_delay=async_close_delay,
                    ),
                    settle_time=0.3,
                    description=f"async_close_leader={leader}_delay={async_close_delay:.2f}",
                )
                plan_approach = None   # 跳过下方同步 approach + bimanual close

            elif not left_grasped and not right_grasped:
                # Both arms approach
                if slow_approach and slow_approach_via_distance > 1e-6:
                    logger.success(
                        f"[slow_approach] via_distance={slow_approach_via_distance:.3f}m "
                        f"(沿 grasp -TCP_z 后退), final_vel={slow_approach_final_vel_scale}, "
                        f"debug_ik=True"
                    )
                    # 两段 waypoint: home→via→grasp.
                    # via = grasp 沿 -TCP_z 后退 via_distance 米, orientation 不变.
                    #   好处: camera 视角一致(看得到 corner), 可达性优于 grasp (离 home seed 近).
                    # via pose_constraint = [1,1,1,0,0,1] (local frame):
                    #   - xyz 锁死位置 + local rz 锁 → 夹爪中轴不扭
                    #   - rx (yaw) 自由, ry (pitch) 自由 → IK 压力最小
                    # end_first_seed=True: grasp IK 先解, 用 grasp_sol 作 via seed
                    #   → via rx/ry 自然贴近 grasp (minimal wrist rotation).
                    via_l = _pre_grasp_along_approach(np.asarray(l_target, dtype=np.float64),
                                                      slow_approach_via_distance)
                    via_r = _pre_grasp_along_approach(np.asarray(r_target, dtype=np.float64),
                                                      slow_approach_via_distance)
                    via_constraint = [1, 1, 1, 0, 0, 1]  # 锁 xyz + local rz. 后面要锁 rx 改第 4 位 0→1.
                    # 复制 kwargs 避免污染后续 retry/lift 分支的 velocity_scale
                    _kw = dict(planner_kwargs)
                    first_vel = _kw.pop("velocity_scale", 0.3)
                    plan_approach = self.move_dual_arm_to(
                        left_target_pose=[via_l, np.asarray(l_target)],
                        right_target_pose=[via_r, np.asarray(r_target)],
                        goal_type="waypoints",
                        left_gripper=open_gripper_value,
                        right_gripper=open_gripper_value,
                        pose_constraint=[via_constraint, pose_constraint],
                        pose_constraint_frame="local",
                        ee_link=ee_link,
                        velocity_scale=[first_vel, slow_approach_final_vel_scale],
                        end_first_seed=True,
                        debug_ik=True,   # 打印 via/grasp IK 结果 + FK 验证
                        **_kw,
                    )
                else:
                    plan_approach = self.move_dual_arm_to(
                        left_target_pose=l_target, right_target_pose=r_target,
                        left_gripper=open_gripper_value,
                        right_gripper=open_gripper_value,
                        pose_constraint=pose_constraint,
                        pose_constraint_frame=pose_constraint_frame,
                        ee_link=ee_link,
                        **planner_kwargs,
                    )

            elif left_grasped and not right_grasped:
                # Left OK, retry right only
                yield ExecutionStep(
                    plan=self.gripper_action(open_gripper_value, "right", duration=0.1),
                    settle_time=0.1, description="open_right_for_retry",
                )
                obs = self._observe(include_visuals=False)
                self.sync_state(obs)
                curr_l_joints = obs.joint_left
                plan_approach = self.move_dual_arm_to(
                    left_target_pose=curr_l_joints, left_goal_type="joint",
                    right_target_pose=r_target, right_goal_type="pose",
                    left_gripper=close_gripper_value,
                    right_gripper=open_gripper_value,
                    pose_constraint=retry_pose_constraint,
                    pose_constraint_frame=retry_constraint_frame,
                    ee_link=ee_link,
                    **planner_kwargs,
                )

            elif not left_grasped and right_grasped:
                # Right OK, retry left only
                yield ExecutionStep(
                    plan=self.gripper_action(open_gripper_value, "left", duration=0.1),
                    settle_time=0.1, description="open_left_for_retry",
                )
                obs = self._observe(include_visuals=False)
                self.sync_state(obs)
                curr_r_joints = obs.joint_right
                plan_approach = self.move_dual_arm_to(
                    left_target_pose=l_target, left_goal_type="pose",
                    right_target_pose=curr_r_joints, right_goal_type="joint",
                    left_gripper=open_gripper_value,
                    right_gripper=close_gripper_value,
                    pose_constraint=retry_pose_constraint,
                    pose_constraint_frame=retry_constraint_frame,
                    ee_link=ee_link,
                    **planner_kwargs,
                )

            if plan_approach:
                yield ExecutionStep(
                    plan=plan_approach, settle_time=0.2,
                    description=f"approach_{attempt}",
                )

            # =========================================
            # B. Grasp (close both grippers)
            # =========================================
            if not async_active:
                yield ExecutionStep(
                    plan=self.gripper_bimanual_action(
                        close_gripper_value, close_gripper_value,
                        duration=grasp_settle_time,
                    ),
                    settle_time=0.3, description="grasp_close",
                )
            # async_active 路径已在 A 段完成 staggered close, 跳过 bimanual close

            # =========================================
            # C. Test lift
            # =========================================
            obs_for_lift = self._observe(include_visuals=False)
            self.sync_state(obs_for_lift)

            l_lift = l_target[:3].copy()
            l_lift[2] += test_lift_height
            r_lift = r_target[:3].copy()
            r_lift[2] += test_lift_height

            # Optional Y-axis retract
            if y_retract_offset > 0:
                l_lift[1] -= y_retract_offset
                r_lift[1] += y_retract_offset

            if selective_lift and (left_grasped != right_grasped):
                # Only lift the arm being tested
                if left_grasped and not right_grasped:
                    plan_lift = self.move_dual_arm_to(
                        left_target_pose=obs_for_lift.joint_left,
                        left_goal_type="joint",
                        right_target_pose=r_lift, right_goal_type="pose",
                        left_gripper=close_gripper_value,
                        pose_constraint=[1, 1, 1, 0, 0, 0],
                        ee_link=ee_link,
                        **planner_kwargs,
                    )
                else:  # not left_grasped and right_grasped
                    plan_lift = self.move_dual_arm_to(
                        left_target_pose=l_lift, left_goal_type="pose",
                        right_target_pose=obs_for_lift.joint_right,
                        right_goal_type="joint",
                        right_gripper=close_gripper_value,
                        pose_constraint=[1, 1, 1, 0, 0, 0],
                        ee_link=ee_link,
                        **planner_kwargs,
                    )
            else:
                # Both arms lift
                plan_lift = self.move_dual_arm_to(
                    left_target_pose=l_lift, right_target_pose=r_lift,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link=ee_link,
                    **planner_kwargs,
                )

            yield ExecutionStep(
                plan=plan_lift, settle_time=0.5, description="test_lift",
            )

            # =========================================
            # D. Check
            # =========================================
            left_grasped, right_grasped = self.check_bimanual_grasp(
                threshold_dist=check_threshold_dist,
                threshold_height=check_threshold_height,
            )

            if left_grasped and right_grasped:
                logger.success(">>> Robust Grasp Success!")
                return True

            if attempt == max_retries:
                logger.error(">>> Robust Grasp Failed after max retries.")
                return False

        return False

    # =========================================================
    # Single-Arm Closed-Loop Grasp
    # =========================================================

    def robust_single_grasp(
        self,
        get_target_func: Callable,
        arm: ArmSide,
        other_arm_mode: Literal["hold", "hold_grasp", "home"] = "hold",
        max_retries: int = 3,
        open_gripper_value: float = 0.7,
        close_gripper_value: float = 0.0,
        test_lift_height: float = 0.05,
        check_threshold_height: float = 0.01,
        check_threshold_dist: float = 0.01,
        ee_link: str = "tcp_link_bottom",
        # --- Approach constraint ---
        pose_constraint: Optional[List] = None,
        pose_constraint_frame: str = "world",
        # --- Adaptive grasp depth ---
        grasp_depth: float = 0.0,
        retry_depth_step: float = 0.0,
        retry_inward_step: float = 0.0,  # per-retry XY shift toward cloth centroid
        grasp_settle_time: float = 0.8,
    ) -> Generator[ExecutionStep, None, bool]:
        """Single-arm closed-loop grasp with retry.

        Flow: approach -> close -> re-observe+sync -> test_lift -> verify.
        Other arm behavior controlled by other_arm_mode:
          - "hold": lock current joints, gripper unchanged
          - "hold_grasp": lock current joints, gripper closed

        Args:
            get_target_func: Callback ``(obs, locked_index_or_None, z_offset=...)``
                -> ``(pose_7d, vertex_idx)``.
            arm: Which arm grasps.
            other_arm_mode: Other arm behavior.
            max_retries: Maximum retry count.
            open_gripper_value: Gripper value for open.
            close_gripper_value: Gripper value for closed.
            test_lift_height: Lift height for verification (m).
            check_threshold_height: Min height threshold for verification.
            ee_link: End-effector link name.
            pose_constraint: Pose constraint for approach. Default [1,1,1,0,0,0].
            pose_constraint_frame: Frame for approach constraint.
            grasp_depth: Initial grasp depth below target (m), passed as z_offset to callback.
            retry_depth_step: Per-retry depth increment (m).
            retry_inward_step: Per-retry XY shift (m) toward cloth centroid. 0 = disabled.
            grasp_settle_time: Gripper close hold duration (s).

        Returns:
            bool: True if grasp succeeded.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]
        locked_index = None
        other_arm = arm.opposite

        for attempt in range(max_retries + 1):
            logger.info(f"--- [Single Grasp {arm}] Attempt {attempt}/{max_retries} ---")

            # 1. Observe + sync
            obs = self._observe(include_visuals=False)
            self.sync_state(obs)

            # Adaptive Z offset
            z_offset = grasp_depth + (attempt * retry_depth_step) if retry_depth_step != 0.0 else None

            # 2. Get target via callback
            try:
                if z_offset is not None:
                    target_pose, vertex_idx = get_target_func(obs, locked_index, z_offset=z_offset)
                else:
                    target_pose, vertex_idx = get_target_func(obs, locked_index)
                locked_index = vertex_idx
            except Exception as e:
                logger.error(f"[Single Grasp {arm}] Target calculation failed: {e}")
                return False

            # Retry inward shift: on each retry move target toward cloth centroid.
            if retry_inward_step > 0.0 and attempt > 0:
                inward_dist = attempt * retry_inward_step
                centroid_xy = np.mean(obs.pcd_cloth_truth[:, :2], axis=0)
                p = np.array(target_pose, dtype=float)
                v = centroid_xy - p[:2]
                n = float(np.linalg.norm(v))
                if n > 1e-4:
                    p[:2] += inward_dist * v / n
                target_pose = p
                logger.info(
                    f"[Single Grasp {arm}] inward shift attempt={attempt}: "
                    f"dist={inward_dist:.3f}m, centroid={centroid_xy}"
                )

            # 3. Approach: active arm to target, other arm behavior by mode
            if other_arm_mode == "home":
                n = len(self.planner_manager.left_joint_names if other_arm == ArmSide.LEFT
                        else self.planner_manager.right_joint_names)
                other_joints = np.zeros(n)
                other_gripper = 1.0
            else:
                other_joints = obs.joint_left if other_arm == ArmSide.LEFT else obs.joint_right
                other_gripper = close_gripper_value if other_arm_mode == "hold_grasp" else None

            if arm == ArmSide.LEFT:
                plan_approach = self.move_dual_arm_to(
                    left_target_pose=target_pose, left_goal_type="pose",
                    right_target_pose=other_joints, right_goal_type="joint",
                    left_gripper=open_gripper_value, right_gripper=other_gripper,
                    pose_constraint=pose_constraint,
                    pose_constraint_frame=pose_constraint_frame,
                    ee_link=ee_link,
                )
            else:
                plan_approach = self.move_dual_arm_to(
                    left_target_pose=other_joints, left_goal_type="joint",
                    right_target_pose=target_pose, right_goal_type="pose",
                    left_gripper=other_gripper, right_gripper=open_gripper_value,
                    pose_constraint=pose_constraint,
                    pose_constraint_frame=pose_constraint_frame,
                    ee_link=ee_link,
                )
            yield ExecutionStep(
                plan=plan_approach, settle_time=0.2,
                description=f"single_approach_{attempt}",
            )

            # 4. Close gripper
            yield ExecutionStep(
                plan=self.gripper_action(
                    close_gripper_value, arm, duration=grasp_settle_time,
                ),
                settle_time=0.3, description="single_grasp_close",
            )

            # 5. Re-observe + sync (critical: state may drift after close)
            obs_for_lift = self._observe(include_visuals=False)
            self.sync_state(obs_for_lift)

            # 6. Test lift
            lift_pos = target_pose[:3].copy()
            lift_pos[2] += test_lift_height
            if other_arm_mode == "home":
                n = len(self.planner_manager.left_joint_names if other_arm == ArmSide.LEFT
                        else self.planner_manager.right_joint_names)
                other_joints_lift = np.zeros(n)
            else:
                other_joints_lift = (
                    obs_for_lift.joint_left if other_arm == ArmSide.LEFT
                    else obs_for_lift.joint_right
                )

            if arm == ArmSide.LEFT:
                plan_lift = self.move_dual_arm_to(
                    left_target_pose=lift_pos, left_goal_type="pose",
                    right_target_pose=other_joints_lift, right_goal_type="joint",
                    left_gripper=close_gripper_value, right_gripper=other_gripper,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link=ee_link,
                )
            else:
                plan_lift = self.move_dual_arm_to(
                    left_target_pose=other_joints_lift, left_goal_type="joint",
                    right_target_pose=lift_pos, right_goal_type="pose",
                    left_gripper=other_gripper, right_gripper=close_gripper_value,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link=ee_link,
                )
            yield ExecutionStep(
                plan=plan_lift, settle_time=0.5, description="single_test_lift",
            )

            # 7. Verify
            grasp_ok = self.check_single_arm_grasp(
                arm, threshold_height=check_threshold_height,
                threshold_dist=check_threshold_dist,
            )

            if grasp_ok:
                logger.success(f">>> Single Grasp {arm} Success!")
                return True

            # 8. Failed -> open gripper for retry
            if attempt < max_retries:
                logger.warning(f"[Single Grasp {arm}] Failed, retrying...")
                yield ExecutionStep(
                    plan=self.gripper_action(open_gripper_value, arm, duration=0.1),
                    settle_time=0.1, description="open_for_retry",
                )
            else:
                logger.error(
                    f">>> Single Grasp {arm} Failed after {max_retries} retries."
                )

        return False

    # =========================================================
    # Single-Arm Closed-Loop Place
    # =========================================================

    def robust_place_single(
        self,
        place_pose: np.ndarray,
        arm: ArmSide,
        other_arm_mode: Literal["hold", "hold_grasp", "home"] = "hold",
        max_retries: int = 2,
        open_gripper_value: float = 1.0,
        close_gripper_value: float = 0.0,
        retreat_height: float = 0.10,
        ee_link: str = "tcp_link_bottom",
        pose_constraint: Optional[List] = None,
        pose_constraint_frame: str = "world",
    ) -> Generator[ExecutionStep, None, bool]:
        """Single-arm closed-loop place with adhesion detection.

        Flow: move to place -> open gripper -> retreat -> verify release.
        On adhesion/penetration: returns False immediately (no recovery).

        Args:
            place_pose: Target placement [x,y,z] or [x,y,z,qw,qx,qy,qz].
            arm: Which arm places.
            other_arm_mode: Other arm behavior ("hold" or "hold_grasp").
            max_retries: Maximum retry count.
            open_gripper_value: Gripper value for release.
            close_gripper_value: Gripper value for hold_grasp mode.
            retreat_height: How high to retreat after release (m).
            ee_link: End-effector link name.
            pose_constraint: Pose constraint for place move. Default [1,1,1,0,0,0].
            pose_constraint_frame: Frame for place constraint.

        Returns:
            True = released, False = adhesion/penetration.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]
        other_arm = arm.opposite
        place_pos = np.array(place_pose)
        if len(place_pos) == 3:
            place_pos = np.concatenate([place_pos, [1.0, 0.0, 0.0, 0.0]])

        for attempt in range(max_retries + 1):
            logger.info(f"--- [Place {arm}] Attempt {attempt}/{max_retries} ---")

            # 1. Observe + sync
            obs = self._observe(include_visuals=False)
            self.sync_state(obs)

            # 2. Move to place position
            if other_arm_mode == "home":
                n = len(self.planner_manager.left_joint_names if other_arm == ArmSide.LEFT
                        else self.planner_manager.right_joint_names)
                other_joints = np.zeros(n)
                other_gripper = 1.0
            else:
                other_joints = (
                    obs.joint_left if other_arm == ArmSide.LEFT else obs.joint_right
                )
                other_gripper = close_gripper_value if other_arm_mode == "hold_grasp" else None

            if arm == ArmSide.LEFT:
                plan_move = self.move_dual_arm_to(
                    left_target_pose=place_pos, left_goal_type="pose",
                    right_target_pose=other_joints, right_goal_type="joint",
                    right_gripper=other_gripper,
                    pose_constraint=pose_constraint,
                    pose_constraint_frame=pose_constraint_frame,
                    ee_link=ee_link,
                )
            else:
                plan_move = self.move_dual_arm_to(
                    left_target_pose=other_joints, left_goal_type="joint",
                    right_target_pose=place_pos, right_goal_type="pose",
                    left_gripper=other_gripper,
                    pose_constraint=pose_constraint,
                    pose_constraint_frame=pose_constraint_frame,
                    ee_link=ee_link,
                )
            yield ExecutionStep(
                plan=plan_move, settle_time=0.2,
                description=f"place_move_{attempt}",
            )

            # 3. Open gripper
            yield ExecutionStep(
                plan=self.gripper_action(open_gripper_value, arm, duration=0.5),
                settle_time=0.5, description="place_open",
            )

            # 4. Retreat upward
            obs_retreat = self._observe(include_visuals=False)
            self.sync_state(obs_retreat)

            retreat_pos = place_pos[:3].copy()
            retreat_pos[2] += retreat_height
            if other_arm_mode == "home":
                n = len(self.planner_manager.left_joint_names if other_arm == ArmSide.LEFT
                        else self.planner_manager.right_joint_names)
                other_joints_retreat = np.zeros(n)
            else:
                other_joints_retreat = (
                    obs_retreat.joint_left if other_arm == ArmSide.LEFT
                    else obs_retreat.joint_right
                )

            if arm == ArmSide.LEFT:
                plan_retreat = self.move_dual_arm_to(
                    left_target_pose=retreat_pos, left_goal_type="pose",
                    right_target_pose=other_joints_retreat, right_goal_type="joint",
                    right_gripper=other_gripper,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link=ee_link,
                )
            else:
                plan_retreat = self.move_dual_arm_to(
                    left_target_pose=other_joints_retreat, left_goal_type="joint",
                    right_target_pose=retreat_pos, right_goal_type="pose",
                    left_gripper=other_gripper,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link=ee_link,
                )
            yield ExecutionStep(
                plan=plan_retreat, settle_time=0.3, description="place_retreat",
            )

            # 5. Verify release
            released = self.check_cloth_released(arm)
            if released:
                logger.success(f">>> Place {arm} Success!")
                return True

            # Adhesion/penetration -> fail immediately
            logger.error(
                f">>> Place {arm} Failed: adhesion/penetration detected "
                f"(attempt {attempt})"
            )
            return False

        return False

    # =========================================================
    # Bimanual Closed-Loop Place
    # =========================================================

    def robust_place_bimanual(
        self,
        left_place_pose: np.ndarray,
        right_place_pose: np.ndarray,
        max_retries: int = 2,
        open_gripper_value: float = 1.0,
        retreat_height: float = 0.10,
        ee_link: str = "tcp_link_bottom",
        pose_constraint: Optional[List] = None,
        pose_constraint_frame: str = "world",
    ) -> Generator[ExecutionStep, None, bool]:
        """Bimanual closed-loop place with adhesion detection.

        Returns False if either arm has adhesion/penetration.

        Args:
            left_place_pose: Left arm target [x,y,z] or [x,y,z,qw,qx,qy,qz].
            right_place_pose: Right arm target.
            max_retries: Maximum retry count.
            open_gripper_value: Gripper value for release.
            retreat_height: Retreat height after release (m).
            ee_link: End-effector link name.
            pose_constraint: Pose constraint for place move. Default [1,1,1,0,0,0].
            pose_constraint_frame: Frame for place constraint.

        Returns:
            True = both arms released, False = at least one stuck.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]
        l_pose = np.array(left_place_pose)
        r_pose = np.array(right_place_pose)
        if len(l_pose) == 3:
            l_pose = np.concatenate([l_pose, [1.0, 0.0, 0.0, 0.0]])
        if len(r_pose) == 3:
            r_pose = np.concatenate([r_pose, [1.0, 0.0, 0.0, 0.0]])

        for attempt in range(max_retries + 1):
            logger.info(f"--- [Bimanual Place] Attempt {attempt}/{max_retries} ---")

            # 1. Observe + sync
            obs = self._observe(include_visuals=False)
            self.sync_state(obs)

            # 2. Move both arms to place positions
            plan_move = self.move_dual_arm_to(
                left_target_pose=l_pose, right_target_pose=r_pose,
                pose_constraint=pose_constraint,
                pose_constraint_frame=pose_constraint_frame,
                ee_link=ee_link,
            )
            yield ExecutionStep(
                plan=plan_move, settle_time=0.2,
                description=f"biplace_move_{attempt}",
            )

            # 3. Open both grippers
            yield ExecutionStep(
                plan=self.gripper_bimanual_action(
                    open_gripper_value, open_gripper_value, duration=0.5,
                ),
                settle_time=0.5, description="biplace_open",
            )

            # 4. Retreat
            obs_retreat = self._observe(include_visuals=False)
            self.sync_state(obs_retreat)

            l_retreat = l_pose[:3].copy()
            l_retreat[2] += retreat_height
            r_retreat = r_pose[:3].copy()
            r_retreat[2] += retreat_height

            plan_retreat = self.move_dual_arm_to(
                left_target_pose=l_retreat, right_target_pose=r_retreat,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link=ee_link,
            )
            yield ExecutionStep(
                plan=plan_retreat, settle_time=0.3, description="biplace_retreat",
            )

            # 5. Verify both arms
            l_ok = self.check_cloth_released(ArmSide.LEFT)
            r_ok = self.check_cloth_released(ArmSide.RIGHT)

            if l_ok and r_ok:
                logger.success(">>> Bimanual Place Success!")
                return True

            # Any adhesion -> fail immediately
            stuck_arms = []
            if not l_ok:
                stuck_arms.append("LEFT")
            if not r_ok:
                stuck_arms.append("RIGHT")
            logger.error(
                f">>> Bimanual Place Failed: {', '.join(stuck_arms)} stuck "
                f"(attempt {attempt})"
            )
            return False

        return False

    # =========================================================
    # Weld-based Grasping (constraint-based, no physical contact)
    # =========================================================

    def weld_bimanual_grasp(
        self,
        env,
        get_targets_func: Callable,
        velocity_scale: float = 0.3,
        ee_link: str = "tcp_link_bottom",
        pose_constraint=None,
        settle_time: float = 0.4,
    ) -> Generator[ExecutionStep, None, bool]:
        """Move both arms to target poses then weld the nearest cloth vertices.

        Unlike :py:meth:`robust_bimanual_grasp`, this method pins cloth
        vertices to the EE via :py:meth:`env.weld_grasp` instead of
        physically closing the gripper.  The cloth is then dragged by the
        kinematic weld on every subsequent physics step.

        Args:
            env:              Environment with ``weld_grasp`` / ``weld_release``.
            get_targets_func: ``(obs) -> (l_pose_7d, r_pose_7d, l_idx, r_idx)``
            velocity_scale:   Motion speed (0–1).
            ee_link:          TCP link name.
            pose_constraint:  6-DOF constraint mask.
            settle_time:      Settle duration after reaching the grasp pose.

        Yields:
            :class:`~common.types.ExecutionStep` objects.

        Returns:
            ``True`` on success.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]

        obs = self._observe(include_visuals=False)
        l_target, r_target, l_idx, r_idx = get_targets_func(obs)

        plan_approach = self.move_dual_arm_to(
            left_target_pose=l_target,
            right_target_pose=r_target,
            pose_constraint=pose_constraint,
            ee_link=ee_link,
            velocity_scale=velocity_scale,
        )
        yield ExecutionStep(plan=plan_approach, settle_time=settle_time,
                            description="weld_bimanual_approach")

        env.weld_grasp(l_idx, "left")
        env.weld_grasp(r_idx, "right")
        logger.info(f"[WeldGrasp] Bimanual welded: L_idx={l_idx}, R_idx={r_idx}")
        return True

    def weld_single_grasp(
        self,
        env,
        arm: ArmSide,
        get_target_func: Callable,
        velocity_scale: float = 0.3,
        ee_link: str = "tcp_link_bottom",
        pose_constraint=None,
        settle_time: float = 0.4,
    ) -> Generator[ExecutionStep, None, bool]:
        """Move one arm to a target pose then weld the nearest cloth vertex.

        Args:
            env:             Environment with ``weld_grasp`` / ``weld_release``.
            arm:             Which arm to use (:class:`~common.types.ArmSide`).
            get_target_func: ``(obs) -> (pose_7d, vertex_idx)``
            velocity_scale:  Motion speed (0–1).
            ee_link:         TCP link name.
            pose_constraint: 6-DOF constraint mask.
            settle_time:     Settle duration after reaching the grasp pose.

        Yields:
            :class:`~common.types.ExecutionStep` objects.

        Returns:
            ``True`` on success.
        """
        if pose_constraint is None:
            pose_constraint = [1, 1, 1, 0, 0, 0]

        obs = self._observe(include_visuals=False)
        target_pose, vertex_idx = get_target_func(obs)

        if arm == ArmSide.LEFT:
            other_joints = obs.joint_right.copy()
            plan_approach = self.move_dual_arm_to(
                left_target_pose=target_pose,
                right_target_pose=other_joints,
                right_goal_type="joint",
                pose_constraint=pose_constraint,
                ee_link=ee_link,
                velocity_scale=velocity_scale,
            )
        else:
            other_joints = obs.joint_left.copy()
            plan_approach = self.move_dual_arm_to(
                left_target_pose=other_joints,
                left_goal_type="joint",
                right_target_pose=target_pose,
                pose_constraint=pose_constraint,
                ee_link=ee_link,
                velocity_scale=velocity_scale,
            )
        yield ExecutionStep(plan=plan_approach, settle_time=settle_time,
                            description=f"weld_single_{arm.value}_approach")

        env.weld_grasp(vertex_idx, arm.value)
        logger.info(f"[WeldGrasp] Single {arm.value} welded to vertex {vertex_idx}")
        return True

    def weld_release_bimanual(self, env) -> None:
        """Release both arm weld constraints immediately (no motion plan)."""
        env.weld_release("left")
        env.weld_release("right")
        logger.info("[WeldGrasp] Bimanual weld released")

    def weld_release_single(self, env, arm: ArmSide) -> None:
        """Release one arm's weld constraint immediately (no motion plan)."""
        env.weld_release(arm.value)
        logger.info(f"[WeldGrasp] {arm.value} weld released")
