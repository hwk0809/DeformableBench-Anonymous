import os
import numpy as np
from loguru import logger

from deformablebench.bench.robot.planning.planner_manager import PlannerManager
from deformablebench.bench.common.types import (
    PlanResult, ArmSide, ObservationMessage, DualPlanResult,
    ExecutionStep, PlannerErrorCode, PlanningState, ArmMotionGoal,
)
from omegaconf import DictConfig
from typing import Optional, List, Dict, Union, Literal


class RobotSkills:
    def __init__(self, planner_manager: PlannerManager, robot_cfg: Optional[Union[Dict, DictConfig]] = None):
        self.planner_manager = planner_manager
        self.robot_cfg = robot_cfg or {}
        self.left_base_pose = self.planner_manager.left_base_pose
        self.right_base_pose = self.planner_manager.right_base_pose

    # =========================================================
    # 状态管理（委托给 PlannerManager 的规划工作状态）
    # =========================================================

    @property
    def planning_state(self) -> PlanningState:
        """访问规划器的工作状态副本（非真实状态）。

        查看 .source 可知当前是 "observed"(可信) 还是 "predicted"(前推)。
        """
        return self.planner_manager.planning_state

    def sync_state(self, obs: ObservationMessage):
        """从观测同步规划状态。每个规划阶段开头调用。

        调用后 planning_state.source = "observed"。
        仿真和真机使用相同机制，区别在于调用频率：
        - 仿真：每个 stage 开头同步一次，中间可连续 advance
        - 真机：每步前同步纠偏（保守），或每 N 步（激进）
        """
        self.planner_manager.sync_from_obs(obs)

    def _advance_state(self, plan: DualPlanResult):
        """从规划结果前推状态，用于连续规划。

        plan 成功后，状态推进到轨迹终点，下一次 plan 从这里开始，
        无需等待执行和获取新观测。使多步连续规划成为可能：
            plan1 → advance_state → plan2 → advance_state → 一起 yield 执行
        调用后 planning_state.source = "predicted"。
        """
        self.planner_manager.advance(plan)

    def _check_state_ready(self):
        """检查规划状态是否已初始化"""
        s = self.planning_state
        if s.left_joints is None or s.right_joints is None:
            raise RuntimeError("Planning state not initialized! Call 'sync_state(obs)' first.")

    # =========================================================
    # 参数解析 helper
    # =========================================================

    @staticmethod
    def _resolve_dual(shared, left_override, right_override):
        """解析 "共享 + 独立覆盖" 参数，返回 (left_val, right_val)。

        规则: override ≠ None → 用 override; 否则 → fallback 到 shared。

        适用于: goal_type, ee_link, gripper, pose_constraint, pose_constraint_frame。

        示例:
            _resolve_dual("pose", None, None)           → ("pose", "pose")       # 两臂相同
            _resolve_dual("pose", "joint", "pose")      → ("joint", "pose")      # 两臂不同
            _resolve_dual("tcp", "tcp_bottom", None)     → ("tcp_bottom", "tcp")  # 单侧覆盖
            _resolve_dual(None, None, 0.5)               → (None, 0.5)           # gripper 单侧指定
        """
        return (
            left_override if left_override is not None else shared,
            right_override if right_override is not None else shared,
        )

    # =========================================================
    # 1. 原子元语 (Atomic Primitives) - 返回 DualPlanResult
    # =========================================================
    def move_to(self,
                target_pose: Union[np.ndarray, List[float]],
                arm: Union[str, ArmSide] = ArmSide.LEFT,
                goal_type: Literal["joint", "pose", "waypoints"] = "pose",
                pose_constraint=None,
                pose_constraint_frame: str = "world",
                ee_link: Optional[str] = None,
                gripper: Optional[float] = None,
                other_arm: str = "hold",
                sync_obs: Optional[ObservationMessage] = None,
                **planner_kwargs) -> DualPlanResult:
        """单臂移动。另一只臂行为由 other_arm 控制。

        Args:
            target_pose: 目标 pose[7]/pos[3]/joints[N]/waypoints
            arm: 要移动的手臂
            goal_type: "joint" | "pose" | "waypoints"
            pose_constraint: 约束 [x,y,z,rx,ry,rz]
            ee_link: TCP 链接名, None = planner 默认
            gripper: 移动臂的夹爪值, None = 保持当前状态
            other_arm: 另一只臂的行为 — "hold"(保持不动) | "home"(回零位+夹爪张开)
            sync_obs: 可选，传入 ObservationMessage 自动调 sync_state() 同步规划状态
            **planner_kwargs: velocity_scale, interpolation_mode 等 backend 参数
        """
        if sync_obs is not None: self.sync_state(sync_obs)
        self._check_state_ready()
        s = self.planning_state
        arm_str = arm.value if isinstance(arm, ArmSide) else arm

        target = self._process_target(target_pose, goal_type)
        if target is None:
            return DualPlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Invalid target_pose")

        # 构建移动臂和保持臂的 ArmMotionGoal
        if arm_str == "left":
            moving = ArmMotionGoal(
                start=s.left_joints, goal=target, goal_type=goal_type,
                gripper=gripper if gripper is not None else s.left_gripper,
                pose_constraint=pose_constraint,
                constraint_frame=pose_constraint_frame, ee_link=ee_link)
            holding = self._make_other_arm_goal("left", other_arm)
            plan = self.planner_manager.plan_dual_arm(left=moving, right=holding, **planner_kwargs)
        elif arm_str == "right":
            holding = self._make_other_arm_goal("right", other_arm)
            moving = ArmMotionGoal(
                start=s.right_joints, goal=target, goal_type=goal_type,
                gripper=gripper if gripper is not None else s.right_gripper,
                pose_constraint=pose_constraint,
                constraint_frame=pose_constraint_frame, ee_link=ee_link)
            plan = self.planner_manager.plan_dual_arm(left=holding, right=moving, **planner_kwargs)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        if plan.success:
            self._advance_state(plan)
            if other_arm != "hold":
                logger.debug(f"[move_to] OK, other_arm={other_arm}, L.dur={plan.left.duration:.2f}s R.dur={plan.right.duration:.2f}s")
        else:
            logger.error(f"[move_to] FAILED, other_arm={other_arm}, L={plan.left.success if plan.left else None}, R={plan.right.success if plan.right else None}")
        return plan

    def move_dual_arm_to(self,
                         left_target_pose: Union[np.ndarray, List[float]],
                         right_target_pose: Union[np.ndarray, List[float]],
                         # Per-arm 参数 (共享 + 独立覆盖, 通过 _resolve_dual 解析)
                         goal_type: Literal["joint", "pose", "waypoints"] = "pose",
                         left_goal_type: Optional[str] = None,
                         right_goal_type: Optional[str] = None,
                         ee_link: Optional[str] = None,
                         left_ee_link: Optional[str] = None,
                         right_ee_link: Optional[str] = None,
                         gripper: Optional[float] = None,
                         left_gripper: Optional[float] = None,
                         right_gripper: Optional[float] = None,
                         pose_constraint=None,
                         left_pose_constraint=None,
                         right_pose_constraint=None,
                         pose_constraint_frame: str = "world",
                         left_pose_constraint_frame: Optional[str] = None,
                         right_pose_constraint_frame: Optional[str] = None,
                         # 其他
                         sync_obs: Optional[ObservationMessage] = None,
                         **planner_kwargs) -> DualPlanResult:
        """双臂同时移动到各自目标。

        参数分两类:
          A 类 (per-arm): 支持"共享 + 独立覆盖"，通过 _resolve_dual() 解析。
              包括: goal_type, ee_link, gripper, pose_constraint, pose_constraint_frame
          B 类 (backend): 透传给底层 planner，如 velocity_scale, interpolation_mode, steps_per_segment

        A 类参数解析规则 — 每个参数有 3 个签名位 (shared, left_, right_):
          - shared 设置两臂共享默认值
          - left_/right_ 覆盖对应臂的值 (≠ None 时生效)
          - 规则: left_val = left_override if left_override is not None else shared

        调用示例:
            # 两臂相同 (最常见)
            move_dual_arm_to(..., goal_type="pose", ee_link="tcp_link_bottom")

            # 两臂不同 goal_type (fold 任务: 一臂锁关节, 另一臂移动)
            move_dual_arm_to(..., left_goal_type="joint", right_goal_type="pose",
                             left_gripper=0.0, right_gripper=1.0)

            # 单侧覆盖 (左臂覆盖, 右臂用共享值)
            move_dual_arm_to(..., ee_link="tcp_link", left_ee_link="tcp_link_bottom")

        gripper 参数: None 表示保持当前状态 (从 planning_state 读取)。
        sync_obs: 可选，传入 ObservationMessage 自动调 sync_state() 同步规划状态。
        """
        if sync_obs is not None: self.sync_state(sync_obs)
        self._check_state_ready()
        s = self.planning_state

        # 1. 统一解析所有 per-arm 参数
        l_gt, r_gt = self._resolve_dual(goal_type, left_goal_type, right_goal_type)
        l_ee, r_ee = self._resolve_dual(ee_link, left_ee_link, right_ee_link)
        l_gp, r_gp = self._resolve_dual(gripper, left_gripper, right_gripper)
        l_pc, r_pc = self._resolve_dual(pose_constraint, left_pose_constraint, right_pose_constraint)
        l_pf, r_pf = self._resolve_dual(pose_constraint_frame, left_pose_constraint_frame, right_pose_constraint_frame)

        # gripper fallback: None → 保持当前状态
        l_gp = l_gp if l_gp is not None else s.left_gripper
        r_gp = r_gp if r_gp is not None else s.right_gripper

        # 2. 处理目标
        left_target = self._process_target(left_target_pose, l_gt)
        if left_target is None:
            return DualPlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Invalid left_target_pose")
        right_target = self._process_target(right_target_pose, r_gt)
        if right_target is None:
            return DualPlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Invalid right_target_pose")

        # 3. 打包 ArmMotionGoal（结构化接口，per-arm 参数全部封装，无 kwargs 泄漏）
        left_goal = ArmMotionGoal(
            start=s.left_joints, goal=left_target, goal_type=l_gt,
            gripper=l_gp, pose_constraint=l_pc,
            constraint_frame=l_pf, ee_link=l_ee)
        right_goal = ArmMotionGoal(
            start=s.right_joints, goal=right_target, goal_type=r_gt,
            gripper=r_gp, pose_constraint=r_pc,
            constraint_frame=r_pf, ee_link=r_ee)

        # 4. 调用 PlannerManager (只有 backend 参数透传)
        plan = self.planner_manager.plan_dual_arm(left=left_goal, right=right_goal, **planner_kwargs)
        if plan.success: self._advance_state(plan)
        return plan

    def move_to_home(self, sync_obs: Optional[ObservationMessage] = None):
        if sync_obs is not None: self.sync_state(sync_obs)
        left_home = np.zeros(len(self.planner_manager.left_joint_names))
        right_home = np.zeros(len(self.planner_manager.right_joint_names))
        return self.move_dual_arm_to(
            left_target_pose=left_home, right_target_pose=right_home, goal_type="joint",
            left_gripper=1.0, right_gripper=1.0
        )



    def gripper_action(self, target_val, arm: Union[str, ArmSide] = ArmSide.LEFT, sync_obs=None, duration=1.0):
        if sync_obs is not None: self.sync_state(sync_obs)
        self._check_state_ready()
        s = self.planning_state
        arm = arm.value if isinstance(arm, ArmSide) else arm

        curr_l, curr_r = s.left_joints, s.right_joints
        curr_g_l, curr_g_r = s.left_gripper, s.right_gripper

        if arm == "left":
            res_l = self.planner_manager.plan_gripper(target_val, current_arm_joints=curr_l, current_gripper_value=curr_g_l, duration=duration, arm="left")
            res_r = self.planner_manager._make_hold_plan(curr_r, res_l, self.planner_manager.right_joint_names, curr_g_r)
        else:
            res_r = self.planner_manager.plan_gripper(target_val, current_arm_joints=curr_r, current_gripper_value=curr_g_r, duration=duration, arm="right")
            res_l = self.planner_manager._make_hold_plan(curr_l, res_r, self.planner_manager.left_joint_names, curr_g_l)

        success = res_l.success and res_r.success

        if success:
            plan = DualPlanResult.make_success(res_l, res_r)
            self._advance_state(plan)
        else:
            plan = DualPlanResult.make_error(PlannerErrorCode.UNKNOWN_ERROR, "Gripper action failed")

        return plan

    def gripper_bimanual_action(self, left_target_val, right_target_val, sync_obs=None,
                                 duration=1.0, stagger_leader: Optional[str] = None,
                                 stagger_delay: float = 0.0):
        """双臂夹爪同时动作, 可选地让 follower 延迟 stagger_delay 秒再启动.

        stagger_leader=None (默认): 两臂完全同步, 同时开始同时结束 (duration 秒).
        stagger_leader='left' / 'right': 指定 leader, follower 前缀 hold stagger_delay 秒,
            leader 末尾补 hold stagger_delay 秒, 两个 trajectory 时长对齐 =
            duration + stagger_delay. 两臂 **并发** 执行, 总时长不翻倍.

        用途: 破 bimanual state shortcut (不让 policy 学到"对侧关 → 自己关").
        """
        if sync_obs is not None: self.sync_state(sync_obs)
        self._check_state_ready()
        s = self.planning_state

        res_l = self.planner_manager.plan_gripper(left_target_val, current_arm_joints=s.left_joints, current_gripper_value=s.left_gripper, duration=duration, arm="left")
        res_r = self.planner_manager.plan_gripper(right_target_val, current_arm_joints=s.right_joints, current_gripper_value=s.right_gripper, duration=duration, arm="right")

        success = res_l.success and res_r.success
        if not success:
            return DualPlanResult.make_error(PlannerErrorCode.UNKNOWN_ERROR, "Bimanual gripper action failed")

        if stagger_leader in ("left", "right") and stagger_delay > 1e-6:
            if stagger_leader == "left":
                res_l = self._append_hold_frames(res_l, stagger_delay)
                res_r = self._prepend_hold_frames(res_r, stagger_delay)
            else:
                res_r = self._append_hold_frames(res_r, stagger_delay)
                res_l = self._prepend_hold_frames(res_l, stagger_delay)

        plan = DualPlanResult.make_success(res_l, res_r)
        self._advance_state(plan)
        return plan

    @staticmethod
    def _prepend_hold_frames(res: PlanResult, hold_dur: float) -> PlanResult:
        """在 res 开头前置 hold_dur 秒的静止段 (保持 res[0] 的 pos/gripper)."""
        if hold_dur <= 1e-6 or len(res.time) == 0:
            return res
        dt = float(res.time[1] - res.time[0]) if len(res.time) > 1 else 0.02
        n_hold = max(1, int(round(hold_dur / dt)))
        actual_hold = n_hold * dt

        hold_pos = np.tile(res.position[0:1], (n_hold, 1))
        hold_vel = np.zeros_like(hold_pos)
        hold_acc = np.zeros_like(hold_pos)
        hold_time = np.arange(n_hold) * dt
        new_time = np.concatenate([hold_time, res.time + actual_hold])
        new_pos = np.concatenate([hold_pos, res.position], axis=0)
        new_vel = np.concatenate([hold_vel, res.velocity], axis=0)
        new_acc = np.concatenate([hold_acc, res.acceleration], axis=0)

        new_res = PlanResult.make_success(
            time=new_time, pos=new_pos, vel=new_vel, acc=new_acc,
            joint_names=res.joint_names, arm_dof=res.arm_dof,
        )
        if len(res.ee_position) > 0:
            new_res.ee_position = np.concatenate(
                [np.tile(res.ee_position[0:1], (n_hold, 1)), res.ee_position], axis=0,
            )
        if len(res.ee_quaternion) > 0:
            new_res.ee_quaternion = np.concatenate(
                [np.tile(res.ee_quaternion[0:1], (n_hold, 1)), res.ee_quaternion], axis=0,
            )
        new_res.info = dict(res.info)
        return new_res

    @staticmethod
    def _append_hold_frames(res: PlanResult, hold_dur: float) -> PlanResult:
        """在 res 末尾追加 hold_dur 秒的静止段 (保持 res[-1] 的 pos/gripper)."""
        if hold_dur <= 1e-6 or len(res.time) == 0:
            return res
        dt = float(res.time[1] - res.time[0]) if len(res.time) > 1 else 0.02
        n_hold = max(1, int(round(hold_dur / dt)))

        hold_pos = np.tile(res.position[-1:], (n_hold, 1))
        hold_vel = np.zeros_like(hold_pos)
        hold_acc = np.zeros_like(hold_pos)
        t_end = float(res.time[-1])
        hold_time = t_end + (np.arange(1, n_hold + 1) * dt)
        new_time = np.concatenate([res.time, hold_time])
        new_pos = np.concatenate([res.position, hold_pos], axis=0)
        new_vel = np.concatenate([res.velocity, hold_vel], axis=0)
        new_acc = np.concatenate([res.acceleration, hold_acc], axis=0)

        new_res = PlanResult.make_success(
            time=new_time, pos=new_pos, vel=new_vel, acc=new_acc,
            joint_names=res.joint_names, arm_dof=res.arm_dof,
        )
        if len(res.ee_position) > 0:
            new_res.ee_position = np.concatenate(
                [res.ee_position, np.tile(res.ee_position[-1:], (n_hold, 1))], axis=0,
            )
        if len(res.ee_quaternion) > 0:
            new_res.ee_quaternion = np.concatenate(
                [res.ee_quaternion, np.tile(res.ee_quaternion[-1:], (n_hold, 1))], axis=0,
            )
        new_res.info = dict(res.info)
        return new_res

    # =========================================================
    # 2. 组合技能 (Compound Skills) - 返回 List[PlanResult]
    # =========================================================

    def pick(self, target_pos_7d: np.ndarray, arm: Union[str, ArmSide] = ArmSide.LEFT, 
             approach_height:float = 0.1, skip_approach: bool = False, close_gripper_value: float = 0.03,
             lift_pose_param:Union[np.ndarray, float] = None, ee_link:str = None, pick_end_settle_time: float = 1.0) -> List[ExecutionStep]:
        """
        组合技：抓取 = 靠近 -> 下探 -> 闭合 -> 抬起
        """
        sequence = []
        target_pos = target_pos_7d.copy()

        arm = arm.value if isinstance(arm, ArmSide) else arm

        # 1. Approach (上方)
        if not skip_approach:
            approach_pose = target_pos.copy(); approach_pose[2] += approach_height
            p1 = self.move_to(approach_pose, arm=arm, pose_constraint=[1, 1, 1, 0, 0, 0], ee_link=ee_link)
            if not p1.success:
                logger.error("[pick] Approach phase failed")
                return sequence
            sequence.append(ExecutionStep(plan=p1, settle_time=0.0, description="approach"))

        # 2. Down (下探)
        grasp_pose = target_pos.copy()
        p2 = self.move_to(grasp_pose, arm=arm, pose_constraint=[1, 1, 1, 0, 0, 0], ee_link=ee_link)
        if not p2.success: 
            logger.error("[pick] Down phase failed")
            return sequence
        sequence.append(ExecutionStep(plan=p2, settle_time=0.1, description="down"))
        
        # 3. Close (闭合)
        p3 = self.gripper_action(target_val=close_gripper_value, arm=arm,duration=1.0)
        if not p3.success: 
            logger.error("[pick] Close phase failed")
            return sequence
        sequence.append(ExecutionStep(plan=p3, settle_time=0.5, description="close"))
        
        # 4. Lift (抬起)
        if lift_pose_param is None:
            return sequence
        
        if isinstance(lift_pose_param, float):
            lift_height = lift_pose_param
            lift_pose = target_pos.copy()
            lift_pose[2] = lift_pose[2] + lift_height
        
        elif isinstance(lift_pose_param, np.ndarray):
            lift_pose = lift_pose_param
        else:
            logger.error(f"[pick] Invalid lift_pose_param type: {type(lift_pose_param)}")
            return sequence
        p4 = self.move_to(lift_pose, arm=arm, pose_constraint=[1, 1, 1, 0, 0, 0], ee_link=ee_link)
        if not p4.success:
            logger.error("[pick] Lift phase failed")
            return sequence
        sequence.append(ExecutionStep(plan=p4, settle_time=pick_end_settle_time, description="lift"))
        
        return sequence
    

    def fling(self, 
            current_pose_left: np.ndarray,
            current_pose_right: np.ndarray,
            # 动作参数
            fling_forward_amplitude: float = 0.33, # 向前摆动的幅度 
            fling_back_amplitude: float = 0.04,    # 向后摆动的幅度
            fling_height: float = 0.10,    # 额外抬起的高度
            drag_dist: float = 0.15,       # 最后拖拽距离
            mid_low_height: float = 0.2,   # 
            low_height: float = 0.03,      # 落地高度
            # 速度参数
            fling_vel_scale: float = 1.0,  # 猛甩速度
            fling_accel_scale: float = None,  # 猛甩加速度 (None=跟随vel_scale)
            ee_link: str = None
            ) -> List[ExecutionStep]:
        """
        类似钟摆，先向后蓄力，再向前猛甩，利用动量展开。
        轨迹: Back -> Forward (Fast) -> Down & Back (Drag)
        """
        sequence = []
        target_l = current_pose_left.copy()
        target_r = current_pose_right.copy()

        # 1. Pre-Swing / Windup (向后蓄力)
        # 对应 simEnv 中 movep(..., -x)
        windup_l = target_l.copy(); windup_r = target_r.copy()
        windup_l[0] -= fling_back_amplitude  # 向后
        windup_r[0] -= fling_back_amplitude

        p1 = self.move_dual_arm_to(
            left_target_pose=windup_l, right_target_pose=windup_r,
            pose_constraint=[1,1,1,0,0,0],
            ee_link=ee_link,
            velocity_scale=0.5 # 蓄力慢
        )
        if p1.success: sequence.append(ExecutionStep(plan=p1, settle_time=0.0, description="swing_windup"))

        # 2. Fling Forward (向前猛甩)
        throw_l = target_l.copy(); throw_r = target_r.copy()
        throw_l[0] += fling_forward_amplitude         # 向前
        throw_r[0] += fling_forward_amplitude
        throw_l[2] += fling_height            # 保持高度
        throw_r[2] += fling_height
        
        
        # 3. Lower & Drag (下放并回拉)
        # mid drag 
        drag_l_1=target_l.copy(); drag_r_1 = target_r.copy()
        drag_l_1[0]=target_l[0] + 1/2* fling_forward_amplitude
        drag_r_1[0]=target_r[0] + 1/2* fling_forward_amplitude
        
        drag_l_1[2]=mid_low_height
        drag_r_1[2]=mid_low_height
        
        # 对应 simEnv 中 movep(..., -x_drag)
        drag_l = target_l.copy(); drag_r = target_r.copy()
        drag_l[0] -= drag_dist  # 回拉
        drag_r[0] -= drag_dist
        drag_l[2] = low_height  # 降低
        drag_r[2] = low_height

        p2 = self.move_dual_arm_to(
            left_target_pose=[throw_l,drag_l_1,drag_l], right_target_pose=[throw_r,drag_r_1,drag_r],
            goal_type="waypoints",
            pose_constraint=[[1,1,1,0,0,0],[1,1,0.5,0,0,0],[1,1,1,0,0,0]],
            interpolation_mode = ["joint","joint","cartesian"],
            steps_per_segment = 3,
            ee_link="tcp_link_bottom",
            velocity_scale=fling_vel_scale, # 极速
            acceleration_scale=fling_accel_scale,
        )
        if p2.success: sequence.append(ExecutionStep(plan=p2, settle_time=0.0, description="swing_throw"))


        # p3 = self.move_dual_arm_to(
        #     left_target_pose=drag_l, right_target_pose=drag_r,
        #     pose_constraint=[1,1,1,0,0,0],
        #     ee_link=ee_link,
        #     velocity_scale=1.0 # 正常速度下放
        # )
        # if p3.success: sequence.append(ExecutionStep(plan=p3, settle_time=0.5, description="swing_drag"))

        return sequence
    

  

    def stroke_down(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        arm: Union[str, ArmSide] = ArmSide.LEFT,
        num_waypoints: int = 6,
        stroke_speed_factor: float = 1.0,
        gripper_open: float = 1.0,
        apply_contact_force: bool = True,
        contact_offset: float = 0.01
    ) -> List[ExecutionStep]:
        """
        组合技：缕边动作 - 从起点沿布料表面向下滑动到终点
        
        这个动作模拟手指沿布料边缘向下"捋"的动作，用于整理布料或找到对角点。
        
        Args:
            start_pos: 起始位置 [x, y, z] 或 [x, y, z, qw, qx, qy, qz]
            end_pos: 终点位置 [x, y, z] 或 [x, y, z, qw, qx, qy, qz]
            arm: 执行动作的手臂
            num_waypoints: 路径点数量（越多越平滑）
            stroke_speed_factor: 速度因子（越小越慢，越精细）
            gripper_open: 夹爪开度（保持打开状态）
            apply_contact_force: 是否施加接触力（轻微下压）
            contact_offset: 接触偏移量（向下的额外压力）
        
        Returns:
            List[ExecutionStep]: 执行步骤序列
        """
        sequence = []
        arm = arm.value if isinstance(arm, ArmSide) else arm
        
        # 1. 确保位置格式正确
        if len(start_pos) == 3:
            start_pose = np.concatenate([start_pos, [1.0, 0.0, 0.0, 0.0]])
        elif len(start_pos) == 7:
            start_pose = start_pos.copy()
        else:
            logger.error(f"[stroke_down] Invalid start_pos length: {len(start_pos)}")
            return sequence
        
        if len(end_pos) == 3:
            end_pose = np.concatenate([end_pos, [1.0, 0.0, 0.0, 0.0]])
        elif len(end_pos) == 7:
            end_pose = end_pos.copy()
        else:
            logger.error(f"[stroke_down] Invalid end_pos length: {len(end_pos)}")
            return sequence
        
        # 2. 确保夹爪打开
        logger.info(f"[stroke_down] Opening gripper on {arm} arm")
        p_open = self.gripper_action(target_val=gripper_open, arm=arm, duration=0.5)
        if not p_open.success:
            logger.warning("[stroke_down] Failed to open gripper, continuing anyway...")
        else:
            sequence.append(ExecutionStep(plan=p_open, settle_time=0.1, description="open_gripper_for_stroke"))
    
        # 3. 移动到起始位置
        logger.info(f"[stroke_down] Moving to stroke start position")
        p_start = self.move_to(start_pose, arm=arm, pose_constraint=[1, 1, 1, 0, 0, 0])
        if not p_start.success:
            logger.error("[stroke_down] Failed to reach start position")
            return sequence
        sequence.append(ExecutionStep(plan=p_start, settle_time=0.15, description="move_to_stroke_start"))
        
        # 4. 生成中间路径点（线性插值）
        logger.info(f"[stroke_down] Generating {num_waypoints} waypoints from start to end")
        
        # 位置插值
        start_xyz = start_pose[:3]
        end_xyz = end_pose[:3]
        
        # 可选：如果需要施加接触力，略微降低Z坐标
        if apply_contact_force:
            logger.debug(f"[stroke_down] Applying contact offset: {contact_offset}m")
        
        waypoints = []
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            
            # 线性插值位置
            interp_xyz = start_xyz * (1 - alpha) + end_xyz * alpha
            
            # 可选：添加接触压力（向下偏移）
            if apply_contact_force:
                # 在中间部分施加更大的压力（抛物线分布）
                pressure_factor = 4 * alpha * (1 - alpha)  # 在alpha=0.5时最大
                interp_xyz[2] -= contact_offset * pressure_factor
            
            # 保持姿态不变（或者可以插值姿态）
            waypoint = np.concatenate([interp_xyz, start_pose[3:]])
            waypoints.append(waypoint)
        
        # 5. 逐个执行路径点
        settle_time = 0.05 / stroke_speed_factor  # 速度因子影响停留时间
        
        for idx, waypoint in enumerate(waypoints):
            logger.debug(f"[stroke_down] Waypoint {idx+1}/{num_waypoints}: {waypoint[:3]}")
            
            plan = self.move_to(
                waypoint,
                arm=arm,
                pose_constraint=[1, 1, 1, 0, 0, 0]  # 只约束位置，姿态自由
            )
            
            if not plan.success:
                logger.warning(f"[stroke_down] Waypoint {idx+1} failed, stopping stroke")
                break
            
            # 最后一个点停留时间稍长
            current_settle = settle_time if idx < len(waypoints) - 1 else 0.2
            
            sequence.append(ExecutionStep(
                plan=plan,
                settle_time=current_settle,
                description=f"stroke_waypoint_{idx+1}"
            ))
        
        logger.success(f"[stroke_down] Stroke completed with {len(sequence)} steps")
        return sequence

    # =========================================================
    # Helper Function
    # =========================================================
    def calculate_base_to_target_horizontal_quat(self, base_pos: np.ndarray, target_pos:np.ndarray,
                                                 yaw_jitter: float = 0.0) -> np.ndarray:
        """Target 抓取 quat: yaw 对准 base→target + TCP-z 直接朝下 (180° pitch flip).

        旧版只返回 yaw-only quat (TCP-z 朝上), IK 得自己找 180° pitch flip 去翻
        夹爪竖直; 远端 reach 不够就塌成"手腕后仰"退化解 (tilt 60-90°).
        新版把 pitch flip 预先写进 target → IK 只需 ±几度微调, 远端 tilt 从
        60-90° 降到 20-40°, 且噪声下波动收窄到 ±5°.

        Quat 构造: q = R_z(yaw) ∘ R_y(180°) (先 pitch-flip, 再 yaw).
        这样 TCP-y (piper finger 开合轴) 会垂直于 base→target 方向 → 夹爪
        "跨过" 抓取路径闭合, 符合直觉. 手算: q_wxyz = (0, -sin(yaw/2), cos(yaw/2), 0).

        yaw_jitter: 额外 yaw offset (rad). task 层每 episode 采样一个小量 (±几度)
            → 打破 L_j6 被 j1 吃死的 IK 零空间, 拓宽手腕 roll 的训练分布.
        """
        dx = target_pos[0] - base_pos[0]
        dy = target_pos[1] - base_pos[1]
        yaw = np.arctan2(dy, dx) + yaw_jitter

        half_yaw = yaw / 2.0
        qw = 0.0
        qx = -np.sin(half_yaw)
        qy = np.cos(half_yaw)
        qz = 0.0
        return np.array([qw, qx, qy, qz])
    
    def _process_target(self, target, goal_type: Literal["joint", "pose", "waypoints"] = "pose"):
        """统一处理目标格式"""
        _VALID_GOAL_TYPES = ("joint", "pose", "waypoints")
        if goal_type not in _VALID_GOAL_TYPES:
            logger.error(f"Unknown goal_type: '{goal_type}', valid options: {_VALID_GOAL_TYPES}")
            return None
        # Case 1: Waypoints - 列表或二维数组
        if goal_type == "waypoints":
            if isinstance(target, list):
                processed = []
                for wp in target:
                    if len(wp) == 3:
                        processed.append(np.concatenate([wp, [1.0, 0.0, 0.0, 0.0]]))
                    elif len(wp) == 7:
                        processed.append(np.array(wp))
                    else:
                        logger.error(f"Invalid waypoint length: {len(wp)}")
                        return None
                return processed
            else:
                # 假设是 numpy array，检查维度
                target = np.squeeze(np.array(target))
                if target.ndim == 2:  # shape (N, 3) or (N, 7)
                    if target.shape[1] == 3:
                        quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(target), 1))
                        return np.hstack([target, quats])
                    elif target.shape[1] == 7:
                        return target
                logger.error(f"Invalid target {target} waypoints shape: {target.shape}")
                return None
        
        # Case 2: Single pose or joint config
        else:
            if len(target) == 3:
                return np.concatenate([target, [1.0, 0.0, 0.0, 0.0]])
            elif len(target) in [6, 7]:  # 关节空间 or 完整pose
                return np.array(target)
            else:
                logger.error(f"Invalid target: {target} length: {len(target)}")
                return None
    
    def _make_other_arm_goal(self, arm_str: str, other_arm: str) -> ArmMotionGoal:
        """为非移动臂构建 ArmMotionGoal。

        Args:
            arm_str: 移动臂 ("left"/"right")，此方法为其对面臂构建目标
            other_arm: "hold" = 保持不动, "home" = 回零位 + 夹爪张开
        """
        s = self.planning_state
        if arm_str == "left":
            other_joints, other_gripper = s.right_joints, s.right_gripper
            other_names = self.planner_manager.right_joint_names
        else:
            other_joints, other_gripper = s.left_joints, s.left_gripper
            other_names = self.planner_manager.left_joint_names

        if other_arm == "hold":
            return ArmMotionGoal(start=other_joints, gripper=other_gripper)
        elif other_arm == "home":
            home_goal = np.zeros(len(other_names))
            return ArmMotionGoal(start=other_joints, goal=home_goal,
                                 goal_type="joint", gripper=1.0)
        else:
            raise ValueError(f"Unknown other_arm mode: {other_arm!r}, expected 'hold' or 'home'")



