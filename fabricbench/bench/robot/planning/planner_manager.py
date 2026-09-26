import numpy as np
from scipy.interpolate import interp1d
import os
import sys
from typing import Literal, Dict, Optional, List, Tuple, Union
from loguru import logger
from dataclasses import asdict, dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import common types (PlanResult, PlannerErrorCode)
# Ensure robot_common is in your python path or adjust import
try:
    from fabricbench.bench.common.types import PlanResult, DualPlanResult, PlanningState, RobotState, ArmMotionGoal, ObservationMessage
    from fabricbench.bench.common.errors import PlannerErrorCode
    from fabricbench.bench.robot.utils.planner_ulils import validate_goal_format
except ImportError:
    # Fallback definition if common module is not set up yet (for standalone testing)
    logger.warning("Could not import fabricbench.bench.common.types, using local definition fallback.")
    from dataclasses import dataclass, field
    from enum import Enum, auto

class PlannerManager:
    """
    Unified Planner Manager for Single and Dual Arm Robots.
    Handles:
    - Lazy loading of planner backends (Curobo, Toppra)
    - Coordinate transformation (World <-> Base)
    - Dual arm time synchronization
    - Gripper state appending & State Management
    """
    
    def __init__(
        self,
        robot_type: Literal["single_arm", "dual_arm"] = "single_arm",
        planner_type: Literal["curobo", "toppra", "min_jerk", "adaptive"] = "curobo",
        urdf_path: Optional[str] = None,
        single_arm_config: Optional[Dict] = None,
        dual_arm_config: Optional[Dict] = None,
    ):
        self.robot_type = robot_type
        self.planner_type = planner_type

        # 规划器工作状态（非真实状态，是用于规划计算的工作副本）
        # 每次规划前需通过 sync_from_obs() 从观测同步
        self.planning_state = PlanningState()
        
        # 2. Lazy load planner class
        self._planner_cls = self._load_planner_backend(planner_type)
        
        # 3. Initialize based on robot type
        if robot_type == "single_arm":
            self._init_single_arm(urdf_path, single_arm_config or {})
        elif robot_type == "dual_arm":
            if dual_arm_config is None:
                raise ValueError("dual_arm_config is required for dual_arm robot")
            self._init_dual_arm(urdf_path, dual_arm_config)
        else:
            raise ValueError(f"Unknown robot_type: {robot_type}")

        logger.success(f"[PlannerManager] Initialized {robot_type} with {planner_type}")

    def _load_planner_backend(self, planner_type: str):
        """Lazy load planner backend class"""
        try:
            if planner_type == "curobo":
                from fabricbench.bench.robot.planning.curobo_planner import CuroboPlanner
                return CuroboPlanner
            elif planner_type == "toppra":
                from fabricbench.bench.robot.planning.toppra_planner import ToppraPlanner
                return ToppraPlanner
            elif planner_type == "min_jerk":
                from fabricbench.bench.robot.planning.min_jerk_planner import MinJerkPlanner
                return MinJerkPlanner
            elif planner_type == "adaptive":
                from fabricbench.bench.robot.planning.adaptive_planner import AdaptivePlanner
                return AdaptivePlanner
            else:
                raise ValueError(f"Unknown planner_type: {planner_type}")
        except ImportError as e:
            logger.critical(f"Failed to import planner backend '{planner_type}': {e}")
            raise

    def _init_single_arm(self, urdf_path: Optional[str], config: Dict):
        """Initialize single arm planner"""
        final_urdf_path = urdf_path or config.get("urdf_path")
        if not final_urdf_path:
            raise ValueError("urdf_path must be provided for single_arm")
        
        self.base_pose = np.array(config.get("base_pose", [0, 0, 0, 1, 0, 0, 0]))
        self.gripper_joints = config.get("gripper_joints", [])  # List of gripper joint names
        j_names = config.get("joint_names") or config.get("active_joint_names")

        # Instantiate planner
        planner_kwargs = {
                "urdf_path": final_urdf_path,
                "base_link": config.get("base_link"),
                "ee_link": config.get("ee_link"),
                "active_joint_names": j_names,
            }
        if "world_config" in config:
            planner_kwargs["world_config"] = config["world_config"]
        if "use_cuda_graph" in config:
            planner_kwargs["use_cuda_graph"] = config["use_cuda_graph"]
        if "velocity_limits" in config:
            planner_kwargs["velocity_limits"] = config["velocity_limits"]
        if "acceleration_limits" in config:
            planner_kwargs["acceleration_limits"] = config["acceleration_limits"]
        if "avg_speed" in config:
            planner_kwargs["avg_speed"] = config["avg_speed"]
        if "default_planner_mode" in config:
            planner_kwargs["default_planner_mode"] = config["default_planner_mode"]

        self.planner = self._planner_cls(**planner_kwargs)

        self.joint_names = self.planner.active_joint_names
        if self.joint_names is None:
            logger.warning("Active joint names are None. Please ensure 'joint_names' is in config.")
       
    def _init_dual_arm(self, urdf_path: Optional[str], config: Dict):
        """Initialize dual arm planners"""
        mode = config.get("mode", "single_urdf")
        
        # Helper to instantiate planner
        def create_planner(cfg, default_urdf):
            u_path = cfg.get("urdf_path") or default_urdf
            if not u_path: raise ValueError(f"URDF path missing in config: {cfg}")

            j_names = cfg.get("joint_names") or cfg.get("active_joint_names")
            planner_kwargs = {
                "urdf_path": u_path,
                "base_link": cfg.get("base_link"),
                "ee_link": cfg.get("ee_link"),
                "active_joint_names": j_names,
            }
            if "world_config" in cfg:
                planner_kwargs["world_config"] = cfg["world_config"]
            if "use_cuda_graph" in cfg:
                planner_kwargs["use_cuda_graph"] = cfg["use_cuda_graph"]
            if "velocity_limits" in cfg:
                planner_kwargs["velocity_limits"] = cfg["velocity_limits"]
            if "acceleration_limits" in cfg:
                planner_kwargs["acceleration_limits"] = cfg["acceleration_limits"]
            if "avg_speed" in cfg:
                planner_kwargs["avg_speed"] = cfg["avg_speed"]
            if "default_planner_mode" in cfg:
                planner_kwargs["default_planner_mode"] = cfg["default_planner_mode"]
            if "base_pose" in cfg:
                planner_kwargs["base_height"] = cfg["base_pose"][2]

            return  self._planner_cls(**planner_kwargs)
        

        left_cfg = config.get("left", {})
        right_cfg = config.get("right", {})
        
        # Configure base poses
        if mode == "single_urdf":
            # Both arms share the same URDF root, usually at [0,0,0]
            self.left_base_pose = np.array([0, 0, 0, 1, 0, 0, 0])
            self.right_base_pose = np.array([0, 0, 0, 1, 0, 0, 0])
            final_urdf = urdf_path or config.get("urdf_path")
        else: # dual_urdf
            # Arms have separate URDFs and separate base poses in world
            self.left_base_pose = np.array(left_cfg.get("base_pose", [0, 0.25, 0, 1, 0, 0, 0]))
            self.right_base_pose = np.array(right_cfg.get("base_pose", [0, -0.25, 0, 1, 0, 0, 0]))
            final_urdf = None # Not used globally in dual_urdf mode

        # Gripper configuration
        self.left_gripper_joints = left_cfg.get("gripper_joints", [])
        self.right_gripper_joints = right_cfg.get("gripper_joints", [])

        # Create planners
        self.left_planner = create_planner(left_cfg, final_urdf)
        self.right_planner = create_planner(right_cfg, final_urdf)
        
        self.left_joint_names = self.left_planner.active_joint_names
        self.right_joint_names = self.right_planner.active_joint_names
    
    
    # =========================================================================
    # State Management (规划工作状态，非真实状态)
    # =========================================================================
    def sync_from_obs(self, obs: ObservationMessage):
        """从观测同步规划状态（锚定到真实值）。

        每个规划阶段开头由 task 调用（通过 Skills.sync_state）。
        调用后 source="observed"，可信赖用于规划。

        仿真和真机使用相同机制，区别在于调用频率：
        - 仿真：每个 stage 开头同步一次，中间可连续 advance
        - 真机：每步前同步纠偏（保守），或每 N 步（激进）
        """
        self.planning_state.left_joints = obs.joint_left
        self.planning_state.right_joints = obs.joint_right
        # The gripper is deliberately NOT re-anchored to the observation.
        #
        # Re-anchoring is for tracking error: an arm is commanded to a pose and
        # should be corrected toward where it actually is. A gripper command is
        # not a pose to converge to, it is an intent -- "squeeze this hard". A
        # gripper holding something sits stalled open by the object's thickness,
        # so ``observed`` is the equilibrium of (command, contact force), not the
        # command. Overwriting the command with it hands the squeeze back, and
        # since every plan boundary syncs, the grip bleeds away over a sequence
        # (measured: 1.07 N -> 0.047 N across a grasp+lift, and the cloth drops).
        #
        # ``advance()`` already carries the commanded value forward from each
        # plan's ``final_gripper_position``, so the command survives across
        # stages on its own. Only seed it here if nothing has commanded yet.
        # The true opening is still available to callers via ``obs.gripper_*``.
        if self.planning_state.left_gripper is None:
            self.planning_state.left_gripper = obs.gripper_left
        if self.planning_state.right_gripper is None:
            self.planning_state.right_gripper = obs.gripper_right
        self.planning_state.source = "observed"
        self.planning_state.advances_since_sync = 0

    def advance(self, plan: DualPlanResult):
        """从规划结果前推状态（预测值），用于连续规划。

        plan 成功后，状态推进到轨迹终点，下一次 plan 可以从这里开始，
        无需等待执行和获取新观测。这使得多步连续规划成为可能：
            plan1 → advance → plan2 → advance → 一起 yield 执行

        调用后 source="predicted"。
        仿真中预测精确（确定性物理），真机中会累积误差，
        需定期调用 sync_from_obs() 纠偏。
        """
        l, r = plan.left, plan.right
        if l and l.success and len(l.position) > 0:
            self.planning_state.left_joints = l.final_arm_position
            self.planning_state.left_gripper = l.final_gripper_position
        if r and r.success and len(r.position) > 0:
            self.planning_state.right_joints = r.final_arm_position
            self.planning_state.right_gripper = r.final_gripper_position
        self.planning_state.source = "predicted"
        self.planning_state.advances_since_sync += 1

    # =========================================================================
    # Planning Interface
    # =========================================================================
    def plan(
        self,
        goal: Union[np.ndarray, List[float], List[np.ndarray]],
        start_joints: Optional[np.ndarray] = None,
        goal_type: Literal["joint", "pose", "waypoints"] = "pose",
        arm: Optional[Literal["left", "right"]] = None,
        gripper_value: Optional[float] = None,
        pose_constraint: Optional[Union[
            np.ndarray,           # 单个约束 [x,y,z,rx,ry,rz]
            List[float],          # 单个约束列表
            List[List[float]],    # 多个约束（for waypoints）
            Dict                  # 混合约束 {"local": [...], "world": [...]}
        ]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        **kwargs
    ) -> PlanResult:
        """
        Unified planning interface.
        
        Args:
            start_joints: Current joint configuration, default is state in planner 
            goal: Target (Joints, Pose, or List of Poses)
            goal_type: "joint", "pose", or "waypoints"
            arm: "left" or "right" (required for dual_arm)
            gripper_value: If provided, appends this value to gripper joints in the result.
            pose_constraint: Pose constraints [x, y, z, rx, ry, rz] (range 0-1)
            **kwargs: Passed to planner (e.g. velocity_scale,...)
        """
        # A. context Selection ( Single vs Dual, Left vs Right )
        target_planner = None
        target_base_pose = None
        target_gripper_joints = []
        expected_joint_dof = None

        cached_start_joints = None
        
        # A. Select Planner & Config
        if self.robot_type == "single_arm":
            target_planner = self.planner
            target_base_pose = self.base_pose
            target_gripper_joints = self.gripper_joints
            cached_start_joints = self.planning_state.left_joints  # single arm uses left_joints slot
            current_gripper_val = self.planning_state.left_gripper if self.planning_state.left_gripper is not None else 0.0
            expected_joint_dof = len(self.joint_names)

        elif self.robot_type == "dual_arm":
            if arm == "left":
                target_planner = self.left_planner
                target_base_pose = self.left_base_pose
                target_gripper_joints = self.left_gripper_joints
                cached_start_joints = self.planning_state.left_joints
                current_gripper_val = self.planning_state.left_gripper if self.planning_state.left_gripper is not None else 0.0
                expected_joint_dof = len(self.left_joint_names) if self.left_joint_names else None
            elif arm == "right":
                target_planner = self.right_planner
                target_base_pose = self.right_base_pose
                target_gripper_joints = self.right_gripper_joints
                cached_start_joints = self.planning_state.right_joints
                current_gripper_val = self.planning_state.right_gripper if self.planning_state.right_gripper is not None else 0.0
                expected_joint_dof = len(self.right_joint_names) if self.right_joint_names else None
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Dual arm robot requires arm='left' or 'right'")


        is_valid, error_msg = validate_goal_format(goal, goal_type, expected_joint_dof)
        if not is_valid:
            return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, f"Goal validation failed: {error_msg}")

        # B. Smart Start, using cached state if start_joints is None
        if start_joints is None:
            if cached_start_joints is not None:
                start_joints = cached_start_joints
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "start_joints is None and no history found.")


        # C. Transform Goal (World -> Base)
        final_goal = goal
        if goal_type in ["pose", "waypoints"]:
            # Helper to transform one pose
            def transform(p): return self._world_to_base_transform(np.array(p), target_base_pose)
            
            if goal_type == "pose":
                final_goal = transform(goal)
            elif goal_type == "waypoints":
                final_goal = [transform(p) for p in goal]

        # D. Execute Planning
        try:
            if ee_link is not None:
                kwargs["ee_link"] = ee_link
            
            if goal_type == "joint":
                res = target_planner.plan_to_joint_config(start_joints, np.array(final_goal), **kwargs)
                
            elif goal_type == "pose":
                res = target_planner.plan_to_pose(
                    start_joints, final_goal,
                    pose_constraint=pose_constraint,  # 单个约束
                    constraint_frame=constraint_frame, 
                    **kwargs
                )
                
            elif goal_type == "waypoints":
                # 🔧 智能处理约束
                if pose_constraint is None:
                    # 默认全约束
                    constraint_list = None
                elif isinstance(pose_constraint, list) and len(pose_constraint) > 0 and isinstance(pose_constraint[0], (list, np.ndarray)):
                    # 已经是列表 -> 直接传递
                    constraint_list = pose_constraint
                else:
                    # 单个约束 -> 广播到所有 waypoints
                    constraint_list = [pose_constraint] * len(final_goal)
                
                res = target_planner.plan_waypoints(
                    start_joints, final_goal,
                    pose_constraint=constraint_list,  # 列表形式
                    constraint_frame=constraint_frame, 
                    **kwargs
                )
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, f"Unknown goal_type: {goal_type}")
        except Exception as e:
            logger.exception(f"Planner backend error: {e}")
            return PlanResult.make_error(PlannerErrorCode.UNKNOWN, str(e))
        

        # E. Post-processing: Append Gripper Data, Update State
        if res.success:
            final_gripper_val = gripper_value if gripper_value is not None else current_gripper_val
            start_gripper_val = current_gripper_val
            # add gripper value to joints
            if len(target_gripper_joints) > 0:
                res = self._append_gripper_state(
                res, 
                start_val=start_gripper_val, 
                end_val=final_gripper_val, 
                g_names=target_gripper_joints
            )

        else:
            logger.error(f"planning fail at {arm}, start joints:{start_joints}, goal:{final_goal}, goal type:{goal_type}, constrain: {pose_constraint}")

        res.arm_dof = expected_joint_dof
        return res

    def plan_dual_arm(
        self,
        left: ArmMotionGoal,
        right: ArmMotionGoal,
        synchronize: bool = True,
        **planner_kwargs
    ) -> DualPlanResult:
        """双臂规划。

        Args:
            left: 左臂运动目标（goal=None 表示 hold）
            right: 右臂运动目标（goal=None 表示 hold）
            synchronize: 是否做时间同步
            **planner_kwargs: Backend 特有参数（velocity_scale, interpolation_mode 等）
        """
        if self.robot_type != "dual_arm":
            raise RuntimeError("plan_dual_arm only available for dual_arm robot")
        res_l, res_r = None, None

        # Plan Left
        if left.goal is not None:
            res_l = self.plan(
                start_joints=left.start, goal=left.goal, goal_type=left.goal_type, arm="left",
                gripper_value=left.gripper, pose_constraint=left.pose_constraint,
                constraint_frame=left.constraint_frame, ee_link=left.ee_link,
                **planner_kwargs
            )
            if not res_l.success:
                logger.error(f"left arm planning failed {res_l}, error code{res_l.error_code}, error msg{res_l.error_msg}")

        # Plan Right
        if right.goal is not None:
            res_r = self.plan(
                start_joints=right.start, goal=right.goal, goal_type=right.goal_type, arm="right",
                gripper_value=right.gripper, pose_constraint=right.pose_constraint,
                constraint_frame=right.constraint_frame, ee_link=right.ee_link,
                **planner_kwargs
            )
            if not res_r.success:
                logger.error(f"right arm planning failed {res_r}, error code{res_r.error_code}, error msg{res_r.error_msg}")

        if res_l is None and res_r is None:
            res = PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "start goal and end goal is None.")
            return DualPlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Both left and right goals are None.")

        # Fill hold arm
        if res_l is not None and res_r is None:
            curr_r_g = right.gripper if right.gripper is not None else self.planning_state.right_gripper
            res_r = self._make_hold_plan(right.start, res_l, self.right_joint_names, curr_r_g)

        elif res_r is not None and res_l is None:
            curr_l_g = left.gripper if left.gripper is not None else self.planning_state.left_gripper
            res_l = self._make_hold_plan(left.start, res_r, self.left_joint_names, curr_l_g)

        elif res_l is not None and res_r is not None and synchronize:
            # Only synchronize when BOTH arms planned successfully. A failed plan
            # (e.g. IK_FAILED_GLOBAL) carries an empty time array; feeding it to
            # the resampler crashes on res.time[-1]. Skip sync so the caller sees
            # a clean status="Fail" and can retry/degrade gracefully.
            if res_l.success and res_r.success:
                res_l, res_r = self._synchronize_trajectories(res_l, res_r)

        if res_l is None:
            res_l = PlanResult.make_error(PlannerErrorCode.DUAL_ARM_PLAN_FAIlED, " dual arm one fail, not input can not copy plan ")
        if res_r is None:
            res_r = PlanResult.make_error(PlannerErrorCode.DUAL_ARM_PLAN_FAIlED, " dual arm one fail, not input can not copy plan ")

        status = "Success" if (res_l.success and res_r.success) else "Fail"
        return DualPlanResult(status=status, left=res_l, right=res_r)

   
    # =========================================================================
    #  Gripper Only Planning
    # =========================================================================
    def plan_gripper(
        self,
        target_value: Union[float, List[float], np.ndarray],
        current_arm_joints: Optional[np.ndarray] = None,
        current_gripper_value: Optional[Union[float,List[float], np.ndarray]] = None,  
        arm: Optional[Literal["left", "right"]] = None,
        duration: float = 1.0,
        time_step: float = 0.05
    ) -> PlanResult:
        """
        Generate a trajectory where only the gripper moves.
        The arm joints remain stationary.
        """
        target_planner = None
        target_gripper_joints = []
        cached_arm_q = None
        cached_gripper_val = None
        active_slot = "single"

        if self.robot_type == "single_arm":
            target_planner = self.planner
            target_gripper_joints = self.gripper_joints
            cached_arm_q = self.planning_state.left_joints  # single arm uses left slot
            cached_gripper_val = self.planning_state.left_gripper
            active_slot = "single"

        elif self.robot_type == "dual_arm":
            if arm == "left":
                target_planner = self.left_planner
                target_gripper_joints = self.left_gripper_joints
                cached_arm_q = self.planning_state.left_joints
                cached_gripper_val = self.planning_state.left_gripper
                active_slot = "left"
            elif arm == "right":
                target_planner = self.right_planner
                target_gripper_joints = self.right_gripper_joints
                cached_arm_q = self.planning_state.right_joints
                cached_gripper_val = self.planning_state.right_gripper
                active_slot = "right"
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Specify arm='left' or 'right' for dual robot")
        if not target_gripper_joints:
             return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "No gripper joints defined")
        num_g_joints = len(target_gripper_joints)

        # --- B. Smart Start ---
        start_raw = current_gripper_value if current_gripper_value is not None else cached_gripper_val
        start_g_arr = self._normalize_gripper_value(start_raw, num_g_joints)
        target_g_arr = self._normalize_gripper_value(target_value, num_g_joints)

        if active_slot == "single": self.planning_state.left_gripper = target_value
        elif active_slot == "left": self.planning_state.left_gripper = target_value
        elif active_slot == "right": self.planning_state.right_gripper = target_value


        # --- C. Interpolation ---
        times = np.arange(0, duration + time_step, time_step)
        n_steps = len(times)
        
        # 1. Arm Part (Hold)
        arm_names = target_planner.active_joint_names
        arm_dof = len(arm_names)

        if current_arm_joints is None: 
            current_arm_joints = cached_arm_q
        
        if current_arm_joints is not None and len(current_arm_joints) >= arm_dof:
            start_q_clean = current_arm_joints[:arm_dof]
        else:
            logger.warning(f"No arm state found for {active_slot}, defaulting to zeros.")
            start_q_clean = np.zeros(arm_dof)
        
        arm_pos = np.tile(start_q_clean, (n_steps, 1))
        arm_vel = np.zeros_like(arm_pos)
        arm_acc = np.zeros_like(arm_pos)

        ee_positions = []
        ee_quaternions = []
        for i in range(n_steps):
            arm_q = arm_pos[i]  # Current arm configuration
            ee_pose = target_planner.kinematics.solve_fk(arm_q)  # [x,y,z,qw,qx,qy,qz]
            ee_positions.append(ee_pose[:3])
            ee_quaternions.append(ee_pose[3:])
        
        ee_positions = np.array(ee_positions)
        ee_quaternions = np.array(ee_quaternions)

        # 2. Gripper Part (Linear Move)

        g_pos_full = np.linspace(start_g_arr, target_g_arr, n_steps)


        if duration > 1e-4:
            g_vel_row = (target_g_arr - start_g_arr) / duration
        else:
            g_vel_row = np.zeros_like(start_g_arr)

        g_vel_full = np.tile(g_vel_row, (n_steps, 1))
        g_acc_full = np.zeros_like(g_pos_full)

        full_pos = np.hstack([arm_pos, g_pos_full])
        full_vel = np.hstack([arm_vel, g_vel_full])
        full_acc = np.hstack([arm_acc, g_acc_full])

        full_names = target_planner.active_joint_names + target_gripper_joints

        res = PlanResult.make_success(
            time=times, 
            pos=full_pos, 
            vel=full_vel, 
            acc=full_acc, 
            joint_names=full_names,
            arm_dof=arm_dof,
            ee_position=ee_positions,
            ee_quaternion=ee_quaternions
        )
        return res
    
   
    
    # =========================================================================
    # Time Synchronization Logic for Dual Arm
    # =========================================================================
    def _synchronize_trajectories(self, res_l: PlanResult, res_r: PlanResult) -> Tuple[PlanResult, PlanResult]:
        """
        时间同步逻辑：
        1. 尝试获取关键帧时间 (waypoint_times)。
        2. 如果具备多点同步条件，计算每段最大耗时，进行非线性时间规整。
        3. 否则回退到简单的总时长同步。
        """
        # 1. 获取关键帧时间
        wps_t_l = res_l.waypoint_times
        wps_t_r = res_r.waypoint_times
        
        # 2. 检查条件：都存在，且关键帧数量一致 (例如都包含 Start + N个waypoints)
        if (wps_t_l is not None and wps_t_r is not None and 
            len(wps_t_l) > 1 and len(wps_t_l) == len(wps_t_r)):
            
            # 计算每段耗时 (Duration per segment)
            durs_l = np.diff(wps_t_l)
            durs_r = np.diff(wps_t_r)
            
            # 取最大值 (每段都对齐到慢的那只手)
            common_durs = np.maximum(durs_l, durs_r)
            
            # 重建时间轴 (0, t1, t2, ...)
            new_key_times = np.concatenate([[0.0], np.cumsum(common_durs)])
            total_duration = new_key_times[-1]
            
            # logger.debug(f"[Sync] Multi-point alignment. Keyframes: {len(new_key_times)}. Total T: {total_duration:.2f}")

            # 执行时间规整
            new_res_l = self._warp_trajectory_time(res_l, wps_t_l, new_key_times)
            new_res_r = self._warp_trajectory_time(res_r, wps_t_r, new_key_times)
            
            # 记录信息
            new_res_l.info['sync_method'] = 'multi_waypoint'
            return new_res_l, new_res_r

        else:
            # 降级：只对齐总时长
            return self._simple_synchronize(res_l, res_r)
        
    def _simple_synchronize(self, res1: PlanResult, res2: PlanResult) -> Tuple[PlanResult, PlanResult]:
        """同步逻辑：只拉齐总时长"""
        t1_end = res1.time[-1] if len(res1.time) > 0 else 0
        t2_end = res2.time[-1] if len(res2.time) > 0 else 0
        target_duration = max(t1_end, t2_end)
        
        if target_duration <= 1e-4: return res1, res2
        
        return self._resample_trajectory(res1, target_duration), self._resample_trajectory(res2, target_duration)

    def _resample_trajectory(self, res: PlanResult, target_dur: float) -> PlanResult:
        """Helper: 简单的线性缩放 (用于 simple sync)"""
        old_dur = res.time[-1]
        if abs(old_dur - target_dur) < 1e-3: return res
        
        scale = old_dur / target_dur
        new_time = np.linspace(0, target_dur, len(res.time))
        mapped_time = new_time * scale
        
        f_pos = interp1d(res.time, res.position, axis=0, kind='linear', fill_value="extrapolate")
        f_vel = interp1d(res.time, res.velocity, axis=0, kind='linear', fill_value="extrapolate")
        f_acc = interp1d(res.time, res.acceleration, axis=0, kind='linear', fill_value="extrapolate")
        
        new_res = PlanResult.make_success(
            time=new_time,
            pos=f_pos(mapped_time),
            vel=f_vel(mapped_time) * scale,
            acc=f_acc(mapped_time) * (scale**2),
            joint_names=res.joint_names,
            arm_dof=res.arm_dof
        )
        # Handle EE
        if len(res.ee_position) > 0:
            f_ee = interp1d(res.time, res.ee_position, axis=0, kind='linear', fill_value="extrapolate")
            new_res.ee_position = f_ee(mapped_time)
        if len(res.ee_quaternion) > 0:
            f_quat = interp1d(res.time, res.ee_quaternion, axis=0, kind='linear', fill_value="extrapolate")
            new_res.ee_quaternion = f_quat(mapped_time)
            
        new_res.info = res.info
        return new_res
    
    def _warp_trajectory_time(self, res: PlanResult, old_key_times: np.ndarray, new_key_times: np.ndarray) -> PlanResult:
        """
        核心函数：时间重整 (Time Warping)
        将轨迹 res 的时间轴，从 old_key_times 映射到 new_key_times。
        """
        final_duration = new_key_times[-1]
        # 保持采样点数，或者根据时长重新采样。这里保持点数，简单。
        num_steps = len(res.time)
        new_time_grid = np.linspace(0, final_duration, num_steps)
        
        # 1. 建立时间映射关系: t_new -> t_old
        # 我们想要知道：对于新的时间点 t_new，它对应原始轨迹的哪个时刻 t_old？
        # 使用分段线性插值计算映射
        mapped_old_times = np.interp(new_time_grid, new_key_times, old_key_times)
        
        # 2. 对位置进行重采样
        f_pos = interp1d(res.time, res.position, axis=0, kind='linear', fill_value="extrapolate")
        new_pos = f_pos(mapped_old_times)
        
        # 3. 重新计算速度和加速度 (基于新位置和新时间网格)
        # 直接使用梯度计算，比缩放原速度更稳健，因为时间缩放是非线性的
        dt = new_time_grid[1] - new_time_grid[0]
        if dt > 1e-6:
            new_vel = np.gradient(new_pos, dt, axis=0)
            new_acc = np.gradient(new_vel, dt, axis=0)
        else:
            new_vel = np.zeros_like(new_pos)
            new_acc = np.zeros_like(new_pos)

        # 4. 构建新结果
        new_res = PlanResult.make_success(
            time=new_time_grid,
            pos=new_pos,
            vel=new_vel,
            acc=new_acc,
            joint_names=res.joint_names,
            arm_dof=res.arm_dof
        )
        
        # 5. 处理 EE 数据
        if len(res.ee_position) > 0:
            f_ee = interp1d(res.time, res.ee_position, axis=0, kind='linear', fill_value="extrapolate")
            new_res.ee_position = f_ee(mapped_old_times)
            
        if len(res.ee_quaternion) > 0:
            f_quat = interp1d(res.time, res.ee_quaternion, axis=0, kind='linear', fill_value="extrapolate")
            raw_quat = f_quat(mapped_old_times)
            # 简单的归一化
            norms = np.linalg.norm(raw_quat, axis=1, keepdims=True)
            new_res.ee_quaternion = raw_quat / (norms + 1e-8)

        # 保留关键信息 (更新 waypoint_times 为新的对齐时间)
        new_res.waypoint_times = new_key_times
        
        return new_res

     # =========================================================================
    # Helpers: Gripper, Coordinate ... 
    # =========================================================================
    def _append_gripper_state(
        self, 
        res: PlanResult, 
        start_val: float,
        end_val: float,
        g_names: List[str]
    ) -> PlanResult:
        """Appends static gripper value to the trajectory."""
        T = len(res.position)
        if T == 0: return res
        n_gripper = len(g_names)

        start_arr = self._normalize_gripper_value(start_val, n_gripper)
        end_arr = self._normalize_gripper_value(end_val, n_gripper)
        g_pos = np.linspace(start_arr, end_arr, T)
    
        duration = res.time[-1] if len(res.time) > 0 and res.time[-1] > 1e-4 else 1.0

        delta = end_arr - start_arr
        vel_val = delta / duration
        g_vel = np.tile(vel_val, (T, 1))
        g_acc = np.zeros((T, n_gripper))

        # Stack
        res.position = np.hstack([res.position, g_pos])
        res.velocity = np.hstack([res.velocity, g_vel])
        res.acceleration = np.hstack([res.acceleration, g_acc])
        
        # Update names if not already present
        if not any(g in res.joint_names for g in g_names):
            res.joint_names = res.joint_names + g_names
            
        return res
    
    def _normalize_gripper_value(self, val: Union[float, List[float], np.ndarray], num_joints: int) -> np.ndarray:
        """input(1.0) tranfer to  (num_joints,) ndarray"""
        if val is None: return np.zeros(num_joints)
        
        if isinstance(val, (float, int)):
            return np.full((num_joints,), float(val))
            
        arr = np.array(val, dtype=float).flatten()
        if len(arr) != num_joints:
            if len(arr) == 1: return np.full((num_joints,), arr[0]) # 单元素数组也广
            if len(arr) > num_joints: return arr[:num_joints]
            return np.pad(arr, (0, num_joints - len(arr)))
        return arr
    
    def _make_hold_plan(self, start_joints: np.ndarray, reference_plan: PlanResult, joint_names: List[str], gripper_val: Optional[float] = None) -> PlanResult:
        """Generate a stationary trajectory based on a reference trajectory"""
        times = reference_plan.time
        pos = np.tile(start_joints, (len(times), 1)) # Repeat N times
        
        res = PlanResult.make_success(
            time=times,
            pos=pos,
            vel=np.zeros_like(pos),
            acc=np.zeros_like(pos),
            joint_names=joint_names
        )
        
        if gripper_val is not None:
            target_g_names = []
            if self.robot_type == "dual_arm":
                if joint_names == self.left_joint_names:
                    target_g_names = self.left_gripper_joints
                elif joint_names == self.right_joint_names:
                    target_g_names = self.right_gripper_joints
            else:
                target_g_names = self.gripper_joints

            if target_g_names:
                res = self._append_gripper_state(res, start_val=gripper_val,end_val=gripper_val, g_names=target_g_names)

        return res
    
    def _world_to_base_transform(self, world_pose: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
        """
        Transform pose from World Frame to Robot Base Frame.
        Input/Output format: [x, y, z, qw, qx, qy, qz]
        """
        from scipy.spatial.transform import Rotation
        
        # Extract
        p_world, q_world = world_pose[:3], world_pose[3:]  # wxyz or xyzw? Assumption: qw, qx, qy, qz
        p_base, q_base = base_pose[:3], base_pose[3:]
        
        # Scipy uses [x, y, z, w]
        def to_scipy(q): return np.array([q[1], q[2], q[3], q[0]])
        def from_scipy(q): return np.array([q[3], q[0], q[1], q[2]])
        
        R_world = Rotation.from_quat(to_scipy(q_world))
        R_base = Rotation.from_quat(to_scipy(q_base))
        
        # Transform Position: p_local = R_base.inv * (p_world - p_base)
        p_local = R_base.inv().apply(p_world - p_base)
        
        # Transform Rotation: R_local = R_base.inv * R_world
        R_local = R_base.inv() * R_world
        q_local = from_scipy(R_local.as_quat())
        
        return np.concatenate([p_local, q_local])



# =============================================================================
# Testing Block
# =============================================================================
if __name__ == "__main__":
    import time
    
    # Setup paths
    piper_urdf = os.path.join(project_root, "assets/robot/piper/urdf/piper_with_gripper.urdf")
    
    # Check if URDF exists to avoid hard crash
    if not os.path.exists(piper_urdf):
        logger.warning(f"URDF file not found at {piper_urdf}. Please check path.")
        # Create a dummy path if you want to test logic without valid URDF (Curobo will likely fail though)
    
    # Define test configuration
    # Note: Added 'joint_names' to prevent initialization errors if backend doesn't infer them
    piper_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    
    dual_config = {
        "mode": "dual_urdf",
        "left": {
            "urdf_path": piper_urdf,
            "base_link": "base_link",
            "ee_link": "tcp_link",
            "base_pose": [0, 0.4, 0, 1, 0, 0, 0],
            "joint_names": piper_joints, # Safety addition
            "gripper_joints": ["joint7"] # Simplified gripper joint
        },
        "right": {
            "urdf_path": piper_urdf,
            "base_link": "base_link",
            "ee_link": "tcp_link",
            "base_pose": [0, -0.4, 0, 1, 0, 0, 0],
            "joint_names": piper_joints, # Safety addition
            "gripper_joints": ["joint7"]
        }
    }

    print("="*60)
    print("Initializing Dual Arm Manager (Curobo)")
    print("="*60)
    
    try:
        manager = PlannerManager(
            robot_type="dual_arm", 
            planner_type="toppra",
            dual_arm_config=dual_config
        )
        
        # Test Data (World Frame)
        start_joints = np.zeros(6)
        left_target = [0.3, 0.4, 0.2, 0.7071, 0.0, 0.7071, 0.0]  # x=0.3 relative to world
        right_target = [0.3, -0.4, 0.2, 0.7071, 0.0, 0.7071, 0.0]
        
        print("\n[Test] Planning Dual Arm Pose...")
        left_goal = ArmMotionGoal(start=start_joints, goal=left_target, gripper=1.0)
        right_goal = ArmMotionGoal(start=start_joints, goal=right_target, gripper=0.0)
        res = manager.plan_dual_arm(left=left_goal, right=right_goal, synchronize=True)

        print(f"Status: {res.status}")
        if res.success:
            l_traj = res.left
            r_traj = res.right
            print(f"Left Duration: {l_traj.time[-1]:.2f}s, Points: {l_traj.position.shape}")
            print(f"Right Duration: {r_traj.time[-1]:.2f}s, Points: {r_traj.position.shape}")
            print("Trajectory shapes (should include gripper): ", l_traj.position.shape) # Expected (N, 6+1) or (N, 6+2) depending on config
            
            # Verify planning state
            print(f"Planning State Left Gripper: {manager.planning_state.left_gripper}")
            print(f"Planning State Right Gripper: {manager.planning_state.right_gripper}")
            print(f"Planning State Source: {manager.planning_state.source}")
        
        print("\n✓ Manager implementation looks good!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()