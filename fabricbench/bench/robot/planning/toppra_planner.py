from operator import pos
import numpy as np
import os
import sys
from typing import List, Optional, Union
from loguru import logger
from warp import quat

try:
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo
    from spatialmath import SE3, UnitQuaternion
except ImportError as e:
    logger.error(f"[ToppraPlanner] Missing dependencies: {e}. Please pip install toppra")
    raise

# path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fabricbench.bench.robot.planning.base_planner import BasePlanner
from fabricbench.bench.robot.kinematics.base_kinematics import BaseKinematics

try:
    from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
    # from fabricbench.bench.robot.kinematics.rtb_kinematics import RtbKinematics
except ImportError as e:
    logger.error(f"[ToppraPlanner] Kinematics not found: {e}. Please ensure Kinematics is installed and properly configured.")
    raise

from fabricbench.bench.common.types import PlanResult
from fabricbench.bench.common.errors import PlannerErrorCode

class ToppraPlanner(BasePlanner):
    def __init__(
        self, 
        urdf_path: str,
        active_joint_names: List[str] = ["joint1","joint2","joint3","joint4","joint5","joint6"],
        kinematics: Union[BaseKinematics,None] = None,  # 必须注入一个 Kinematics 实例
        base_link: str = "base_link",
        ee_link: str = "tcp_link",
        # velocity_limits: float = 3.0,
        velocity_limits: Union[float, List[float], np.ndarray] = np.deg2rad([180, 195, 180, 225, 225, 225]),
        acceleration_limits: float = 10.0,
        jerk_limits: float = 1.0,
        trajectory_frequency: int = 50,
        **kwargs
    ):
        super().__init__(urdf_path, active_joint_names, **kwargs)
        
        # default kinematics using Pinocchio
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
        
        # !!!!TODO: Active Name should get from urdf, not to set here
        self.dof = len(active_joint_names)

        self.trajectory_frequency = trajectory_frequency
        
        # set velocity and acceleration limits
        self._set_limits(velocity_limits, acceleration_limits)
        
    def _set_limits(self, v_lim, a_lim):
        """Internal helper: set velocity and acceleration limits"""
        if isinstance(v_lim, (int, float)):
            self.vlims = np.array([v_lim] * self.dof)
        else:
            self.vlims = np.array(v_lim)

        if isinstance(a_lim, (int, float)):
            self.alims = np.array([a_lim] * self.dof)
        else:
            self.alims = np.array(a_lim)

    def _run_toppra(self, waypoints: np.ndarray, scaling_factor: float = 1.0,
                    acceleration_scaling_factor: Optional[float] = None,
                    ee_link: Optional[str] = None,
                    keyframe_indices: Optional[List[int]] = None) -> PlanResult:
        """Core: Perform Toppra time parameterization on joint space waypoints.

        Per-segment scaling (NEW):
          - scaling_factor 可以是 float (全局) 或 list (per-segment).
          - list 时必须配合 keyframe_indices, len(list) == len(keyframe_indices) - 1.
          - 用 JointVelocityConstraintVarying 让 vlim 随 path param s 变化,
            TOPPRA 优化保证 v 在 segment 边界处连续 (不像分段 concat 会 v=0).
          - acceleration 无 Varying API, 取 min(scales) 保守兜底.
        """
        if len(waypoints) < 2:
            return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "Not enough waypoints for Toppra")

        # --- 归一化 scaling 为 list[float], per-segment ---
        vel_is_list = isinstance(scaling_factor, (list, tuple))
        vel_scales = list(scaling_factor) if vel_is_list else [float(scaling_factor)]
        if acceleration_scaling_factor is None:
            acc_scales = list(vel_scales)  # default: same as vel
        elif isinstance(acceleration_scaling_factor, (list, tuple)):
            acc_scales = [float(x) for x in acceleration_scaling_factor]
        else:
            acc_scales = [float(acceleration_scaling_factor)]

        try:
            # 1. Construct path (Spline Interpolation) with normalized s ∈ [0,1]
            path_scalars = np.linspace(0, 1, len(waypoints))
            path = ta.SplineInterpolator(path_scalars, waypoints)

            # 2. Set constraints
            use_varying = (
                len(vel_scales) > 1
                and keyframe_indices is not None
                and len(keyframe_indices) == len(vel_scales) + 1
            )
            if use_varying:
                # Map each segment [keyframe_indices[i], keyframe_indices[i+1]]
                # to s-range in [0, 1]. vlim_func(s) picks segment i and returns
                # base_vlim * vel_scales[i].
                seg_s_bounds = [float(path_scalars[idx]) for idx in keyframe_indices]
                base_vlim = np.vstack([-self.vlims, self.vlims]).T  # (n_dof, 2)

                def _vlim_func(s, _bounds=seg_s_bounds, _scales=vel_scales, _base=base_vlim):
                    for i in range(len(_scales)):
                        if s <= _bounds[i + 1] + 1e-9:
                            return _base * _scales[i]
                    return _base * _scales[-1]

                pc_vel = constraint.JointVelocityConstraintVarying(_vlim_func)
                # acceleration 无 Varying API. 取 max(acc_scales) 给足加速余量:
                # - velocity 本身 per-segment varying, 不会超 vlim
                # - min(acc) 会让慢段拖累快段 (短 approach 段加速不到峰值 → 实际慢于预期)
                # - max(acc) 让 TOPPRA 优化在各段内自然选择, 快段真正跑快, 慢段仍受 vlim 限制
                a_scale = float(max(acc_scales))
            else:
                # 全局 single scaling
                vlim = np.vstack([-self.vlims, self.vlims]).T * vel_scales[0]
                a_scale = float(acc_scales[0]) if acc_scales else float(vel_scales[0])
                pc_vel = constraint.JointVelocityConstraint(vlim)

            alim = np.vstack([-self.alims, self.alims]).T * a_scale
            pc_acc = constraint.JointAccelerationConstraint(alim)

            # 3. Solve
            instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper="seidel")
            jnt_traj = instance.compute_trajectory(0, 0) # 0 start vel, 0 end vel
            
            if jnt_traj is None:
                logger.warning("[ToppraPlanner] Toppra optimization failed (constraints too tight?)")
                return PlanResult.make_error(PlannerErrorCode.TIME_PARAM_FAILED, "Toppra optimization failed")

            # 4. Sampling
            duration = jnt_traj.get_duration()
            # Default 50Hz or based on duration, at least 10 points
            num_samples = max(10, int(duration * self.trajectory_frequency)) 
            ts = np.linspace(0, duration, num_samples)
            
            qs = jnt_traj.eval(ts)
            qds = jnt_traj.evald(ts)
            qdds = jnt_traj.evaldd(ts)

            # calculate Fk
            N = len(qs)
            ee_positions = np.zeros((N, 3))
            ee_quaternions = np.zeros((N, 4))
            for i, q in enumerate(qs):
                ee_pose = self.kinematics.solve_fk(q, ee_link=ee_link)  # [x,y,z,qw,qx,qy,qz]
                ee_positions[i] = ee_pose[:3]
                ee_quaternions[i] = ee_pose[3:]
            
            res = PlanResult.make_success(
                time=ts, pos=qs, vel=qds, acc=qdds, 
                joint_names=self.active_joint_names,
                ee_position=ee_positions,
                ee_quaternion=ee_quaternions,
            )


            if keyframe_indices is not None:
                key_times = []
                last_search_idx = 0
                
                for idx in keyframe_indices:
                    if idx >= len(waypoints): 
                        continue
                    
                    target_q = waypoints[idx]
                    
                    # Search in the remaining slice of the trajectory to preserve time monotonicity
                    search_slice = qs[last_search_idx:]
                    if len(search_slice) == 0:
                        key_times.append(ts[-1])
                        continue

                    # Find point in generated trajectory closest to the target waypoint configuration
                    dists = np.linalg.norm(search_slice - target_q, axis=1)
                    local_min_idx = np.argmin(dists)
                    global_idx = last_search_idx + local_min_idx
                    
                    key_times.append(ts[global_idx])
                    
                    # Advance search start to avoid backtracking (assumes sequential waypoints)
                    last_search_idx = global_idx
                
                res.waypoint_times = key_times
                # logger.debug(f"key_times: {key_times}")
                # logger.debug(f"keyframe_indices:{keyframe_indices} ")
            
            return res

        except Exception as e:
            logger.exception(f"[ToppraPlanner] Exception in _run_toppra: {e}")
            return PlanResult.make_error(PlannerErrorCode.TIME_PARAM_FAILED, str(e))

    def plan_to_joint_config(
        self, 
        start_joints: np.ndarray, 
        goal_joints: np.ndarray,
        **kwargs
    ) -> PlanResult:
        """Joint space planning: linear interpolation + Toppra"""
        velocity_scale = kwargs.get("velocity_scale", 1.0)
        acceleration_scale = kwargs.get("acceleration_scale", None)
        num_interp = 20 # Number of interpolation points

        # Simple linear interpolation
        diff = goal_joints - start_joints
        waypoints = start_joints + np.outer(np.linspace(0, 1, num_interp), diff)

        return self._run_toppra(waypoints, scaling_factor=velocity_scale,
                                acceleration_scaling_factor=acceleration_scale)

    def plan_to_pose(
        self, 
        start_joints: np.ndarray, 
        goal_pose: np.ndarray, # [x, y, z, qw, qx, qy, qz]
        pose_constraint:Optional[Union[np.ndarray, List[float]]]=None,
        constraint_frame: str = "world",
        **kwargs
    ) -> PlanResult:
        """Cartesian space planning: linear interpolation -> IK -> Toppra"""
        # Reuse plan_waypoints but only pass one target point
        if pose_constraint is not None:
            constraint_arg = [pose_constraint]
        else:
            constraint_arg = None
        return self.plan_waypoints(start_joints, [goal_pose],pose_constraint=constraint_arg, constraint_frame=constraint_frame, **kwargs)
    

    def plan_waypoints(
        self, 
        start_joints: np.ndarray, 
        waypoints: List[np.ndarray], # List of [x,y,z,qw,qx,qy,qz]
        pose_constraint: Optional[Union[np.ndarray, List[float], List[List[float]]]] = None, 
        constraint_frame: str = "world",
        interpolation_mode: Union[str, List[str]] = "joint",
        **kwargs
    ) -> PlanResult:
        """
        Cartesian path planning (Waypoint Following)
        Features:
        1. Linear interpolation (ctraj) in Cartesian space connecting each Waypoint
        2. IK for each interpolated point
        3. Generate joint trajectory
        
        Args:
            start_joints: Starting joint configuration
            waypoints: List of target poses [x,y,z,qw,qx,qy,qz]
            **kwargs:
                - velocity_scale: float, scaling factor for velocity limits
                - steps: int, number of interpolation points between waypoints
                - interpolation_mode: str, "cartesian" or "joint" (default: "joint")
                - pose_constraint: Pose constraints [x, y, z, rx, ry, rz], [0,1]
        """
        velocity_scale = kwargs.get("velocity_scale", 1.0)
        acceleration_scale = kwargs.get("acceleration_scale", None)
        steps_per_segment = kwargs.get("steps_per_segment", 10) # Number of interpolation points between each waypoint
        ee_link = kwargs.get("ee_link", None)
        # end_first_seed: 先解最后一个 waypoint (final goal) 的 IK, 用它作为
        # 中间 waypoints 的 seed. 效果: 中间点 (如 via) 的 IK 解倾向于接近 final
        # joint config → minimal wrist rotation across the whole path.
        # 默认 False 保持 sequential seed 原行为.
        end_first_seed = kwargs.get("end_first_seed", False)
        # debug_ik: 打印每个 waypoint 的 IK 解 + FK 验证误差 (position + rotation).
        # 方便调试 via/grasp 的 EE 朝向是否符合预期.
        debug_ik = kwargs.get("debug_ik", False)
        # ground_check: 是否在 IK 求解中启用地面穿透过滤. 默认 False (新).
        # grasp 点 z~0 时 ground_check 经常把所有 seed 全拒绝 → fallback 到 least-penetrating,
        # 既慢又会刷 warning. 关心地面穿透的 task 应在调用处显式 ground_check=True.
        ground_check = kwargs.get("ground_check", False)

        num_wps = len(waypoints)
        if num_wps == 0: return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, "No waypoints provided")

        if self.kinematics is None: return PlanResult.make_error(PlannerErrorCode.BACKEND_UNAVAILABLE, "Kinematics solver missing")

        # --- 1. Standardize Inputs ---
        # Ensure modes, constraints, and steps are all lists of length N
        modes = self._ensure_list_input(interpolation_mode, num_wps, "interpolation_mode", default="joint")
        constraints = self._ensure_list_input(pose_constraint, num_wps, "pose_constraint", default=[1]*6)
        steps_list = self._ensure_list_input(steps_per_segment, num_wps, "steps", default=10)

        # --- 2. Global Skeleton IK (Solve IK for all key waypoints first) ---
        skeleton_joints = [start_joints]

        if end_first_seed and num_wps >= 2:
            # (a) 先解 FINAL waypoint (seed=start_joints).
            final_sol = self.kinematics.solve_ik(
                target_pose=waypoints[-1],
                seed_joints=start_joints,
                pose_constraint=constraints[-1],
                constraint_frame=constraint_frame,
                ee_link=ee_link,
                ground_check=ground_check,
            )
            if final_sol is None:
                msg = (
                    f"IK Failed at FINAL waypoint (end_first_seed mode), "
                    f"Pose: {np.round(waypoints[-1][:3], 3)}"
                )
                logger.warning(f"[ToppraPlanner] {msg}")
                return PlanResult.make_error(PlannerErrorCode.IK_FAILED_GLOBAL, msg)

            # (b) 中间 waypoints 用 final_sol 作 seed (非 sequential).
            for i in range(num_wps - 1):
                sol = self.kinematics.solve_ik(
                    target_pose=waypoints[i],
                    seed_joints=final_sol,
                    pose_constraint=constraints[i],
                    constraint_frame=constraint_frame,
                    ee_link=ee_link,
                    ground_check=ground_check,
                )
                if sol is None:
                    msg = (
                        f"IK Failed at intermediate wp {i} (end_first_seed mode), "
                        f"Pose: {np.round(waypoints[i][:3], 3)}"
                    )
                    logger.warning(f"[ToppraPlanner] {msg}")
                    return PlanResult.make_error(PlannerErrorCode.IK_FAILED_GLOBAL, msg)
                skeleton_joints.append(sol)

            # (c) Append final at the end.
            skeleton_joints.append(final_sol)
        else:
            # Sequential seed (原默认): 上一个 wp 的解作为下一个 wp 的 seed.
            curr_seed = start_joints
            for i, wp_pose in enumerate(waypoints):
                sol = self.kinematics.solve_ik(
                    target_pose=wp_pose,
                    seed_joints=curr_seed,
                    pose_constraint=constraints[i],
                    constraint_frame=constraint_frame,
                    ee_link=ee_link,
                    ground_check=ground_check,
                )
                if sol is None:
                    msg = f"IK Failed at waypoint index {i} (Pose: {np.round(wp_pose[:3], 3)})"
                    logger.warning(f"[ToppraPlanner] {msg}")
                    return PlanResult.make_error(PlannerErrorCode.IK_FAILED_GLOBAL, msg)
                skeleton_joints.append(sol)
                curr_seed = sol
        
        # --- 3. Construct Trajectory Segment by Segment ---
        full_path_joints = [skeleton_joints[0]] # Start with initial config
        waypoint_indices = [0]
        # segment_indices: 每段**结束**的 index (len = num_wps + 1). 用于 per-segment
        # velocity scaling. 与 waypoint_indices 区分: 后者 cart 段可能 extend 多个插值点,
        # 破坏 "N+1 entries for N segments" 的语义.
        segment_indices = [0]

        for i in range(num_wps):
            mode = modes[i]
            target_joints_knot = skeleton_joints[i+1] # The pre-solved IK for this waypoint
            target_pose_knot = waypoints[i]           # The cartesian target
            current_seg_steps = steps_list[i]
            current_constraint = constraints[i]

            # The start of this segment is the end of the previous one
            segment_start_joints = full_path_joints[-1]

            if mode == "joint":
                full_path_joints.append(target_joints_knot)
                waypoint_indices.append(len(full_path_joints) - 1)
                segment_indices.append(len(full_path_joints) - 1)

            elif mode == "cartesian":
                # Note: We re-calculate FK here to ensure the line starts exactly where the robot IS
                seg_start_pose = self.kinematics.solve_fk(segment_start_joints, ee_link=ee_link)
                
                # 2. Interpolate
                cartesian_path_joints = self._cartesian_interpolate(
                    start_joints=segment_start_joints,
                    start_pose=seg_start_pose,
                    target_pose=target_pose_knot,
                    steps=current_seg_steps,
                    constraint=current_constraint,
                    constraint_frame=constraint_frame,
                    ee_link=ee_link,
                    ground_check=ground_check,
                )
                
                if cartesian_path_joints is None:
                    return PlanResult.make_error(
                        PlannerErrorCode.IK_FAILED_GLOBAL,
                        f"Cartesian interpolation failed in segment {i}"
                    )
                
                # 3. Append points
                # _cartesian_interpolate returns [start, p1, p2, ... target]. 
                # We slice [1:] to avoid duplicating the start point which is already in full_path_joints.
                # Note: The last point of cartesian_path_joints should theoretically be very close to target_joints_knot,
                # but due to numerical drift in IK, it might differ slightly. 
                # To maintain the "Global Skeleton" consistency, we can force the last point to be the pre-solved knot,
                # OR trust the cartesian interpolation. Trusting interpolation is usually smoother for the line.
                cartesian_path_joints[-1] = target_joints_knot
                full_path_joints.extend(cartesian_path_joints[1:])

                # TODO： 这里可以同步每一段，也可在每段中同步关键点索引
                # waypoint_indices.append(len(full_path_joints) - 1)
                # 如果插值点全部同步
                waypoint_indices.extend(range(len(full_path_joints) - len(cartesian_path_joints) + 1, len(full_path_joints)))
                # segment_indices 只记这段的终点, 不记中间插值点
                segment_indices.append(len(full_path_joints) - 1)
                
                
            else:
                return PlanResult.make_error(PlannerErrorCode.INVALID_INPUT, f"Unknown mode: {mode}")

        full_path_joints = np.array(full_path_joints)
        # logger.debug(f"skeleton_joints: {len(skeleton_joints)}")
        # logger.debug(f"full joint_path: {len(full_path_joints)}")
        # logger.debug(f"waypoint_indices: {waypoint_indices}")
        # --- 4. Run Toppra ---
        # 如果 velocity_scale 是 list (per-segment), 用 segment_indices (N+1 entries) 作
        # 边界. 否则 keyframe_indices (可能包含 cart 插值点) 不影响 per-segment 判断逻辑.
        vel_is_list = isinstance(velocity_scale, (list, tuple, np.ndarray))
        if vel_is_list and len(velocity_scale) == num_wps:
            return self._run_toppra(full_path_joints, scaling_factor=velocity_scale,
                                    acceleration_scaling_factor=acceleration_scale,
                                    ee_link=ee_link, keyframe_indices=segment_indices)
        return self._run_toppra(full_path_joints, scaling_factor=velocity_scale,
                                acceleration_scaling_factor=acceleration_scale,
                                ee_link=ee_link, keyframe_indices=waypoint_indices)
        

    
    def _cartesian_interpolate(
        self,
        start_joints: np.ndarray,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        steps: int,
        constraint: List[float],
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        ground_check: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Cartesian space linear interpolation (SE3)
        
        Args:
            start_joints: Starting joint configuration
            start_pose: Starting end-effector pose [x,y,z,qw,qx,qy,qz]
            target_pose: Target end-effector pose [x,y,z,qw,qx,qy,qz]
            steps: Number of interpolation points
            constraint: Position/orientation constraint mask [x,y,z,rx,ry,rz]
            constraint_frame: "world" or "local"
            ee_link: End-effector link name
            
        Returns:
            Joint trajectory (Nx6) or None if IK fails
        """
        def to_se3(pose):
            return SE3(pose[:3]) * UnitQuaternion(pose[3:]).SE3()
        
        # Generate SE3 trajectory
        T_start = to_se3(start_pose)
        T_target = to_se3(target_pose)
        s_vals = np.linspace(0, 1, steps)
        T_sequence = [T_start.interp(T_target, s) for s in s_vals]
        
        # IK for each interpolated pose
        joint_traj = [start_joints]
        curr_joints = start_joints
        
        for i, T in enumerate(T_sequence):
            if i == 0:  # Skip start (already added)
                continue
            
            # Extract position and quaternion
            pos = T.t
            quat = UnitQuaternion(T.R).A
            target_vec = np.concatenate([pos, quat])
            
            # Solve IK
            sol = self.kinematics.solve_ik(
                target_pose=target_vec,
                seed_joints=curr_joints,
                pose_constraint=constraint,
                constraint_frame=constraint_frame,
                ee_link=ee_link,
                ground_check=ground_check,
            )
            
            if sol is None:
                logger.warning(f"IK failed at interpolation step {i}/{steps}")
                return None
            
            joint_traj.append(sol)
            curr_joints = sol
        
        return np.array(joint_traj)

    def _ensure_list_input(self, input_val, length, name, default=None):
        """Helper to ensure input is a list of correct length"""
        if input_val is None:
            return [default] * length
        if isinstance(input_val, (list, np.ndarray)):
            # Check if it's a list of lists (for constraints) or just a flat list (for single constraint broadcast)
            # This is a bit tricky for pose_constraints (list of 6 floats) vs list of pose_constraints
            if name == "pose_constraint":
                # If element 0 is a number, it's a single constraint -> broadcast
                if len(input_val) > 0 and isinstance(input_val[0], (int, float)):
                    return [input_val] * length
            
            # If length matches, return it
            if len(input_val) == length:
                return input_val
            
            # If input is a single item wrapped in list but length doesn't match, implies broadcast intent failed or error
            # For simplicity, if it's not matching length, we treat strictly unless it looks like a single item to broadcast
            if len(input_val) != length and name != "pose_constraint":
                 # Try to broadcast if it looks like a single item? 
                 # Better to be strict: if list provided, must match length.
                 logger.warning(f"[{name}] Input list length {len(input_val)} != waypoints length {length}. This might cause errors.")
                 return input_val
        
        # Scalar case -> broadcast
        return [input_val] * length
       

if __name__ == "__main__":
    import time
    import numpy as np
    from fabricbench.bench.robot.utils.visualize_trajectory import plot_plan_trajectory
    # Path setup
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_path = os.path.join(
        project_root, 
        "assets/robot/piper/urdf/piper_with_gripper.urdf"
    )
    
    start_joints = np.array([0.0, 0.2, -0.2, 0.0, 0.0, 0.0])
    goal_joints = np.array([0.5, 0.3, -0.2, 0.0, 0.4, 0.1])
    target_pose = np.array([0.3, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

    print("=" * 60)
    print("Testing ToppraPlanner")
    print("=" * 60)

    planner = ToppraPlanner(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="tcp_link",
        active_joint_names=["joint1","joint2","joint3","joint4","joint5","joint6"],
        velocity_limits=1.5,
        acceleration_limits=0.8,
        trajectory_frequency=50,
    )


   # --- 1.Test Joint Planning ---
    print("\n🔵 [Test 1] Plan to Joint Config...")
    t_start = time.time()
    res = planner.plan_to_joint_config(start_joints, goal_joints)
    
    print(f"   计算耗时: {(time.time() - t_start)*1000:.2f} ms")
    print(f"   规划状态: {res.status}")     
    print(f"   是否成功: {res.success}")

    if res.success:
        print(f"   轨迹点数: {len(res.position)}")
        print(f"   总时长:   {res.time[-1]:.2f} s")

    # --- 测试 2: 笛卡尔空间 Pose 规划 ---
    print("\n🟢 [Test 2] Plan to Pose (Linear Cartesian)...")
    t_start = time.time()
    res = planner.plan_to_pose(start_joints, target_pose)
    
    print(f"   计算耗时: {(time.time() - t_start)*1000:.2f} ms")
    print(f"   规划状态: {res.status}")
    
    if res.success:
        print(f"   轨迹点数: {len(res.position)}")
        # plot_plan_trajectory(res)
    else:
        print(f"   错误代码: {res.error_code}")
        print(f"   错误信息: {res.error_msg}")

    # --- 测试 3: Waypoints 序列规划（统一设置）---
    print("\n🟣 [Test 3] Plan Waypoints Sequence.")
    waypoints = [
        np.array([0.3,  0.1, 0.2, 1, 0, 0, 0]),
        np.array([0.3, -0.1, 0.2, 1, 0, 0, 0]),
        np.array([0.3,  0.0, 0.3, 1, 0, 0, 0]),
    ]
    res = planner.plan_waypoints(start_joints, waypoints, velocity_scale=0.8)
    
    print(f"   规划状态: {res.status}")
    if res.success:
        print(f"   总时长: {res.time[-1]:.2f} s")

    # ---  测试 4: 每个waypoint单独设置ignore_orientation ---
    print("\n🟡 [Test 4] Plan Waypoints with Individual Orientation Flags...")
    waypoints = [
        np.array([0.3,  0.1, 0.2, 1, 0, 0, 0]),      
        np.array([0.3, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0]),  
        np.array([0.3, -0.1, 0.2, 1, 0, 0, 0]),
        np.array([0.259, -0.27, 0.056, 1, 0, 0, 0 ])
    ]
    # 第二个点要姿态，其他不要
    waypoint_constraints = [
        [1, 1, 1, 0, 0, 0],    # WP1: Position only → mask [1,1,1,0,0,0]
        [1, 1, 1, 1, 1, 1],    # WP2: Full constraint → mask [1,1,1,1,1,1]
        [0.8, 0.3, 0.9, 0, 0, 0],  # WP3: Soft → mask [1,0,1,0,0,0]
        [1, 1, 1, 0, 0, 0],
    ]
    

    
    res = planner.plan_waypoints(
        start_joints, 
        waypoints, 
        velocity_scale=0.8,
        pose_constraint=waypoint_constraints 
    )
    
    print(f"   规划状态: {res.status}")
    if res.success:
        print(f"   总时长: {res.time[-1]:.2f} s")
        print(f"   ✅ 成功处理了不同waypoint的姿态要求！")
    else:
        print(f"   错误: {res.error_msg}")


    # --- 测试 4b: 笛卡尔空间插值（更严格）---
    print("\n🟡 [Test 4b] Plan Waypoints (Cartesian Space Interpolation)...")
    waypoints_easy = [
        np.array([0.3, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0]),      
        np.array([0.3, 0.1, 0.2, 0.7071, 0.0, 0.7071, 0.0]),   
    ]
    
    res = planner.plan_waypoints(
        start_joints, 
        waypoints_easy, 
        velocity_scale=0.8,
        interpolation_mode="cartesian",  # 🔥 笛卡尔空间插值
        steps=20  # 增加插值点
    )
    
    print(f"   规划状态: {res.status}")
    if res.success:
        print(f"   ✅ 笛卡尔插值成功！")

    print("\n✓ All tests passed successfully!")