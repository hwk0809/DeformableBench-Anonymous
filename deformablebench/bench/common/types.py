import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
from .errors import PlannerErrorCode
from enum import Enum

# ObservationMessage is the sim env's output contract -> defined in the sim layer.
# Re-exported here so bench code keeps importing it from common.types unchanged
# (sim never depends on bench).
from deformablebench.sim.types import ObservationMessage  # noqa: F401

class ArmSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    
    @property
    def opposite(self):
        """get the opposite arm side"""
        return ArmSide.RIGHT if self == ArmSide.LEFT else ArmSide.LEFT
    
    def __str__(self):
        return self.value

@dataclass
class PlanningState:
    """规划器的工作状态副本。

    每次规划前需要通过 sync_from_obs() 从观测同步到真实值。
    连续规划时通过 advance() 前推到轨迹终点（预测值）。

    source 标记（用于调试/日志，不做强制校验）:
    - "uninitialized": 未初始化，不可用于规划
    - "observed":  从传感器/仿真观测同步（可信）
    - "predicted": 从规划结果前推（仿真精确，真机会累积误差）

    Sim vs Real:
        两种环境使用相同的 sync_from_obs/advance 机制。
        区别在于 task 层调用 sync_from_obs() 的频率：
        - 仿真：每个 stage 开头同步一次，中间可连续 advance
        - 真机：每步前同步纠偏（保守），或每 N 步（激进）
    """
    left_joints: Optional[np.ndarray] = None     # 左臂关节角
    right_joints: Optional[np.ndarray] = None    # 右臂关节角
    left_gripper: Optional[float] = None         # 左夹爪值
    right_gripper: Optional[float] = None        # 右夹爪值
    source: str = "uninitialized"                # "uninitialized" | "observed" | "predicted"
    advances_since_sync: int = 0                 # 距上次 sync_from_obs 前推了几步

# 向后兼容
RobotState = PlanningState


@dataclass
class ArmMotionGoal:
    """单臂运动规划的完整目标描述。
    """
    start: np.ndarray                   # 起始关节角 (arm_dof,)
    goal: Optional[Union[np.ndarray, List]] = None   # 目标：pose[7], joints[N], waypoints list。None = hold
    goal_type: str = "pose"             # "joint" | "pose" | "waypoints"
    gripper: Optional[float] = None     # 夹爪值 [0,1]，None = 保持当前
    pose_constraint: Optional[Union[np.ndarray, List[float], List[List[float]]]] = None
    constraint_frame: str = "world"     # "world" | "local"
    ee_link: Optional[str] = None       # TCP 链接名，None = 使用 planner 默认


@dataclass
class PlanResult:
    # 核心状态
    status: str  # "Success" or "Fail"
    error_code: PlannerErrorCode = PlannerErrorCode.NONE
    error_msg: str = ""

    arm_dof: int = 0 # degrees of freedom for the arm
    
    joint_names: List[str] = field(default_factory=list) # Arm + Gripper, same as position dimension
    
    # trajectory data
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    position: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([]))

    # end position and rotation
    ee_position: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3) [x, y, z], arm/ arm + gripper 
    ee_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 4) [qw, qx, qy, qz]
    
    # additional info (e.g., debug losses, iteration counts, optional)
    info: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Provide a quick property to check if successful"""
        return self.status == "Success" and self.error_code == PlannerErrorCode.NONE

    @property
    def duration(self) -> float:
        """Get the total duration of the trajectory"""
        if len(self.time) > 0:
            return self.time[-1]
        return 0.0
    
    @property
    def waypoint_times(self) -> Optional[np.ndarray]:
        return self.info.get('waypoint_times')

    @waypoint_times.setter
    def waypoint_times(self, times: np.ndarray):
        self.info['waypoint_times'] = np.array(times)


    def __post_init__(self):
        """Data validation logic"""
        # If successful, ensure trajectory data dimensions align
        if self.success and len(self.time) > 0:
            n_steps = len(self.time)
            if len(self.position) != n_steps:
                raise ValueError(f"Data Mismatch: Time has {n_steps} steps, but Position has {len(self.position)}")
            if len(self.ee_position) > 0 and len(self.ee_position) != n_steps:
                raise ValueError(f"EE position mismatch: expected {n_steps}, got {len(self.ee_position)}")
            if len(self.ee_quaternion) > 0 and len(self.ee_quaternion) != n_steps:
                raise ValueError(f"EE quaternion mismatch: expected {n_steps}, got {len(self.ee_quaternion)}")
            
            # Additional validations can be added here as needed

    @classmethod
    def make_error(cls, code: PlannerErrorCode, msg: str = "") -> 'PlanResult':
        """ 
           usage: return PlanResult.make_error(PlannerErrorCode.IK_FAILED, "xxxx")
        """
        return cls(
            status="Fail",
            error_code=code,
            error_msg=msg,
            # other fields default to empty
        )

    @classmethod
    def make_success(
        cls, 
        time: np.ndarray, 
        pos: np.ndarray, 
        vel: Optional[np.ndarray] = None, 
        acc: Optional[np.ndarray] = None, 
        joint_names: List[str] = None,
        ee_position: Optional[np.ndarray] = None,
        ee_quaternion: Optional[np.ndarray] = None,
        arm_dof: Optional[int] = None,
    ) -> 'PlanResult':
        """
              usage: return PlanResult.make_success(time, pos, vel, acc, joint_names)
        """
        # make sure vel/acc is not None
        if vel is None: vel = np.array([])
        if acc is None: acc = np.array([])
        if joint_names is None: joint_names = []
        if ee_position is None: ee_position = np.array([])
        if ee_quaternion is None: ee_quaternion = np.array([])

        if arm_dof is None:
            arm_dof = len(joint_names)

        return cls(
            status="Success",
            error_code=PlannerErrorCode.NONE,
            time=time,
            position=pos,
            velocity=vel,
            acceleration=acc,
            joint_names=joint_names,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            arm_dof=arm_dof,
        )
    

    @property
    def end_position(self) -> Optional[np.ndarray]:
        """Get the final end-effector position if available"""
        if len(self.ee_position) > 0:
            return self.ee_position[-1]
        return None

    @property
    def end_quaternion(self) -> Optional[np.ndarray]:
        """Get the final end-effector quaternion if available"""
        if len(self.ee_quaternion) > 0:
            return self.ee_quaternion[-1]
        return None
    
    @property
    def final_full_position(self) -> Optional[np.ndarray]:
        """Get the final joint positions if available, arm + gripper"""
        if len(self.position) > 0:
            return self.position[-1]
        return None
    
    @property
    def final_arm_position(self) -> Optional[np.ndarray]:
        """arm joint positions only"""
        full = self.final_full_position
        if full is not None:
            return full[:self.arm_dof] 
        return None
    
    @property
    def final_gripper_position(self) -> Optional[np.ndarray]:
        full = self.final_full_position
        if full is not None:
            return full[self.arm_dof:]
        return None

@dataclass
class DualPlanResult:
    status: str
    left: Optional['PlanResult'] = None
    right: Optional['PlanResult'] = None
    error_msg: str = ""
    time_synchronize: Optional[bool] = None

    @property
    def success(self) -> bool:
        return self.status == "Success" and \
               (self.left is None or self.left.success) and \
               (self.right is None or self.right.success)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    @classmethod
    def make_error(cls, code: PlannerErrorCode, msg: str = "") -> 'DualPlanResult':
        return cls(
            status="Fail",
            left=None,
            right=None,
            error_msg=msg
        )
    
    @classmethod
    def make_success(cls,res_l:PlanResult, res_r:PlanResult) -> 'DualPlanResult':
        return cls(
            status="Success",
            left=res_l,
            right=res_r,
            error_msg=""
        )



@dataclass
class ExecutionStep:
    """
    the minimal execution step
    """
    plan: DualPlanResult        # physical data
    settle_time: float = 0.0    # execution strategy: how long to settle
    description: str = ""       # debug info