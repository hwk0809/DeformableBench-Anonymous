from abc import ABC, abstractmethod
from typing import List, Dict, Optional,Union
import numpy as np

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


from fabricbench.bench.common.types import PlanResult
from fabricbench.bench.common.errors import PlannerErrorCode

class BasePlanner(ABC):
    """
    All planners should inherit from this base class.
    """
    
    def __init__(self, urdf_path: str, active_joint_names: List[str], **kwargs):
        """
        Args:
            urdf_path: Robot URDF file path
            active_joint_names: List of joint names controlled by this planner (used to populate PlanResult)
            **kwargs: Other common parameters
        """
        self.urdf_path = urdf_path
        self.active_joint_names = active_joint_names
    
    @abstractmethod
    def plan_to_joint_config(
        self, 
        start_joints: np.ndarray, 
        goal_joints: np.ndarray,
        **kwargs
    ) -> PlanResult:
        """
        Joint space planning interface.
        
        Args:
            start_joints: Starting joint angles (numpy array)
            goal_joints: Goal joint angles (numpy array)
            **kwargs: Optional parameters (e.g., velocity_scale, time_step)
            
        Returns:
            PlanResult: Contains trajectory (time, pos, vel, acc) or error information
        """
        pass
    
    @abstractmethod
    def plan_to_pose(
        self, 
        start_joints: np.ndarray, 
        goal_pose: np.ndarray,
        pose_constraint: Optional[Union[np.ndarray, List[float]]] = None,
        **kwargs
    ) -> PlanResult:
        """
        Cartesian space planning interface (IK + path planning).
        
        Args:
            start_joints: Starting joint angles
            goal_pose: Goal pose [x, y, z, qw, qx, qy, qz] (Note the quaternion order should be consistent with Manager)
            pose_constraint: [x, y, z, rx, ry, rz] weights (0-1)
            **kwargs: Optional parameters
            
        Returns:
            PlanResult: Contains trajectory (time, pos, vel, acc) or error information
        """
        pass

    @abstractmethod
    def plan_waypoints(
        self, 
        start_joints: np.ndarray, 
        waypoints: List[np.ndarray],
        pose_constraint: Optional[Union[np.ndarray, List[List[float]]]] = None,
        **kwargs
    ):
        """Plan through multiple waypoints"""
        pass