from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict
import numpy as np

ConstraintType = Union[
    np.ndarray, 
    List[float], 
    Dict[str, Union[np.ndarray, List[float]]] # 新增：支持字典
]


class BaseKinematics(ABC):
    """Base class for robot kinematics solvers."""
    
    def __init__(self, urdf_path: str, base_link: str, ee_link: str):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        self.joint_limits = None
        self.num_joints = None
    
    @abstractmethod
    def solve_fk(self, joint_angles: np.ndarray, ee_link: Optional[str] = None) -> np.ndarray:
        """Forward kinematics: joint angles -> end-effector pose.
        
        Args:
            joint_angles: List of joint angles in radians
            ee_link: Optional end-effector link name override (defaults to self.ee_link)
            
        Returns:
            pose: [x, y, z, qw, qx, qy, qz] numpy array
        """
        pass
    
    @abstractmethod
    def solve_ik(
        self, 
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[np.ndarray] = None,
        threshold: float = 0.005,
        ee_link: Optional[str] = None,
        pose_constraint: Optional[ConstraintType] = None,
        constraint_frame: str = "world",
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Inverse kinematics: end-effector pose -> joint angles.
        
        Args:
            target_pose: [x, y, z, qw, qx, qy, qz]
            seed: Initial guess for joint angles
            ee_link: Optional end-effector link name override (defaults to self.ee_link)
            pose_constraint: 
                - Mode A (Simple): [x,y,z, rx,ry,rz] weights (e.g. [1,1,1,0,0,0]).
                - Mode B (Mixed): {"local": [...], "world": [...]} for advanced solvers.
            constraint_frame: Reference frame for the constraint if in Mode A ("world" or "local").
        Returns:
            joint_angles: numpy array or None if no solution
        """
        pass