from typing import Tuple,Union,List,Optional
from loguru import logger
import numpy as np

@staticmethod
def validate_goal_format(
    goal: Union[np.ndarray, List[float], List[np.ndarray]],
    goal_type: str,
    expected_joint_dof: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate goal format based on goal_type.
    
    Args:
        goal: Input goal (joints, pose, or waypoints)
        goal_type: "joint", "pose", or "waypoints"
        expected_joint_dof: Expected number of joints (for joint space validation)
        
    Returns:
        (is_valid, error_message)
    """
    goal_arr = np.array(goal) if not isinstance(goal, np.ndarray) else goal
    
    if goal_type == "joint":
        # Joint space goal should be 1D array
        if goal_arr.ndim != 1:
            return False, f"Joint goal must be 1D, got shape {goal_arr.shape}"
        
        if expected_joint_dof is not None and len(goal_arr) != expected_joint_dof:
            return False, f"Joint goal dimension mismatch. Expected {expected_joint_dof}, got {len(goal_arr)}"
        
        return True, ""
    
    elif goal_type == "pose":
        # Pose should be [x, y, z, qw, qx, qy, qz] (7 elements)
        if len(goal_arr) != 7:
            return False, f"Pose must be [x,y,z,qw,qx,qy,qz] (7 elements), got {len(goal_arr)}"
        
        # Check quaternion normalization
        quat = goal_arr[3:]
        quat_norm = np.linalg.norm(quat)
        if not (0.95 < quat_norm < 1.05):
            return False, f"Quaternion not normalized (norm={quat_norm:.3f}). Expected ~1.0"
        
        return True, ""
    
    elif goal_type == "waypoints":
        # Waypoints should be list of poses
        if not isinstance(goal, list):
            return False, "Waypoints must be a list of poses"
        
        if len(goal) == 0:
            return False, "Waypoints list is empty"
        
        for i, wp in enumerate(goal):
            wp_arr = np.array(wp)
            if len(wp_arr) != 7:
                return False, f"Waypoint {i} must be [x,y,z,qw,qx,qy,qz], got {len(wp_arr)}"
            
            quat = wp_arr[3:]
            quat_norm = np.linalg.norm(quat)
            if not (0.95 < quat_norm < 1.05):
                return False, f"Waypoint {i} quaternion not normalized (norm={quat_norm:.3f})"
        
        return True, ""
    
    else:
        return False, f"Unknown goal_type: {goal_type}"

def normalize_quaternion(pose: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion in pose [x, y, z, qw, qx, qy, qz].
    
    Args:
        pose: Input pose (7 elements)
        
    Returns:
        Normalized pose
    """
    pose = np.array(pose)
    quat = pose[3:]
    quat_norm = np.linalg.norm(quat)
    
    if quat_norm < 1e-6:
        # Degenerate quaternion, set to identity
        logger.warning("Degenerate quaternion detected, setting to identity [1,0,0,0]")
        pose[3:] = [1.0, 0.0, 0.0, 0.0]
    else:
        pose[3:] = quat / quat_norm
    
    return pose