"""Sim-layer I/O contract types.

These are produced BY the simulator (``DeformableEnv.get_observation``), so they
live in the sim layer; ``bench.common.types`` re-exports ``ObservationMessage`` so
existing bench imports keep working without sim ever depending on bench.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ObservationMessage:
    """Unified observation for deformable-object manipulation."""

    # 1. Proprioception
    joint_left: np.ndarray
    joint_right: np.ndarray
    eef_pose_left: np.ndarray       # [x,y,z, qw,qx,qy,qz]
    eef_pose_right: np.ndarray
    gripper_left: float
    gripper_right: float

    # 2. Visual (filled by the render bridge; None when visuals are off)
    rgb_image: Optional[np.ndarray] | Dict[str, np.ndarray] = None
    depth_image: Optional[np.ndarray] | Dict[str, np.ndarray] = None
    camera_pose: Optional[np.ndarray] | Dict[str, np.ndarray] = None

    # 3. Point clouds
    pcd_scene: Optional[np.ndarray] | Dict[str, np.ndarray] = None
    pcd_cloth: Optional[np.ndarray] | Dict[str, np.ndarray] = None
    pcd_cloth_truth: Optional[np.ndarray] | Dict[str, np.ndarray] = None

    # 4. Keypoints
    cloth_keypoints: Optional[np.ndarray] = None

    # 5. Misc
    timestamp: float = 0.0
