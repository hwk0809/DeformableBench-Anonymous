import os
import sys
import numpy as np
from typing import Optional, Any, Generator, Tuple, Dict, List
from omegaconf import DictConfig
from loguru import logger
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fabricbench.bench.task.base_task import BaseTask
from fabricbench.bench.common.types import ArmSide, ExecutionStep, DualPlanResult, ObservationMessage
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from dataclasses import dataclass

@dataclass
class ClothGraspState:
    """跟踪双臂抓取状态（用于重试机制）"""
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    left_pos: Optional[np.ndarray] = None
    right_pos: Optional[np.ndarray] = None
    left_grasped: bool = False
    right_grasped: bool = False

    def reset_grasp_flags(self):
        self.left_grasped = False
        self.right_grasped = False


class BaseFoldTask(BaseTask):
    """
    Base class for folding tasks.
    Contains common logic for folding operations, such as defining keypoints, fold lines, and execution steps.
    Specific fold tasks (e.g., T-shirt fold, towel fold) will inherit from this class and implement task-specific details.
    """

    # 高度常量
    SAFE_HEIGHT = 0.15        # waypoint安全高度
    TABLE_HEIGHT = 0.05       # 桌面/布料表面z
    GRASP_Z_OFFSET = -0.01   # 抓取时夹爪略低于布料
    GRASP_Z_OFFSET_STEP = -0.005  # 每次重试额外下压量
    GRASP_XY_OFFSET_STEP = 0   # 每次重试的XY随机偏移量
    OPEN_GRIPPER = 0.7
    OPEN_SMALL_GRIPPER = 0.2
    CLOSE_GRIPPER = 0.0

    # Stage 前折参数 (Boss 可调)
    FORWARD_FOLD_LIFT_HEIGHT = 0.1   # 前折时先抬多高再观测
    FOLD_DEST_Z_OFFSET = 0.05        # 折叠目的地比桌面高多少

    def __init__(
        self,
        env,
        keypoints: Optional[Dict[str, Dict[str, Any]]] = None,
        planner_manager: Optional[PlannerManager] = None,
        collector=None,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__(env, planner_manager, collector, cfg)

        tp: Dict[str, Any] = {}
        if cfg is not None:
            tp = dict(cfg.get("task_params", {}))

        self.safe_height = float(tp.get("safe_height", self.SAFE_HEIGHT if hasattr(self, "SAFE_HEIGHT") else 0.22))
        
        self.grasp_settle_time = float(tp.get("grasp_settle_time", 0.6))
        self.test_lift_height = float(tp.get("test_lift_height", 0.06))

        self.grasp_val = float(tp.get("grasp_close_val", 0.0))
        self.OPEN_SMALL_GRIPPER = float(tp.get("open_small_gripper", self.OPEN_SMALL_GRIPPER))
        self.fold_lift_height = float(tp.get("fold_lift_height", 0.18))
        self.place_z = float(tp.get("place_z", 0.02))
        self.place_hover_z = float(tp.get("place_hover_z", 0.12))
        self.place_hover_z_max = float(tp.get("place_hover_z_max", 0.2))
        self.retract_y_offset = float(tp.get("retract_y_offset", 0.02))
        self.lift_dist_gain = float(tp.get("lift_dist_gain", 0.45))
        self.lift_base_margin = float(tp.get("lift_base_margin", 0.06))
        self.center_bias = float(tp.get("center_bias", 1.0))
        self.arc_midpoint_lift = float(tp.get("arc_midpoint_lift", 0.02))
        self.success_threshold = float(tp.get("success_threshold", 0.15))

        self.task_success = False

        self.cloth_grasp_state = ClothGraspState()
        if keypoints is None:
            keypoints = self._load_keypoints_from_asset(env)

        self.keypoints = keypoints.copy()

        grasp_cfg = {}
        if cfg is not None:
            grasp_cfg = cfg.get("grasp", {}) or {}

        self.slow_approach = bool(grasp_cfg.get("slow_approach", False))
        via_dist_range = grasp_cfg.get("slow_approach_via_distance_range", [0.05, 0.10])
        self.via_distance_range = [float(via_dist_range[0]), float(via_dist_range[1])]
        self.slow_approach_final_vel_scale = float(
            grasp_cfg.get("slow_approach_final_vel_scale", 0.08)
        )
        if self.slow_approach:
            logger.info(
                f"[TshirtFoldV2] slow_approach ENABLED, "
                f"via_distance_range={self.via_distance_range}m, "
                f"final_vel_scale={self.slow_approach_final_vel_scale}"
            )

        self.grasp_ground_check = bool(grasp_cfg.get("ground_check", True))

        fold_cfg = {}
        if cfg is not None:
            fold_cfg = cfg.get("fold", {}) or {}
        # < 1.0 biases the fold crease toward the hem, avoiding cloth drag.
        self.forward_fold_ratio = float(fold_cfg.get("forward_fold_ratio", 0.9))

        self.yaw_jitter_enabled = bool(grasp_cfg.get("yaw_jitter", False))
        yaw_range_deg = grasp_cfg.get("yaw_jitter_range_deg", [-3.0, 3.0])
        self.yaw_jitter_range_rad = [
            float(np.deg2rad(yaw_range_deg[0])),
            float(np.deg2rad(yaw_range_deg[1])),
        ]
        self.yaw_jitter_L = 0.0
        self.yaw_jitter_R = 0.0
        self._via_distance = 0.0
        if self.yaw_jitter_enabled:
            logger.info(
                f"[TshirtFoldV2] yaw_jitter ENABLED, "
                f"range_deg={yaw_range_deg} (L/R independent per episode)"
            )


    # ========================================
    # Helper Methods
    # ========================================
    def _get_keypoint_pos(self, cloth_pos: np.ndarray, name: str) -> np.ndarray:
        """根据关键点名称查找顶点位置"""
        idx = self.keypoints[name]['index']
        return cloth_pos[idx].copy()

    def _make_grasp_pose(self, target_pos: np.ndarray, arm_side: str, z_offset: float = None) -> np.ndarray:
        """构建7D抓取位姿 [x,y,z,qw,qx,qy,qz]"""
        pos = target_pos.copy()
        if z_offset is None:
            z_offset = self.GRASP_Z_OFFSET
        pos[2] = max(pos[2] + z_offset, 0.000)
        
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        quat = self.skills.calculate_base_to_target_horizontal_quat(base, pos)
        logger.debug(f"Calculated grasp position for {arm_side} arm: {pos},quat={quat}")
        return np.concatenate([pos, quat])

    def _make_keypoint_grasp_targets(
        self, obs: ObservationMessage,
        left_key: str, right_key: str,
        specific_indices: Optional[Tuple[int, int]] = None,
        z_offset: float = None
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """ 
        基于关键点的抓取目标计算（供 robust_bimanual_grasp 回调使用）

        Args:
            obs: 当前观测
            left_key: 左臂抓取的关键点名
            right_key: 右臂抓取的关键点名
            specific_indices: 重试时锁定的顶点索引
            z_offset: 抓取Z轴偏移，None则使用默认GRASP_Z_OFFSET
        Returns:
            (left_pose_7d, right_pose_7d, left_idx, right_idx)
        """

        pose_l, l_idx = self._make_single_keypoint_grasp_targets(
            obs, left_key, "left", 
            specific_index=specific_indices[0] if specific_indices else None,
            z_offset=z_offset
        )

        pose_r, r_idx = self._make_single_keypoint_grasp_targets(
            obs, right_key, "right", 
            specific_index=specific_indices[1] if specific_indices else None,
            z_offset=z_offset
        )

        logger.info(f"[Target Select] L_idx={l_idx} ({left_key}), R_idx={r_idx} ({right_key}), "
                    f"L_pos={pose_l[:3]}, R_pos={pose_r[:3]}")

        return pose_l, pose_r, l_idx, r_idx
    
    def _make_single_keypoint_grasp_targets(self, obs: ObservationMessage, 
                                            key: str, 
                                            arm_side: str, 
                                            specific_index: Optional[int] = None,
                                            z_offset: float = None) -> np.ndarray:
        """单臂基于关键点的抓取目标计算（供单臂抓取回调使用）"""
        cloth_pos = obs.pcd_cloth_truth
        if specific_index is not None:
            idx = specific_index
        else:
            idx = self.keypoints[key]['index']
        target_pos = cloth_pos[idx].copy()
        pose_7d = self._make_grasp_pose(target_pos, arm_side, z_offset=z_offset)

        return pose_7d, idx

        # Gripper orientation stays locked to this yaw across grasp/lift/place in the

    # same episode — once the fingers close on the cloth we can't twist the wrist.
    def _yaw_jitter_for(self, arm_side: str) -> float:
        return self.yaw_jitter_L if arm_side == "left" else self.yaw_jitter_R

    def _make_grasp_pose(
        self, target_pos: np.ndarray, arm_side: str, z_offset: Optional[float] = None
    ) -> np.ndarray:
        pos = target_pos.copy()
        if z_offset is None:
            z_offset = self.GRASP_Z_OFFSET
        pos[2] = max(pos[2] + z_offset, 0.000)
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        quat = self.skills.calculate_base_to_target_horizontal_quat(
            base, pos, yaw_jitter=self._yaw_jitter_for(arm_side)
        )
        return np.concatenate([pos, quat])

    def _make_place_pose(
        self, target_pos: np.ndarray, arm_side: str, z: float
    ) -> np.ndarray:
        pos = target_pos.copy()
        pos[2] = z
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        quat = self.skills.calculate_base_to_target_horizontal_quat(
            base, pos, yaw_jitter=self._yaw_jitter_for(arm_side)
        )
        return np.concatenate([pos, quat])

    @staticmethod
    def _project_point_to_line_xy(
        point_xy: np.ndarray,
        line_start_xy: np.ndarray,
        line_end_xy: np.ndarray,
    ) -> np.ndarray:
        line_vec = line_end_xy - line_start_xy
        denom = float(np.dot(line_vec, line_vec))
        if denom < 1e-8:
            return line_start_xy.copy()
        t = float(np.dot(point_xy - line_start_xy, line_vec) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        return line_start_xy + t * line_vec

    def _compute_fold_geometry(
        self,
        cloth_pos: np.ndarray,
        src_top_key: str,
        src_hem_key: str,
        dst_top_key: str,
        dst_hem_key: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        src_top = self._get_keypoint_pos(cloth_pos, src_top_key)
        src_hem = self._get_keypoint_pos(cloth_pos, src_hem_key)
        dst_top = self._get_keypoint_pos(cloth_pos, dst_top_key)
        dst_hem = self._get_keypoint_pos(cloth_pos, dst_hem_key)
        left_collar = self._get_keypoint_pos(cloth_pos, "left_collar")
        right_collar = self._get_keypoint_pos(cloth_pos, "right_collar")

        top_mid = 0.5 * (left_collar + right_collar)
        bottom_mid = 0.5 * (src_hem + dst_hem)

        top_xy = self._project_point_to_line_xy(
            src_top[:2], top_mid[:2], bottom_mid[:2]
        )
        hem_xy = self._project_point_to_line_xy(
            src_hem[:2], top_mid[:2], bottom_mid[:2]
        )

        target_top = src_top.copy()
        target_hem = src_hem.copy()
        target_top[:2] = (1.0 - self.center_bias) * src_top[
            :2
        ] + self.center_bias * top_xy
        target_hem[:2] = (1.0 - self.center_bias) * src_hem[
            :2
        ] + self.center_bias * hem_xy

        top_move_xy = float(np.linalg.norm(target_top[:2] - src_top[:2]))
        hem_move_xy = float(np.linalg.norm(target_hem[:2] - src_hem[:2]))
        logger.info(
            f"[TshirtFoldV1] midline fold: top_move={top_move_xy:.3f}m, "
            f"hem_move={hem_move_xy:.3f}m, top_mid={top_mid[:2]}, bottom_mid={bottom_mid[:2]}"
        )
        return target_top, target_hem

    def _compute_adaptive_lift_z(
        self,
        left_start_xy: np.ndarray,
        right_start_xy: np.ndarray,
        left_target_xy: np.ndarray,
        right_target_xy: np.ndarray,
    ) -> Tuple[float, float]:
        """根据双臂的水平移动距离，自适应计算悬停高度和抬升高度。

        折叠动作中手臂需要先抬起衣物、平移后再放下。移动距离越长，
        中途悬停高度应越高，以避免衣物拖拽或碰撞桌面。

        Returns:
            adaptive_hover_z: 平移阶段的悬停高度（夹爪底部 Z 坐标）
            adaptive_lift_z:  抓取后初始抬升高度，略高于悬停高度以留出余量
        """
        # 计算左右臂各自在 XY 平面内的移动距离（欧氏距离）
        left_move_xy = float(np.linalg.norm(left_target_xy - left_start_xy))
        right_move_xy = float(np.linalg.norm(right_target_xy - right_start_xy))
        # 取双臂中移动距离较大的一侧，作为高度自适应的基准
        max_move_xy = max(left_move_xy, right_move_xy)

        # 悬停高度 = 放置高度 + 固定裕量 + 距离增益 * 最大水平位移
        # 同时保证不低于配置的最小悬停高度
        adaptive_hover_z = max(
            self.place_hover_z,
            self.place_z + self.lift_base_margin + self.lift_dist_gain * max_move_xy,
        )
        # 限制上限，防止抬升过高造成不稳定
        adaptive_hover_z = min(adaptive_hover_z, self.place_hover_z_max)
        # 抬升高度在悬停高度基础上再加 3 cm，确保平移前衣物已完全离开桌面
        adaptive_lift_z = max(self.fold_lift_height, adaptive_hover_z + 0.03)
        logger.info(
            f"[TshirtFoldV1] transfer_move_xy: left={left_move_xy:.3f}m, "
            f"right={right_move_xy:.3f}m, hover_z={adaptive_hover_z:.3f}, "
            f"lift_z={adaptive_lift_z:.3f}"
        )
        return adaptive_hover_z, adaptive_lift_z
    
    @staticmethod
    def _load_keypoints_from_asset(env) -> Dict[str, Dict[str, int]]:
        cloth_obj_path = getattr(env, "cloth_obj_path", None)
        if not cloth_obj_path:
            raise ValueError("env.cloth_obj_path is not available")

        cfg_path = os.path.join(os.path.dirname(cloth_obj_path), "asset_meta.yaml")
        with open(cfg_path, "r") as f:
            cloth_cfg = yaml.safe_load(f) or {}

        keypoints = cloth_cfg.get("keypoints", None)
        if not keypoints:
            raise ValueError(f"No keypoints found in {cfg_path}")
        logger.info(
            f"[TshirtFoldV1] keypoints loaded from {cfg_path} "
            f"(obj={cloth_cfg.get('obj')}, name={cloth_cfg.get('name')})"
        )
        return keypoints
    
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        """
        Abstract method to be implemented by specific fold tasks.
        Should return a list of stages with their corresponding plans and metadata.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    # ========================================
    # Success Check
    # ========================================
    def check_success(self) -> bool:
        cloth_pos = self.env.get_cloth_position()
        try:
            l_shoulder = self._get_keypoint_pos(cloth_pos, "L_Shoulder")
            r_shoulder = self._get_keypoint_pos(cloth_pos, "R_Shoulder")
            l_hem = self._get_keypoint_pos(cloth_pos, "L_Hem")
            r_hem = self._get_keypoint_pos(cloth_pos, "R_Hem")
        except Exception:
            l_shoulder = self._get_keypoint_pos(cloth_pos, "left_sleeve_upper")
            r_shoulder = self._get_keypoint_pos(cloth_pos, "right_sleeve_upper")
            l_hem = self._get_keypoint_pos(cloth_pos, "left_hem")
            r_hem = self._get_keypoint_pos(cloth_pos, "right_hem")
        
        shoulder_center = (l_shoulder + r_shoulder) / 2
        hem_center = (l_hem + r_hem) / 2
        dist = np.linalg.norm(
            shoulder_center - hem_center
        )

        success = dist < self.success_threshold

        logger.info(
            f"[check_success] dist={dist:.4f}, threshold={self.success_threshold}, success={success}"
        )

        return success
