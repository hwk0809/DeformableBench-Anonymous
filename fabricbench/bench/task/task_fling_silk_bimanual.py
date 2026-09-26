"""Dual-arm grasp + single fling task for silk cloth.

Grasp uses TopologicalGrasper + robust_bimanual_grasp (same as task_grasp_silk_bimanual),
then performs a single fling (same as task_flatten_silk_bimanual stage "fling").
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fabricbench.bench.common.types import ArmSide, ExecutionStep, ObservationMessage
from fabricbench.bench.common.metric_utils import get_covered_area_from_particles
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.base_task import BaseTask
from fabricbench.bench.task.TopologicalGrasper import TopologicalGrasper


@dataclass
class ClothGraspState:
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    left_pos: Optional[np.ndarray] = None
    right_pos: Optional[np.ndarray] = None
    mesh_dist: Optional[float] = None
    select_debug_info: Optional[Dict[str, Any]] = None


class SilkFlingBimanualTask(BaseTask):
    """Dual-arm grasp of silk cloth + single fling."""

    GRASP_Z_OFFSET = -0.001
    GRASP_Z_OFFSET_STEP = -0.005

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        approach_height: float = 0.12,
        grasp_height_limit: float = 0.025,
        grasp_close_val: float = 0.0,
        lift_height: float = 0.4,
        success_thresh: float = 0.15,
        fling_accel_scale: float = 2.0,
        collector=None,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__(env, planner_manager, collector, cfg)
        tp = {}
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})

        self.approach_h = float(tp.get("approach_height", approach_height))
        self.grasp_h_limit = float(tp.get("grasp_height_limit", grasp_height_limit))
        self.grasp_val = float(tp.get("grasp_close_val", grasp_close_val))
        self.lift_h = float(tp.get("lift_height", lift_height))
        self.success_thresh = float(tp.get("success_thresh", success_thresh))
        # cfg 优先，fallback 到构造参数
        if cfg is not None and hasattr(cfg, 'get'):
            self.fling_accel_scale = float(
                tp.get("fling_accel_scale", cfg.get("fling_accel_scale", fling_accel_scale))
            )
        else:
            self.fling_accel_scale = float(fling_accel_scale)

        # TopologicalGrasper — same config as flatten_silk_bimanual
        # Keypoints from the cloth asset_meta (single source), not hardcoded.
        obj_path = self.env.cloth_obj_path
        kp = self.env.cloth_keypoints
        SILK_NODES = {
            "LEFT_UP_CORNER":    {"index": kp["corner_RD"], "core_radius": 0.03, "weight": 1.5},
            "RIGHT_UP_CORNER":   {"index": kp["corner_RU"], "core_radius": 0.03, "weight": 1.5},
            "RIGHT_DOWN_CORNER": {"index": kp["corner_LU"], "core_radius": 0.03, "weight": 1.5},
            "LEFT_DOWN_CORNER":  {"index": kp["corner_LD"], "core_radius": 0.03, "weight": 1.5},
        }
        RECTANGLE_GRAPH = {
            ("LEFT_UP_CORNER", "RIGHT_UP_CORNER"),
            ("LEFT_DOWN_CORNER", "RIGHT_DOWN_CORNER"),
            ("LEFT_UP_CORNER", "LEFT_DOWN_CORNER"),
            ("RIGHT_UP_CORNER", "RIGHT_DOWN_CORNER"),
        }
        self.grasp_selector = TopologicalGrasper(
            SILK_NODES, RECTANGLE_GRAPH, obj_path=obj_path
        )
        self.cloth_grasp_state = ClothGraspState()

    # ------------------------------------------------------------------
    # Grasp target callback (same as task_grasp_silk_bimanual)
    # ------------------------------------------------------------------
    def _make_grasp_pose(self, target_pos: np.ndarray, arm_side: str,
                         z_offset: float = None) -> np.ndarray:
        pos = target_pos.copy()
        if z_offset is None:
            z_offset = self.GRASP_Z_OFFSET
        pos[2] = max(pos[2] + z_offset, 0.000)
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        quat = self.skills.calculate_base_to_target_horizontal_quat(base, pos)
        return np.concatenate([pos, quat])

    def _calculate_grasp_targets(
        self,
        obs: ObservationMessage,
        specific_indices: Optional[Tuple[int, int]] = None,
        z_offset: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        cloth_pos = obs.pcd_cloth_truth

        if specific_indices is None:
            cloth_vis_indices = self.env.get_unoccluded_cloth_indices()
            l_idx, r_idx, mesh_dist, debug_info = (
                self.grasp_selector.selector_optimal_grasp_pair(
                    cloth_pos=cloth_pos,
                    vis_indices=cloth_vis_indices,
                    base_left=self.env.left_base,
                    base_right=self.env.right_base,
                    min_safe_dist=0.15,
                    workspace_radius=0.7,
                    visualize=False,
                )
            )
            self.cloth_grasp_state.mesh_dist = mesh_dist
            self.cloth_grasp_state.select_debug_info = debug_info
        else:
            l_idx, r_idx = specific_indices
            logger.info(
                f"[GraspTargets] Retry with locked indices: L={l_idx}, R={r_idx}"
            )

        target_l = cloth_pos[l_idx].copy()
        target_r = cloth_pos[r_idx].copy()

        pose_l = self._make_grasp_pose(target_l, "left", z_offset=z_offset)
        pose_r = self._make_grasp_pose(target_r, "right", z_offset=z_offset)

        self.cloth_grasp_state.left_idx = l_idx
        self.cloth_grasp_state.right_idx = r_idx
        self.cloth_grasp_state.left_pos = pose_l
        self.cloth_grasp_state.right_pos = pose_r

        logger.info(
            f"[GraspTargets] L={l_idx} {pose_l[:3]}, R={r_idx} {pose_r[:3]}"
        )
        return pose_l, pose_r, l_idx, r_idx

    # ------------------------------------------------------------------
    # Plan episode: grasp → lift → single fling → home
    # ------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        logger.info(f"[SilkFlingBimanual] Starting episode {episode_id}")

        self.init_covered_area = get_covered_area_from_particles(
            np.array(self.env.initial_cloth_states)[0], 0.005
        )

        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        # ---- Stage 1: Robust bimanual grasp ----
        with self.recording("grasp"):
            success = yield from self.skills.robust_bimanual_grasp(
                get_targets_func=self._calculate_grasp_targets,
                max_retries=3,
                close_gripper_value=self.grasp_val,
                grasp_depth=self.GRASP_Z_OFFSET,
                retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                pose_constraint_frame="world",
                test_lift_height=0.15,
                check_threshold_height=0.08,
                ee_link="tcp_link_bottom",
            )

            if not success:
                logger.error("[SilkFlingBimanual] Robust bimanual grasp failed")
                return

            # Lift to prepare for fling
            target_left_lift = [0.2, 0.17, self.lift_h]
            target_right_lift = [0.2, -0.17, self.lift_h]
            plan_lift = self.skills.move_dual_arm_to(
                left_target_pose=target_left_lift,
                right_target_pose=target_right_lift,
                pose_constraint=[1, 1, 1, 0, 0, 0],
            )
            yield ExecutionStep(
                plan=plan_lift, settle_time=0.1, description="lift"
            )

            if not self._check_grasp_success():
                logger.error("[SilkFlingBimanual] Grasp check failed after lift")
                return

            logger.success("[SilkFlingBimanual] Grasp + lift completed")

        # ---- Stage 2: Single fling ----
        with self.recording("fling"):
            fling_pos_left = list(target_left_lift)
            fling_pos_right = list(target_right_lift)

            # Adjust Y separation based on mesh distance between grasped points
            if self.cloth_grasp_state.mesh_dist is not None:
                fling_pos_left[1] = self.cloth_grasp_state.mesh_dist / 2 - 0.01
                fling_pos_right[1] = -self.cloth_grasp_state.mesh_dist / 2 + 0.01
                logger.debug(
                    f"[SilkFlingBimanual] Fling adjusted dist: {self.cloth_grasp_state.mesh_dist}, "
                    f"left_y: {fling_pos_left[1]}, right_y: {fling_pos_right[1]}"
                )

            forward_q = np.array([0.707106781, 0.0, 0.707106781, 0.0])
            target_left_fling = np.concatenate([fling_pos_left, forward_q])
            target_right_fling = np.concatenate([fling_pos_right, forward_q])
            yield from self.skills.fling(target_left_fling, target_right_fling,
                                        fling_accel_scale=self.fling_accel_scale)

            plan_home = self.skills.move_to_home()
            yield ExecutionStep(
                plan=plan_home, settle_time=0.0, description="go_home"
            )

            now_covered_area = get_covered_area_from_particles(
                np.array(self.env.get_cloth_position()), 0.005
            )
            cov = now_covered_area / self.init_covered_area
            logger.info(f"[SilkFlingBimanual] Fling coverage: {cov:.4f}")

    # ------------------------------------------------------------------
    # Success checks
    # ------------------------------------------------------------------
    def _check_grasp_success(self) -> bool:
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None:
            return False
        z_mean = float(np.mean(cloth_pos[:, 2]))
        ok = z_mean > self.success_thresh
        logger.debug(
            f"[GraspCheck] Cloth mean height: {z_mean:.4f}m, "
            f"thresh: {self.success_thresh}m, success: {ok}"
        )
        return ok

    def check_success(self) -> bool:
        curr_pos = self.env.get_cloth_position()
        if curr_pos is None:
            return False
        now_covered_area = get_covered_area_from_particles(curr_pos, 0.005)
        cov = now_covered_area / self.init_covered_area
        logger.info(f"[SilkFlingBimanual] Final coverage: {cov:.4f}")
        return cov > 0.5
