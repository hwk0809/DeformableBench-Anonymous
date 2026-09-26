import os
import sys
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import yaml
from loguru import logger
from omegaconf import DictConfig

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fabricbench.bench.common.types import ExecutionStep, ArmSide
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.task_base_fold import BaseFoldTask


class TshirtFoldV2Task(BaseFoldTask):
    """Single-stage side fold for the blue T-shirt.

    Stage 1 only:
      grasp one side's sleeve-center + hem, then fold them toward the garment midline.
    """

    # Task variants can opt into a simpler drag-edge policy: one unchecked grasp,
    # one drag, then exactly one cloth-presence check immediately before release.
    # Keep the legacy closed-loop grasp/regrasp path as the default for existing
    # task configs that use this base class directly.
    DRAG_EDGE_PRE_RELEASE_ABORT = False

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        weld_grasp: bool = False,
        collector=None,
        cfg: Optional[DictConfig] = None,
    ):

        super().__init__(
            env=env,
            keypoints=None,
            planner_manager=planner_manager,
            collector=collector,
            cfg=cfg,
        )

        self.weld_grasp = weld_grasp
        self.stage_physics_cfg = {}
        task_params = {}
        if cfg is not None:
            self.stage_physics_cfg = dict(cfg.get("stage_physics", {}) or {})
            task_params = dict(cfg.get("task_params", {}) or {})
        self.side_fold_hem_inward_offset = float(
            task_params.get("side_fold_hem_inward_offset", 0.01)
        )
        self.drag_edge_grasp_x_offset = float(
            task_params.get("drag_edge_grasp_x_offset", 0.0)
        )
        self.forward_fold_lift_height = float(
            task_params.get("forward_fold_lift_height", self.fold_lift_height)
        )
        self.forward_fold_place_z = float(
            task_params.get("forward_fold_place_z", self.place_z)
        )

        logger.info(
            f"[TshirtFoldV2] fixed fold side=left, "
            f"safe_height={self.safe_height}, fold_lift_height={self.fold_lift_height}, "
            f"forward_fold_lift_height={self.forward_fold_lift_height}, "
            f"forward_fold_place_z={self.forward_fold_place_z}, "
            f"center_bias={self.center_bias}"
        )

    def _apply_stage_physics(self, stage_name: str):
        cfg = self.stage_physics_cfg.get(stage_name, {}) or {}
        if not cfg or not bool(cfg.get("enabled", True)):
            return None
        if not hasattr(self.env, "set_runtime_physics"):
            logger.warning(f"[TshirtFoldV2] runtime physics not supported for stage {stage_name}")
            return None

        allowed = (
            "soft_contact_mu",
            "soft_contact_margin",
            "soft_contact_ke",
            "soft_contact_kd",
            "default_shape_mu",
            "gripper_shape_mu",
        )
        updates = {key: cfg[key] for key in allowed if key in cfg}
        if not updates:
            return None
        snapshot = self.env.snapshot_runtime_physics()
        self.env.set_runtime_physics(**updates)
        logger.info(f"[TshirtFoldV2] stage_physics {stage_name}: {updates}")
        return snapshot

    def _restore_stage_physics(self, snapshot, stage_name: str) -> None:
        if snapshot is None:
            return
        self.env.restore_runtime_physics(snapshot)
        logger.info(f"[TshirtFoldV2] stage_physics {stage_name}: restored")

    def _release_bimanual(self, description: str) -> Generator[ExecutionStep, None, None]:
        if self.weld_grasp:
            self.skills.weld_release_bimanual(self.env)
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(
                self.OPEN_SMALL_GRIPPER, self.OPEN_SMALL_GRIPPER, duration=0.3
            ),
            settle_time=0.3,
            description=description,
        )

    def _release_single(self, arm: ArmSide, description: str) -> Generator[ExecutionStep, None, None]:
        if self.weld_grasp:
            self.skills.weld_release_single(self.env, arm)
        yield ExecutionStep(
            plan=self.skills.gripper_action(
                self.OPEN_SMALL_GRIPPER, arm, duration=0.3
            ),
            settle_time=0.3,
            description=description,
        )

    def _side_fold_hold_status(self, label: str) -> Tuple[bool, bool]:
        obs = self.reset_interface_state()
        l_ok, r_ok = self.skills.check_bimanual_grasp(
            obs=obs,
            threshold_dist=0.045,
            threshold_height=max(self.place_z + 0.02, 0.04),
        )
        ok = bool(l_ok and r_ok)
        logger.info(f"[TshirtFoldV2] {label} hold_check: L={l_ok}, R={r_ok}, ok={ok}")
        return bool(l_ok), bool(r_ok)

    def _shift_pose_toward_garment_center(
        self,
        pose: np.ndarray,
        cloth_pos: np.ndarray,
    ) -> np.ndarray:
        """Move a hem grasp directly toward the garment center keypoint."""
        garment_center = self._get_keypoint_pos(cloth_pos, "center")
        inward = garment_center[:2] - pose[:2]
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm < 1e-8 or self.side_fold_hem_inward_offset <= 0.0:
            return pose

        shifted = pose.copy()
        shifted[:2] += self.side_fold_hem_inward_offset * inward / inward_norm
        return shifted

    def _make_zero_z_grasp_pose(self, target_pos: np.ndarray, arm_side: str) -> np.ndarray:
        """Build a grasp pose with the TCP bottom fixed at the world Z=0 plane."""
        pose = self._make_grasp_pose(target_pos, arm_side, z_offset=0.0)
        pose[2] = 0.0
        return pose

    def _repair_side_fold_grasp(
        self,
        side: ArmSide,
        l_ok: bool,
        r_ok: bool,
    ) -> Generator[ExecutionStep, None, bool]:
        if side == ArmSide.LEFT:
            left_key, right_key = "left_sleeve_center", "left_hem"
        else:
            left_key, right_key = "right_sleeve_center", "right_hem"

        def _single_target(key: str, arm_name: str):
            def target(obs, locked_index=None, z_offset=None):
                pose, idx = self._make_single_keypoint_grasp_targets(
                    obs,
                    key,
                    arm_name,
                    specific_index=locked_index,
                    z_offset=z_offset,
                )
                if arm_name == "right":
                    pose = self._shift_pose_toward_garment_center(
                        pose, obs.pcd_cloth_truth
                    )
                return pose, idx
            return target

        repaired = True
        if not l_ok:
            logger.warning(f"[TshirtFoldV2] repairing LEFT grasp on {left_key}; right gripper stays closed")
            ok = yield from self.skills.robust_single_grasp(
                arm=ArmSide.LEFT,
                get_target_func=_single_target(left_key, "left"),
                other_arm_mode="hold_grasp",
                max_retries=2,
                close_gripper_value=self.grasp_val,
                grasp_depth=self.GRASP_Z_OFFSET,
                retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                retry_inward_step=self.GRASP_XY_OFFSET_STEP,
                grasp_settle_time=self.grasp_settle_time,
                test_lift_height=self.test_lift_height,
                pose_constraint=[1, 1, 1, 1, 0, 1],
                pose_constraint_frame="local",
                ee_link="tcp_link_bottom",
            )
            repaired = repaired and bool(ok)
        if not r_ok:
            logger.warning(f"[TshirtFoldV2] repairing RIGHT grasp on {right_key}; left gripper stays closed")
            ok = yield from self.skills.robust_single_grasp(
                arm=ArmSide.RIGHT,
                get_target_func=_single_target(right_key, "right"),
                other_arm_mode="hold_grasp",
                max_retries=2,
                close_gripper_value=self.grasp_val,
                grasp_depth=self.GRASP_Z_OFFSET,
                retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                retry_inward_step=self.GRASP_XY_OFFSET_STEP,
                grasp_settle_time=self.grasp_settle_time,
                test_lift_height=self.test_lift_height,
                pose_constraint=[1, 1, 1, 1, 0, 1],
                pose_constraint_frame="local",
                ee_link="tcp_link_bottom",
            )
            repaired = repaired and bool(ok)
        return repaired

    def _select_drag_edge_indices(
        self,
        cp: np.ndarray,
        lbase_xy: np.ndarray,
        rbase_xy: np.ndarray,
    ) -> Tuple[int, int]:
        """Select the two cloth particles used by the edge-drag grasp.

        This is the legacy selection policy.  It lives in a method (rather than
        inside ``_drag_fold_edge_inward``) so task variants can change one arm's
        selection without copying the full grasp/drag/recovery routine.
        """
        y_med = float(np.median(cp[:, 1]))
        l_idxs = np.where(cp[:, 1] >= y_med)[0]
        r_idxs = np.where(cp[:, 1] < y_med)[0]
        l_order = np.argsort(np.linalg.norm(cp[l_idxs, :2] - lbase_xy, axis=1))
        r_order = np.argsort(np.linalg.norm(cp[r_idxs, :2] - rbase_xy, axis=1))
        # Use 2nd-closest: skips the absolute tip where fabric is thinnest.
        l_idx = int(l_idxs[l_order[min(1, len(l_order) - 1)]])
        r_idx = int(r_idxs[r_order[min(1, len(r_order) - 1)]])
        return l_idx, r_idx

    def _drag_fold_edge_inward(
        self,
        drag_dist: float = 0.15,
    ) -> Generator[ExecutionStep, None, bool]:
        """Grasp the two cloth edge points closest to each arm, then drag toward the arms.

        Point selection is delegated to ``_select_drag_edge_indices``.  The base
        implementation splits cloth at the Y median so the left arm draws from
        the Y+ half and the right arm from the Y− half; task variants may
        override only that selector while reusing this grasp/drag routine.
        The legacy policy uses a robust grasp; variants may instead opt into one
        unchecked grasp followed by a single pre-release failure check.  The arms
        descend to place_z and drag each point by drag_dist toward their bases.
        """
        lbase_xy = np.array(self.env.left_base[:2], dtype=float)
        rbase_xy = np.array(self.env.right_base[:2], dtype=float)
        hold_check_dist = 0.02
        supplemental_drag_dist = float(np.clip(0.4 * drag_dist, 0.04, 0.07))

        def _make_drag_grasp_pose(point: np.ndarray, arm_name: str) -> np.ndarray:
            target = point.copy()
            target[0] += self.drag_edge_grasp_x_offset
            return self._make_zero_z_grasp_pose(target, arm_name)

        def edge_targets(obs, specific_indices=None, z_offset=None):
            cp = obs.pcd_cloth_truth
            if specific_indices is not None:
                l_idx, r_idx = specific_indices
            else:
                l_idx, r_idx = _select_edge_indices(cp)
            # The drag-edge pinch is deliberately fixed at the table plane.
            # Ignore robust-grasp retry depth offsets for this action.
            pose_l = _make_drag_grasp_pose(cp[l_idx], "left")
            pose_r = _make_drag_grasp_pose(cp[r_idx], "right")
            return pose_l, pose_r, l_idx, r_idx

        def _select_edge_indices(cp: np.ndarray) -> Tuple[int, int]:
            l_idx, r_idx = self._select_drag_edge_indices(cp, lbase_xy, rbase_xy)
            logger.info(
                f"[TshirtFoldV2] drag_edge selected: "
                f"left_idx={l_idx}, right_idx={r_idx}, "
                f"raw_left={cp[l_idx, :2]}, raw_right={cp[r_idx, :2]}, "
                f"grasp_x_offset={self.drag_edge_grasp_x_offset:+.3f}m"
            )
            return l_idx, r_idx

        def _single_edge_target(arm: ArmSide):
            def target(obs, locked_index=None, z_offset=None):
                cp = obs.pcd_cloth_truth
                if locked_index is not None:
                    idx = int(locked_index)
                else:
                    l_idx, r_idx = _select_edge_indices(cp)
                    idx = l_idx if arm == ArmSide.LEFT else r_idx
                arm_name = "left" if arm == ArmSide.LEFT else "right"
                pose = _make_drag_grasp_pose(cp[idx], arm_name)
                return pose, idx
            return target

        def _toward_xy(ee_xy: np.ndarray, base_xy: np.ndarray, distance: float) -> np.ndarray:
            v = base_xy - ee_xy
            n = float(np.linalg.norm(v))
            return ee_xy + distance * (v / n if n > 1e-4 else np.zeros(2))

        def _ee_near_cloth_hold(label: str) -> Tuple[bool, bool]:
            obs = self.reset_interface_state()
            cp = obs.pcd_cloth_truth
            if cp is None or len(cp) == 0:
                logger.warning(f"[TshirtFoldV2] drag_edge/{label} hold_check: no cloth points")
                return False, False
            l_dist = float(np.min(np.linalg.norm(cp - obs.eef_pose_left[:3], axis=1)))
            r_dist = float(np.min(np.linalg.norm(cp - obs.eef_pose_right[:3], axis=1)))
            l_ok = l_dist < hold_check_dist
            r_ok = r_dist < hold_check_dist
            ok = bool(l_ok and r_ok)
            logger.info(
                f"[TshirtFoldV2] drag_edge/{label} ee_dist_hold_check: "
                f"L={l_dist:.4f}({l_ok}), R={r_dist:.4f}({r_ok}), ok={ok}, "
                f"threshold={hold_check_dist:.3f}"
            )
            return bool(l_ok), bool(r_ok)

        def _grasp_edge(label: str) -> Generator[ExecutionStep, None, bool]:
            if self.weld_grasp:
                ok = yield from self.skills.weld_bimanual_grasp(
                    env=self.env,
                    get_targets_func=edge_targets,
                    velocity_scale=0.3,
                    ee_link="tcp_link_bottom",
                )
            else:
                ok = yield from self.skills.robust_bimanual_grasp(
                    get_targets_func=edge_targets,
                    max_retries=3,
                    close_gripper_value=self.grasp_val,
                    grasp_depth=self.GRASP_Z_OFFSET,
                    retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                    grasp_settle_time=self.grasp_settle_time,
                    test_lift_height=self.test_lift_height,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link="tcp_link_bottom",
                    velocity_scale=0.3,
                )
            logger.info(f"[TshirtFoldV2] drag_edge {label} grasp -> {ok}")
            return bool(ok)

        def _grasp_edge_without_check() -> Generator[ExecutionStep, None, None]:
            """Approach and close once, without test lift, verification, or retry."""
            if self.weld_grasp:
                ok = yield from self.skills.weld_bimanual_grasp(
                    env=self.env,
                    get_targets_func=edge_targets,
                    velocity_scale=0.3,
                    ee_link="tcp_link_bottom",
                )
                if not ok:
                    raise RuntimeError("Failed to plan unchecked weld grasp for edge drag")
                return

            obs = self.reset_interface_state()
            pose_l, pose_r, _, _ = edge_targets(obs)
            plan_approach = self.skills.move_dual_arm_to(
                left_target_pose=pose_l,
                right_target_pose=pose_r,
                left_gripper=self.OPEN_GRIPPER,
                right_gripper=self.OPEN_GRIPPER,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link="tcp_link_bottom",
                velocity_scale=0.3,
            )
            if not getattr(plan_approach, "success", False):
                raise RuntimeError(
                    f"Failed to plan unchecked grasp approach for edge drag: {plan_approach}"
                )
            yield ExecutionStep(
                plan=plan_approach,
                settle_time=0.2,
                description="drag_edge_approach",
            )
            yield ExecutionStep(
                plan=self.skills.gripper_bimanual_action(
                    self.grasp_val,
                    self.grasp_val,
                    duration=self.grasp_settle_time,
                ),
                settle_time=0.3,
                description="drag_edge_grasp_close",
            )

        def _repair_edge_grasp(l_ok: bool, r_ok: bool) -> Generator[ExecutionStep, None, bool]:
            repaired = True
            if not l_ok:
                logger.warning("[TshirtFoldV2] drag_edge repairing LEFT only; right gripper stays closed")
                ok = yield from self.skills.robust_single_grasp(
                    arm=ArmSide.LEFT,
                    get_target_func=_single_edge_target(ArmSide.LEFT),
                    other_arm_mode="hold_grasp",
                    max_retries=2,
                    close_gripper_value=self.grasp_val,
                    grasp_depth=self.GRASP_Z_OFFSET,
                    retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                    grasp_settle_time=self.grasp_settle_time,
                    test_lift_height=self.test_lift_height,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link="tcp_link_bottom",
                )
                repaired = repaired and bool(ok)
            if not r_ok:
                logger.warning("[TshirtFoldV2] drag_edge repairing RIGHT only; left gripper stays closed")
                ok = yield from self.skills.robust_single_grasp(
                    arm=ArmSide.RIGHT,
                    get_target_func=_single_edge_target(ArmSide.RIGHT),
                    other_arm_mode="hold_grasp",
                    max_retries=2,
                    close_gripper_value=self.grasp_val,
                    grasp_depth=self.GRASP_Z_OFFSET,
                    retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                    grasp_settle_time=self.grasp_settle_time,
                    test_lift_height=self.test_lift_height,
                    pose_constraint=[1, 1, 1, 0, 0, 0],
                    ee_link="tcp_link_bottom",
                )
                repaired = repaired and bool(ok)
            return repaired

        def _drag_once(
            distance: float,
            label: str,
            active_left: bool = True,
            active_right: bool = True,
            check_hold: bool = True,
        ) -> Generator[ExecutionStep, None, Tuple[bool, bool]]:
            if not active_left and not active_right:
                return _ee_near_cloth_hold(label) if check_hold else (True, True)

            obs = self.reset_interface_state()
            l_ee = obs.eef_pose_left[:3].copy()
            r_ee = obs.eef_pose_right[:3].copy()

            drag_l_xy = _toward_xy(l_ee[:2], lbase_xy, distance)
            drag_r_xy = _toward_xy(r_ee[:2], rbase_xy, distance)

            logger.info(
                f"[TshirtFoldV2] drag_fold_edge/{label}: "
                f"from_l={l_ee[:2]}, from_r={r_ee[:2]}, "
                f"drag_l={drag_l_xy}, drag_r={drag_r_xy}, "
                f"separation={np.linalg.norm(l_ee[:2] - r_ee[:2]):.3f}m, "
                f"dist={distance:.3f}m, active_left={active_left}, active_right={active_right}"
            )

            # Descend to place_z at current XY, then drag horizontally.
            left_target = [
                self._make_place_pose(np.array([l_ee[0], l_ee[1], 0.06]), "left", self.place_z),
                self._make_place_pose(np.array([drag_l_xy[0], drag_l_xy[1], 0.06]), "left", self.place_z),
            ]
            right_target = [
                self._make_place_pose(np.array([r_ee[0], r_ee[1], 0.06]), "right", self.place_z),
                self._make_place_pose(np.array([drag_r_xy[0], drag_r_xy[1], 0.06]), "right", self.place_z),
            ]
            left_goal_type = "waypoints"
            right_goal_type = "waypoints"
            if not active_left:
                left_target = obs.joint_left
                left_goal_type = "joint"
            if not active_right:
                right_target = obs.joint_right
                right_goal_type = "joint"

            try:
                plan_drag = self.skills.move_dual_arm_to(
                    left_target_pose=left_target,
                    right_target_pose=right_target,
                    left_goal_type=left_goal_type,
                    right_goal_type=right_goal_type,
                    left_gripper=self.grasp_val,
                    right_gripper=self.grasp_val,
                    pose_constraint=[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
                    ee_link="tcp_link_bottom",
                    velocity_scale=0.15,
                )
            except Exception as exc:
                logger.warning(f"[TshirtFoldV2] drag_edge/{label} drag planning failed: {exc}")
                if not check_hold:
                    raise RuntimeError(f"Failed to plan edge drag: {exc}") from exc
                return _ee_near_cloth_hold(label)
            if not getattr(plan_drag, "success", False):
                logger.warning(f"[TshirtFoldV2] drag_edge/{label} drag plan failed: {plan_drag}")
                if not check_hold:
                    raise RuntimeError(f"Failed to plan edge drag: {plan_drag}")
                return _ee_near_cloth_hold(label)
            yield ExecutionStep(plan=plan_drag, settle_time=0.3, description=f"drag_edge_{label}_drag")

            return _ee_near_cloth_hold(label) if check_hold else (True, True)

        physics_snapshot = self._apply_stage_physics("drag_edge")
        try:
            if self.DRAG_EDGE_PRE_RELEASE_ABORT:
                # This policy deliberately has no grasp test lift, no grasp
                # verification, and no repair.  The only cloth-presence decision
                # is made after the full drag and immediately before release.
                yield from _grasp_edge_without_check()
                yield from _drag_once(drag_dist, "initial", check_hold=False)
                l_held, r_held = _ee_near_cloth_hold("pre_release")
                if not (l_held and r_held):
                    missing = []
                    if not l_held:
                        missing.append("LEFT")
                    if not r_held:
                        missing.append("RIGHT")
                    self.task_success = False
                    raise RuntimeError(
                        "Edge drag failed before release: cloth missing from "
                        + "/".join(missing)
                        + " gripper(s)"
                    )
                yield from self._release_bimanual("drag_edge_release")
                return True

            success = yield from _grasp_edge("initial")
            if not success:
                raise RuntimeError("Failed to grasp fold edge for dragging")

            l_held, r_held = yield from _drag_once(drag_dist, "initial")
            held = bool(l_held and r_held)
            if not held:
                logger.warning(
                    "[TshirtFoldV2] drag_edge initial pull lost cloth; "
                    f"repairing failed side(s), then supplemental pull ({supplemental_drag_dist:.3f}m)"
                )
                repaired = yield from _repair_edge_grasp(l_held, r_held)
                if repaired:
                    l_held, r_held = yield from _drag_once(
                        supplemental_drag_dist,
                        "supplemental",
                        active_left=not l_held,
                        active_right=not r_held,
                    )
                    held = bool(l_held and r_held)
                else:
                    logger.warning("[TshirtFoldV2] drag_edge single-side repair failed")

            yield from self._release_bimanual("drag_edge_release")
            return bool(held)
        finally:
            self._restore_stage_physics(physics_snapshot, "drag_edge")

    def _side_grasp(self, side: ArmSide):
        if side == ArmSide.LEFT:
            src_top, src_hem, dst_top, dst_hem = (
                "left_sleeve_center",
                "left_hem",
                "right_sleeve_center",
                "right_hem",
            )
        else:
            src_top, src_hem, dst_top, dst_hem = (
                "right_sleeve_center",
                "right_hem",
                "left_sleeve_center",
                "left_hem",
            )

        logger.info(
            "[TshirtFoldV2] grasp pair: "
            f"L_arm->{src_top} (idx={self.keypoints[src_top]['index']}), "
            f"R_arm->{src_hem} (idx={self.keypoints[src_hem]['index']}), "
            f"right_hem_inward_offset={self.side_fold_hem_inward_offset:.3f}m; "
            f"fold reference -> {dst_top}/{dst_hem}"
        )

        def side_targets(obs, specific_indices=None, z_offset=None):
            pose_l, pose_r, l_idx, r_idx = self._make_keypoint_grasp_targets(
                obs,
                left_key=src_top,
                right_key=src_hem,
                specific_indices=specific_indices,
                z_offset=z_offset,
            )
            pose_r = self._shift_pose_toward_garment_center(
                pose_r, obs.pcd_cloth_truth
            )
            return pose_l, pose_r, l_idx, r_idx

        if self.weld_grasp:
            success = yield from self.skills.weld_bimanual_grasp(
                env=self.env,
                get_targets_func=side_targets,
                velocity_scale=0.15,
                ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 0, 0, 0],
            )
        else:
            success = yield from self.skills.robust_bimanual_grasp(
                get_targets_func=side_targets,
                max_retries=3,
                close_gripper_value=self.grasp_val,
                grasp_depth=self.GRASP_Z_OFFSET,
                retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                retry_inward_step=self.GRASP_XY_OFFSET_STEP,
                grasp_settle_time=self.grasp_settle_time,
                test_lift_height=self.test_lift_height,
                pose_constraint=[1, 1, 1, 1, 0, 1],
                pose_constraint_frame="local",
                y_retract_offset=self.retract_y_offset,
                ee_link="tcp_link_bottom",
                velocity_scale=0.15,
                slow_approach=self.slow_approach,
                slow_approach_via_distance=self._via_distance,
                slow_approach_final_vel_scale=self.slow_approach_final_vel_scale,
                ground_check=self.grasp_ground_check,
            )
        if not success:
            raise RuntimeError(f"Failed to grasp for side {side}")
            
    def _stage_close_side_fold(self, side: ArmSide, 
                               retract_cloth: bool = False,
                               fixed_z: Optional[float] = None) -> Generator[ExecutionStep, None, bool]:
    
        yield from self._side_grasp(side)

        if side == ArmSide.LEFT:
            src_top, src_hem, dst_top, dst_hem = (
                "left_sleeve_center",
                "left_hem",
                "right_sleeve_center",
                "right_hem",
            )
        else:
            src_top, src_hem, dst_top, dst_hem = (
                "right_sleeve_center",
                "right_hem",
                "left_sleeve_center",
                "left_hem",
            )

        for lift_attempt in range(2):
            obs = self.reset_interface_state()
            cloth_pos = obs.pcd_cloth_truth
            left_target_pos, right_target_pos = self._compute_fold_geometry(
                cloth_pos, src_top, src_hem, dst_top, dst_hem
            )
            l_ee = obs.eef_pose_left[:3].copy()
            r_ee = obs.eef_pose_right[:3].copy()
            _, adaptive_lift_z = self._compute_adaptive_lift_z(
                l_ee[:2], r_ee[:2], left_target_pos[:2], right_target_pos[:2]
            )

            l_lift_z = adaptive_lift_z
            r_lift_z = adaptive_lift_z
            if fixed_z is not None:
                l_lift_z = fixed_z
                r_lift_z = fixed_z
            l_lift = self._make_place_pose(
                np.array([l_ee[0], l_ee[1], 0.06]), "left", l_lift_z
            )
            r_lift = self._make_place_pose(
                np.array([r_ee[0], r_ee[1], 0.06]), "right", r_lift_z
            )

            plan_lift = self.skills.move_dual_arm_to(
                left_target_pose=l_lift,
                right_target_pose=r_lift,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link="tcp_link_bottom",
                velocity_scale=0.4,
            )
            suffix = "" if lift_attempt == 0 else "_retry"
            yield ExecutionStep(
                plan=plan_lift, settle_time=0.3, description=f"side_fold_lift{suffix}"
            )
            l_ok, r_ok = self._side_fold_hold_status(f"close_side_lift{suffix}")
            if l_ok and r_ok:
                break
            if lift_attempt == 1:
                logger.error("[TshirtFoldV2] close side lift still lost cloth after repair; continuing placement")
                break
            repaired = yield from self._repair_side_fold_grasp(side, l_ok, r_ok)
            if not repaired:
                logger.error("[TshirtFoldV2] close side grasp repair failed; continuing placement")
                break

        # Re-observe after lift because the cloth often translates while hanging.
        obs = self.reset_interface_state()
        cloth_pos = obs.pcd_cloth_truth
        left_target_pos, right_target_pos = self._compute_fold_geometry(
            cloth_pos, src_top, src_hem, dst_top, dst_hem
        )
        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        hover_z, recomputed_lift_z = self._compute_adaptive_lift_z(
            l_ee[:2], r_ee[:2], left_target_pos[:2], right_target_pos[:2]
        )
        if fixed_z is not None:
            hover_z = fixed_z
            recomputed_lift_z = fixed_z

        if recomputed_lift_z > adaptive_lift_z + 0.01:
            logger.info(
                f"[TshirtFoldV2] extra clearance needed after reobserve: "
                f"{adaptive_lift_z:.3f} -> {recomputed_lift_z:.3f}"
            )
            relift_l = self._make_place_pose(
                np.array([l_ee[0], l_ee[1], 0.06]), "left", recomputed_lift_z
            )
            relift_r = self._make_place_pose(
                np.array([r_ee[0], r_ee[1], 0.06]), "right", recomputed_lift_z
            )
            relift = self.skills.move_dual_arm_to(
                left_target_pose=relift_l,
                right_target_pose=relift_r,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link="tcp_link_bottom",
                velocity_scale=0.35,
            )
            yield ExecutionStep(plan=relift, settle_time=0.25, description="side_fold_relift")
            obs = self.reset_interface_state()
            cloth_pos = obs.pcd_cloth_truth
            left_target_pos, right_target_pos = self._compute_fold_geometry(
                cloth_pos, src_top, src_hem, dst_top, dst_hem
            )
            hover_z, _ = self._compute_adaptive_lift_z(
                obs.eef_pose_left[:2], obs.eef_pose_right[:2], left_target_pos[:2], right_target_pos[:2]
            )

        arc_z = min(self.place_hover_z_max, max(float(l_ee[2]), float(r_ee[2]), hover_z) + self.arc_midpoint_lift)
        left_arc_pos = np.array([
            0.5 * (l_ee[0] + left_target_pos[0]),
            0.5 * (l_ee[1] + left_target_pos[1]),
            arc_z,
        ])
        right_arc_pos = np.array([
            0.5 * (r_ee[0] + right_target_pos[0]),
            0.5 * (r_ee[1] + right_target_pos[1]),
            arc_z,
        ])

        left_arc = self._make_place_pose(left_arc_pos, "left", left_arc_pos[2])
        right_arc = self._make_place_pose(right_arc_pos, "right", right_arc_pos[2])
        left_place = self._make_place_pose(left_target_pos, "left", self.place_z)
        right_place = self._make_place_pose(right_target_pos, "right", self.place_z)

        logger.info(
            f"[TshirtFoldV2] left_arc={left_arc[:3]}, right_arc={right_arc[:3]}, "
            f"left_place={left_place[:3]}, right_place={right_place[:3]}"
        )

        plan_fold = self.skills.move_dual_arm_to(
            left_target_pose=[left_arc, left_place],
            right_target_pose=[right_arc, right_place],
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
            ee_link="tcp_link_bottom",
            velocity_scale=0.30,
        )
        yield ExecutionStep(plan=plan_fold, settle_time=0.2, description="side_fold_place")

        yield from self._release_bimanual("side_fold_release")

        retract_l = self._make_place_pose(
            np.array([left_target_pos[0], left_target_pos[1], 0.06]), "left", self.safe_height
        )
        retract_r = self._make_place_pose(
            np.array([right_target_pos[0], right_target_pos[1], 0.06]), "right", self.safe_height
        )
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=retract_l,
            right_target_pose=retract_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_retract, settle_time=0.3, description="side_fold_retract")

        # Drag the edge of the garment to align with the original edge.
        if retract_cloth:
            yield from self._drag_fold_edge_inward()

        home_l = self._make_place_pose(np.array([0.2, 0.3, 0.06]), "left", self.safe_height)
        home_r = self._make_place_pose(np.array([0.2, -0.3, 0.06]), "right", self.safe_height)
        plan_home = self.skills.move_dual_arm_to(
            left_target_pose=home_l,
            right_target_pose=home_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_home, settle_time=0.3, description="side_fold_home")
        return True

    def _stage_far_side_fold(
        self,
        side: ArmSide,
        retract_cloth: bool = False,
        fixed_z: Optional[float] = None,
    ) -> Generator[ExecutionStep, None, bool]:
        
        if retract_cloth:
            yield from self._drag_fold_edge_inward()

        yield from self._side_grasp(side)

        for lift_attempt in range(2):
            obs = self.reset_interface_state()
            l_ee = obs.eef_pose_left[:3].copy()
            r_ee = obs.eef_pose_right[:3].copy()
            _, adaptive_lift_z = self._compute_adaptive_lift_z(
                l_ee[:2], r_ee[:2], l_ee[:2], r_ee[:2]
            )

            l_lift_z = adaptive_lift_z
            r_lift_z = adaptive_lift_z
            if fixed_z is not None:
                l_lift_z = fixed_z
                r_lift_z = fixed_z
            l_lift = self._make_place_pose(
                np.array([l_ee[0], l_ee[1], 0.06]), "left", l_lift_z
            )
            r_lift = self._make_place_pose(
                np.array([r_ee[0], r_ee[1], 0.06]), "right", r_lift_z
            )

            plan_lift = self.skills.move_dual_arm_to(
                left_target_pose=l_lift,
                right_target_pose=r_lift,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link="tcp_link_bottom",
                velocity_scale=0.4,
            )
            suffix = "" if lift_attempt == 0 else "_retry"
            yield ExecutionStep(
                plan=plan_lift, settle_time=0.3, description=f"side_fold_lift{suffix}"
            )
            l_ok, r_ok = self._side_fold_hold_status(f"far_side_lift{suffix}")
            if l_ok and r_ok:
                break
            if lift_attempt == 1:
                logger.error("[TshirtFoldV2] far side lift still lost cloth after repair; continuing placement")
                break
            repaired = yield from self._repair_side_fold_grasp(
                side,
                l_ok,
                r_ok,
            )
            if not repaired:
                logger.error("[TshirtFoldV2] far side grasp repair failed; continuing placement")
                break

        # Re-observe after lift because the cloth often translates while hanging.
        obs = self.reset_interface_state()
        cloth_pos = obs.pcd_cloth_truth
        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        lbase_x = float(self.env.left_base[0])
        rbase_x = float(self.env.right_base[0])
        l_near_x = float(cloth_pos[np.argmin(np.abs(cloth_pos[:, 0] - lbase_x)), 0])
        r_near_x = float(cloth_pos[np.argmin(np.abs(cloth_pos[:, 0] - rbase_x)), 0])
        left_target_pos = np.array([l_near_x, l_ee[1], 0.0])
        right_target_pos = np.array([r_near_x, r_ee[1], 0.0])
        lift_z = fixed_z if fixed_z is not None else adaptive_lift_z
        arc_z = min(
            self.place_hover_z_max,
            max(float(l_ee[2]), float(r_ee[2]), lift_z) + self.arc_midpoint_lift,
        )
        left_arc_pos = np.array([
            0.5 * (l_ee[0] + left_target_pos[0]),
            0.5 * (l_ee[1] + left_target_pos[1]),
            arc_z,
        ])
        right_arc_pos = np.array([
            0.5 * (r_ee[0] + right_target_pos[0]),
            0.5 * (r_ee[1] + right_target_pos[1]),
            arc_z,
        ])

        left_arc = self._make_place_pose(left_arc_pos, "left", left_arc_pos[2])
        right_arc = self._make_place_pose(right_arc_pos, "right", right_arc_pos[2])
        left_place = self._make_place_pose(left_target_pos, "left", self.place_z)
        right_place = self._make_place_pose(right_target_pos, "right", self.place_z)

        logger.info(
            f"[TshirtFoldV2] left_arc={left_arc[:3]}, right_arc={right_arc[:3]}, "
            f"left_place={left_place[:3]}, right_place={right_place[:3]}"
        )

        plan_fold = self.skills.move_dual_arm_to(
            left_target_pose=[left_arc, left_place],
            right_target_pose=[right_arc, right_place],
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
            ee_link="tcp_link_bottom",
            velocity_scale=0.30,
        )
        yield ExecutionStep(plan=plan_fold, settle_time=0.2, description="side_fold_place")

        yield from self._release_bimanual("side_fold_release")

        retract_l = self._make_place_pose(
            np.array([left_target_pos[0], left_target_pos[1], 0.06]), "left", self.safe_height
        )
        retract_r = self._make_place_pose(
            np.array([right_target_pos[0], right_target_pos[1], 0.06]), "right", self.safe_height
        )
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=retract_l,
            right_target_pose=retract_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_retract, settle_time=0.3, description="side_fold_retract")

        home_l = self._make_place_pose(np.array([0.2, 0.3, 0.06]), "left", self.safe_height)
        home_r = self._make_place_pose(np.array([0.2, -0.3, 0.06]), "right", self.safe_height)
        plan_home = self.skills.move_dual_arm_to(
            left_target_pose=home_l,
            right_target_pose=home_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_home, settle_time=0.3, description="side_fold_home")
        return True
    
    def _stage_forward_fold(self) -> Generator[ExecutionStep, None, bool]:
        """Grab the hem (2nd-lowest X) with the right arm and fold it toward the collar.

        Point selection: sorts cloth particles by X; skips the absolute tip (index 0)
        and uses the 2nd-closest particle to the X minimum for a more reliable pinch.
        Fold destination: midpoint of left_collar / right_collar keypoints.
        """
        def neck_target(obs, locked_index=None, z_offset=None):
            cp = obs.pcd_cloth_truth
            if locked_index is not None:
                idx = locked_index
                neck_z = float(cp[:, 2].max())
            else:
                # Collar region: points whose Y is within the top 5 % of cloth Y range.
                y_min, y_max = float(cp[:, 1].min()), float(cp[:, 1].max())
                y_thresh = y_min + 0.95 * max(y_max - y_min, 1e-4)
                collar_mask = cp[:, 1] >= y_thresh
                collar_idxs = np.where(collar_mask)[0]
                # Among collar candidates, pick the one closest to the collar X center.
                x_center = float(np.median(cp[collar_idxs, 0]))
                x_dists = np.abs(cp[collar_idxs, 0] - x_center)
                idx = int(collar_idxs[np.argmin(x_dists)])
                neck_z = float(cp[collar_idxs, 2].max())
            pos = cp[idx].copy()
            pos[2] = neck_z
            pose = self._make_grasp_pose(pos, "right", z_offset)
            logger.debug(f"[ForwardFold] neck_center idx={idx} pos={pos[:2]} z={neck_z:.4f}")
            return pose, idx
        
        def hem_target(obs, locked_index=None, z_offset=None):
            cp = obs.pcd_cloth_truth
            if locked_index is not None:
                idx = locked_index
                hem_z = float(cp[:, 2].min())
            else:
                # Bottom region: points whose Y is within the lowest 5 % of cloth Y range.
                y_min, y_max = float(cp[:, 1].min()), float(cp[:, 1].max())
                y_thresh = y_min + 0.05 * max(y_max - y_min, 1e-4)
                bottom_mask = cp[:, 1] <= y_thresh
                bottom_idxs = np.where(bottom_mask)[0]
                # Among bottom candidates, pick the one closest to the hem X center.
                x_center = float(np.median(cp[bottom_idxs, 0]))
                x_dists = np.abs(cp[bottom_idxs, 0] - x_center)
                idx = int(bottom_idxs[np.argmin(x_dists)])
                hem_z = float(cp[bottom_idxs, 2].min())
            pos = cp[idx].copy()
            pos[2] = hem_z
            pose = self._make_zero_z_grasp_pose(pos, "right")
            logger.debug(f"[ForwardFold] hem_center idx={idx} pos={pos[:2]} z={hem_z:.4f}")
            return pose, idx

        if self.weld_grasp:
            success = yield from self.skills.weld_single_grasp(
                env=self.env,
                arm=ArmSide.RIGHT,
                get_target_func=hem_target,
                velocity_scale=0.3,
                ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 0, 0, 0],
            )
        else:
            success = yield from self.skills.robust_single_grasp(
                arm=ArmSide.RIGHT,
                get_target_func=hem_target,
                other_arm_mode="hold",
                max_retries=3,
                close_gripper_value=self.grasp_val,
                grasp_depth=self.GRASP_Z_OFFSET,
                retry_depth_step=self.GRASP_Z_OFFSET_STEP,
                retry_inward_step=self.GRASP_XY_OFFSET_STEP,
                grasp_settle_time=self.grasp_settle_time,
                test_lift_height=self.test_lift_height,
                pose_constraint=[1, 1, 1, 0, 0, 0],
                ee_link="tcp_link_bottom",
            )
        if not success:
            raise RuntimeError("Failed to grasp hem for forward fold")

        obs = self.reset_interface_state()
        r_ee = obs.eef_pose_right[:3].copy()
        l_joints = obs.joint_left.copy()

        # Lift to clear cloth layers before translating
        lift_z = max(self.forward_fold_lift_height, r_ee[2] + 0.12)
        lift_pose = self._make_place_pose(r_ee, "right", lift_z)
        plan_lift = self.skills.move_dual_arm_to(
            left_target_pose=l_joints,
            left_goal_type="joint",
            right_target_pose=lift_pose,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.4,
        )
        yield ExecutionStep(plan=plan_lift, settle_time=0.2, description="forward_fold_lift")

        # Re-observe at the lifted pose, then choose the final fold destination.
        obs = self.reset_interface_state()
        cloth_pos = obs.pcd_cloth_truth
        r_ee = obs.eef_pose_right[:3].copy()
        l_joints = obs.joint_left.copy()
        _, neck_idx = neck_target(obs)
        collar_center = cloth_pos[neck_idx].copy()

        # Fold destination: bias fold crease toward hem to avoid dragging cloth.
        # ratio=1.0 -> hem lands exactly at collar (perfect half-fold);
        # ratio<1.0 -> crease shifts toward hem, hem lands short of collar.
        fold_dest = r_ee + self.forward_fold_ratio * (collar_center[:3] - r_ee)

        logger.info(
            f"[TshirtFoldV2] forward_fold after_lift: hem={r_ee[:2]}, "
            f"collar={collar_center[:2]}, dest={fold_dest[:2]} "
            f"(ratio={self.forward_fold_ratio})"
        )

        # Place: move directly from the lifted hem pose to the final fold destination.
        dest_pose = self._make_place_pose(fold_dest, "right", self.forward_fold_place_z)
        plan_fold = self.skills.move_dual_arm_to(
            left_target_pose=l_joints,
            left_goal_type="joint",
            right_target_pose=dest_pose,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.2,
        )
        yield ExecutionStep(plan=plan_fold, settle_time=0.3, description="forward_fold_place")

        yield from self._release_single(ArmSide.RIGHT, "forward_fold_release")

        retract_r = self._make_place_pose(
            np.array([dest_pose[0], dest_pose[1], 0.06]), "right", self.safe_height
        )
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=l_joints,
            left_goal_type="joint",
            right_target_pose=retract_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_retract, settle_time=0.2, description="forward_fold_retract")
        return True
    
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        logger.info(f"[TshirtFoldV2] Starting episode {episode_id}")
        self.task_meta = {
            "episode_id": episode_id,
            "description": "Single-stage T-shirt side fold: one sleeve-center + one hem",
            "fold_side": "left",
        }

        self._via_distance = 0.0
        if self.slow_approach:
            self._via_distance = float(np.random.uniform(*self.via_distance_range))
            logger.info(
                f"[TshirtFoldV2] slow_approach via_distance={self._via_distance:.3f}m"
            )

        if self.yaw_jitter_enabled:
            lo, hi = self.yaw_jitter_range_rad
            self.yaw_jitter_L = float(np.random.uniform(lo, hi))
            self.yaw_jitter_R = float(np.random.uniform(lo, hi))
            logger.info(
                f"[TshirtFoldV2] yaw_jitter L={np.rad2deg(self.yaw_jitter_L):+.2f}° "
                f"R={np.rad2deg(self.yaw_jitter_R):+.2f}°"
            )
        else:
            self.yaw_jitter_L = 0.0
            self.yaw_jitter_R = 0.0

        with self.recording("stage1_side_fold"):
            success = yield from self._stage_close_side_fold(ArmSide.LEFT, True)
            if not success:
                raise RuntimeError("Stage 1 close side fold failed")
            success = yield from self._stage_far_side_fold(ArmSide.RIGHT, fixed_z=0.1)
            if not success:
                raise RuntimeError("Stage 1 far side fold failed")

        with self.recording("stage2_forward_fold"):
            physics_snapshot = self._apply_stage_physics("forward_fold")
            try:
                success = yield from self._stage_forward_fold()
                if not success:
                    raise RuntimeError("Stage 2 forward fold failed")
            finally:
                self._restore_stage_physics(physics_snapshot, "forward_fold")

        self.task_success = True
        logger.success("[TshirtFoldV2] Stage 1 side fold completed")


# ======================================================================
# Closed-loop eval scorer (eval_sim EVALUATORS entry)
# ======================================================================
class TshirtSideFoldEvaluator:
    """Staged keypoint scorer for fold_tshirt_side (3 fold steps -> 30/30/40).

    Design (RoboDojo x DexGarmentLab x flatten-evaluator synthesis):
      * Steps mirror the scripted expert: (1) left sleeve-center+hem folded to
        the garment midline, (2) right side likewise, (3) forward fold (hem
        carried past the body midpoint toward the collar).
      * All geometry lives in a garment-local frame FROZEN AT THE FIRST POLL
        (v-axis hem_mid->collar_mid, u-axis lateral) so the predicates are
        translation-invariant and thresholds scale with the measured garment
        size (w0/l0), not absolute meters.
      * A step latches when its keypoints reach the target region while
        RESTING on the stack (z guard) — a gripper holding the sleeve at the
        midline mid-air does not count.
      * Step 3 credit additionally requires the SETTLED end state to pass the
        envelope gates (footprint extents shrunk to a folded-shirt box and the
        stack on the table) — the DexGarmentLab-style guard against a balled-up
        shirt cashing in the forward-fold points.
      * Every poll records the full measurement row (eval_metrics), so any
        other score formula is recomposable offline without re-running.
    """

    FINAL_MEAN_Z = 0.09         # settled folded stack: mean particle height (hanging cloth >0.1)
    FINAL_MAX_Z = 0.30          # coarse guard only — a standing fold corner reaches 0.21 (expert ep0)
    KP_REST_Z = 0.10            # keypoint resting on the stack (expert rest z up to ~0.08-0.10)
    KP_REST_Z_FWD = 0.14        # hem on the full stack ~0.10; transport is >=0.19
    LAT_FRAC = 0.35             # |u(kp)-u(center)| <= LAT_FRAC * initial |u|  == folded to midline
    FWD_V_FRAC = 0.10           # v(hem_mid)-v(center) >= FWD_V_FRAC * l0  == carried past center
    EXTENT_U_FRAC = 0.60        # settled footprint width  <= frac * w0
    EXTENT_V_FRAC = 0.65        # settled footprint length <= frac * l0
    CLOTH_PARTICLE_RADIUS = 0.005

    early_exit = False

    def __init__(self, spec):
        self.spec = spec
        p = spec.task_params
        settle_sec = float(p.get("terminal_settle_sec", 2.0))
        self.terminal_settle_ticks = max(1, int(round(spec.control_fps * settle_sec)))
        self._kp_idx: Optional[Dict[str, int]] = None
        self.reset_eval()

    SIDE_KPS = ("left_sleeve_center", "left_hem", "right_sleeve_center", "right_hem")

    def reset_eval(self) -> None:
        self._series: list[dict] = []
        self._frame = None          # (c0, u_hat, v_hat, w0, l0, uL0, uR0)
        self._flags = {"left_folded": False, "right_folded": False, "forward_folded": False}
        self._flag_poll = {k: None for k in self._flags}
        # Side-fold keypoints latch INDEPENDENTLY (expert ep0: the right sleeve
        # settles at poll 14 but the right hem only lands, on top of the stack,
        # after the forward fold — a joint same-poll test never fires).
        self._kp_latch = {k: None for k in self.SIDE_KPS}

    # ---- keypoints ----------------------------------------------------
    def _keypoint_indices(self, env) -> Dict[str, int]:
        if self._kp_idx is None:
            kps = BaseFoldTask._load_keypoints_from_asset(env)
            self._kp_idx = {k: int(v["index"]) for k, v in kps.items()}
        return self._kp_idx

    def _kp(self, cloth, name: str) -> np.ndarray:
        return cloth[self._kp_idx[name]]

    # ---- garment-local frame (frozen at first poll) --------------------
    def _init_frame(self, cloth) -> None:
        hem_mid = 0.5 * (self._kp(cloth, "left_hem") + self._kp(cloth, "right_hem"))[:2]
        collar_mid = 0.5 * (self._kp(cloth, "left_collar") + self._kp(cloth, "right_collar"))[:2]
        v_hat = collar_mid - hem_mid
        l0 = float(np.linalg.norm(v_hat))
        if l0 < 1e-6:
            return
        v_hat = v_hat / l0
        u_hat = np.array([v_hat[1], -v_hat[0]])
        c0 = 0.5 * (hem_mid + collar_mid)
        uv = (cloth[:, :2] - c0) @ np.stack([u_hat, v_hat], axis=1)
        w0 = float(np.percentile(uv[:, 0], 97.5) - np.percentile(uv[:, 0], 2.5))
        uL0 = float(np.mean([abs(self._u(c0, u_hat, self._kp(cloth, k)))
                             for k in ("left_sleeve_center", "left_hem")]))
        uR0 = float(np.mean([abs(self._u(c0, u_hat, self._kp(cloth, k)))
                             for k in ("right_sleeve_center", "right_hem")]))
        self._frame = dict(c0=c0, u_hat=u_hat, v_hat=v_hat, w0=w0, l0=l0, uL0=uL0, uR0=uR0)
        logger.info(f"[FoldEval] frame: w0={w0:.3f} l0={l0:.3f} uL0={uL0:.3f} uR0={uR0:.3f}")

    @staticmethod
    def _u(c0, u_hat, p) -> float:
        return float(np.dot(p[:2] - c0, u_hat))

    # ---- per-poll measurement ------------------------------------------
    def _measure(self, env) -> Optional[dict]:
        cloth = env.get_cloth_position()
        if cloth is None:
            return None
        self._keypoint_indices(env)
        if self._frame is None:
            self._init_frame(cloth)
            if self._frame is None:
                return None
        f = self._frame
        uv = (cloth[:, :2] - f["c0"]) @ np.stack([f["u_hat"], f["v_hat"]], axis=1)
        rec = {
            "max_z": round(float(cloth[:, 2].max()), 4),
            "mean_z": round(float(cloth[:, 2].mean()), 4),
            "extent_u": round(float(np.percentile(uv[:, 0], 97.5) - np.percentile(uv[:, 0], 2.5)), 4),
            "extent_v": round(float(np.percentile(uv[:, 1], 97.5) - np.percentile(uv[:, 1], 2.5)), 4),
        }
        for name in ("left_sleeve_center", "left_hem", "right_sleeve_center", "right_hem",
                     "center"):
            p = self._kp(cloth, name)
            rec[name] = [round(self._u(f["c0"], f["u_hat"], p), 4),
                         round(float(np.dot(p[:2] - f["c0"], f["v_hat"])), 4),
                         round(float(p[2]), 4)]
        hem_mid = 0.5 * (self._kp(cloth, "left_hem") + self._kp(cloth, "right_hem"))
        rec["hem_mid"] = [round(self._u(f["c0"], f["u_hat"], hem_mid), 4),
                          round(float(np.dot(hem_mid[:2] - f["c0"], f["v_hat"])), 4),
                          round(float(hem_mid[2]), 4)]
        return rec

    def _kp_in_place(self, rec: dict, name: str) -> bool:
        # Relative to the torso-center keypoint -> translation invariant (the
        # scripted drag shifts the whole garment ~0.3m mid-episode).
        f = self._frame
        u_ref = max(f["uL0" if name.startswith("left") else "uR0"], 1e-6)
        kp = rec[name]
        return (abs(kp[0] - rec["center"][0]) <= self.LAT_FRAC * u_ref
                and kp[2] <= self.KP_REST_Z)

    def _forward_folded(self, rec: dict) -> bool:
        hm = rec["hem_mid"]
        return (hm[1] - rec["center"][1] >= self.FWD_V_FRAC * self._frame["l0"]
                and hm[2] <= self.KP_REST_Z_FWD)

    def observe(self, env) -> None:
        rec = self._measure(env)
        if rec is None:
            return
        self._series.append(rec)
        poll = len(self._series) - 1
        for name in self.SIDE_KPS:
            if self._kp_latch[name] is None and self._kp_in_place(rec, name):
                self._kp_latch[name] = poll
                logger.info(f"[FoldEval] keypoint in place: {name} @poll{poll}")
        # Tiered (RoboDojo-style): the forward fold only counts once BOTH side
        # folds are done — flipping a crumpled shirt over is not step 3 (user
        # review of ep0: a rolled-over unfolded shirt must score 0, not 40).
        # Sides are settled first so a same-poll side+forward completion counts.
        def _latch(flag: str, ok: bool) -> None:
            if not self._flags[flag] and ok:
                self._flags[flag] = True
                self._flag_poll[flag] = poll
                logger.info(f"[FoldEval] step latched: {flag} @poll{poll}")

        _latch("left_folded", self._kp_latch["left_sleeve_center"] is not None
                              and self._kp_latch["left_hem"] is not None)
        _latch("right_folded", self._kp_latch["right_sleeve_center"] is not None
                               and self._kp_latch["right_hem"] is not None)
        _latch("forward_folded", self._flags["left_folded"] and self._flags["right_folded"]
                                 and self._forward_folded(rec))

    # ---- end-state gates + score ---------------------------------------
    def _end_gates_ok(self) -> bool:
        if not self._series or self._frame is None:
            return False
        last = self._series[-1]
        return (last["mean_z"] < self.FINAL_MEAN_Z
                and last["max_z"] < self.FINAL_MAX_Z
                and last["extent_u"] <= self.EXTENT_U_FRAC * self._frame["w0"]
                and last["extent_v"] <= self.EXTENT_V_FRAC * self._frame["l0"])

    def check_success(self) -> bool:
        return (self._flags["left_folded"] and self._flags["right_folded"]
                and self._flags["forward_folded"] and self._end_gates_ok())

    def get_score(self) -> float:
        score = 30.0 * self._flags["left_folded"] + 30.0 * self._flags["right_folded"]
        if self._flags["forward_folded"] and self._end_gates_ok():
            score += 40.0
        return round(score, 1)

    def eval_metrics(self) -> dict:
        f = self._frame or {}
        return {
            "flags": dict(self._flags),
            "flag_poll": dict(self._flag_poll),
            "kp_latch": dict(self._kp_latch),
            "end_gates_ok": self._end_gates_ok(),
            "frame": {k: (v.tolist() if isinstance(v, np.ndarray) else round(v, 4))
                      for k, v in f.items()},
            "thresholds": {"LAT_FRAC": self.LAT_FRAC, "FWD_V_FRAC": self.FWD_V_FRAC,
                           "KP_REST_Z": self.KP_REST_Z, "KP_REST_Z_FWD": self.KP_REST_Z_FWD,
                           "FINAL_MEAN_Z": self.FINAL_MEAN_Z, "FINAL_MAX_Z": self.FINAL_MAX_Z,
                           "EXTENT_U_FRAC": self.EXTENT_U_FRAC,
                           "EXTENT_V_FRAC": self.EXTENT_V_FRAC},
            "series": self._series,
        }
