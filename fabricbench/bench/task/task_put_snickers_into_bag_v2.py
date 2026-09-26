"""task/task_put_snickers_into_bag_v2.py

Single-arm pick & drop: grasp the rigid Snickers bar and release it into the
non-woven bag mouth. Ported from deformable_bench (Stage A of pack_snicker_v2);
the MuJoCo body queries are replaced by the newton env's rigid-body interface
(``env.get_rigid_body_pose``), everything else is verbatim.

Pipeline:
  A1 hold home + cloth/snickers settle, derive bag mouth
  A2 single-arm move to snickers grasp pose
  A3 close gripper on snickers
  A4 lift + waypoint above bag mouth
  A5 descend into bag
  A6 release snickers

Requires ``control_mode: pd``: the bar is a free rigid body and only real
contact forces (SolverMuJoCo) can hold it — kinematic mode has none.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Generator, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from fabricbench.bench.common.types import ArmSide, ExecutionStep
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.base_task import BaseTask


class PutSnickersIntoBagV2Task(BaseTask):
    """Single-arm snickers pick-and-place into bag mouth (Stage A of v2)."""

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        *,
        cfg: Optional[DictConfig] = None,
        collector=None,
        # ============================================================
        # Snickers segment — single-arm pick & drop into bag.
        # ============================================================
        snickers_arm: str = "right",
        snickers_body_name: str = "snickers_body",
        snickers_grasp_close_val: float = 0.0,
        snickers_grasp_height_offset: float = -0.018,
        snickers_grasp_z: float = 0.005,
        snickers_lift_z: float = 0.35,
        snickers_above_mouth_offset: float = 0.10,
        snickers_descend_offset: float = 0.08,
        snickers_mouth_dx: float = 0.0,
        snickers_mouth_dy: float = 0.0,
        snickers_release_val: float = 1.0,
        snickers_move_to_grasp_velocity_scale: float = 1.0,
        snickers_close_duration: float = 0.4,
        snickers_close_settle_time: float = 0.1,
        snickers_transport_velocity_scale: Tuple[float, float] = (0.2, 0.1),
        snickers_transport_settle_time: float = 0.3,
        snickers_descend_velocity_scale: float = 0.4,
        snickers_descend_settle_time: float = 0.3,
        snickers_release_duration: float = 0.6,
        snickers_release_settle_time: float = 0.0,
        # success threshold for snickers placement
        snickers_success_xy_radius: float = 0.12,
        snickers_success_z_below_mouth: float = 0.02,
    ):
        # ---- yaml overrides ----
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})
            snickers_arm = str(tp.get("snickers_arm", snickers_arm))
            snickers_body_name = str(tp.get("snickers_body_name", snickers_body_name))
            snickers_grasp_close_val = float(tp.get("snickers_grasp_close_val", snickers_grasp_close_val))
            snickers_grasp_height_offset = float(tp.get("snickers_grasp_height_offset", snickers_grasp_height_offset))
            snickers_grasp_z = float(tp.get("snickers_grasp_z", snickers_grasp_z))
            snickers_lift_z = float(tp.get("snickers_lift_z", snickers_lift_z))
            snickers_above_mouth_offset = float(tp.get("snickers_above_mouth_offset", snickers_above_mouth_offset))
            snickers_descend_offset = float(tp.get("snickers_descend_offset", snickers_descend_offset))
            snickers_mouth_dx = float(tp.get("snickers_mouth_dx", snickers_mouth_dx))
            snickers_mouth_dy = float(tp.get("snickers_mouth_dy", snickers_mouth_dy))
            snickers_release_val = float(tp.get("snickers_release_val", snickers_release_val))
            snickers_move_to_grasp_velocity_scale = float(tp.get(
                "snickers_move_to_grasp_velocity_scale", snickers_move_to_grasp_velocity_scale))
            snickers_close_duration = float(tp.get("snickers_close_duration", snickers_close_duration))
            snickers_close_settle_time = float(tp.get("snickers_close_settle_time", snickers_close_settle_time))
            _trans = tp.get("snickers_transport_velocity_scale", snickers_transport_velocity_scale)
            snickers_transport_velocity_scale = tuple(float(v) for v in _trans)
            snickers_transport_settle_time = float(tp.get("snickers_transport_settle_time", snickers_transport_settle_time))
            snickers_descend_velocity_scale = float(tp.get("snickers_descend_velocity_scale", snickers_descend_velocity_scale))
            snickers_descend_settle_time = float(tp.get("snickers_descend_settle_time", snickers_descend_settle_time))
            snickers_release_duration = float(tp.get("snickers_release_duration", snickers_release_duration))
            snickers_release_settle_time = float(tp.get("snickers_release_settle_time", snickers_release_settle_time))
            snickers_success_xy_radius = float(tp.get("snickers_success_xy_radius", snickers_success_xy_radius))
            snickers_success_z_below_mouth = float(tp.get("snickers_success_z_below_mouth", snickers_success_z_below_mouth))

        super().__init__(env, planner_manager, collector, cfg)

        # ---- snickers state ----
        sa = snickers_arm.lower().strip()
        if sa not in ("left", "right"):
            raise ValueError(f"snickers_arm must be 'left'|'right', got {snickers_arm!r}")
        self.snickers_arm_str = sa
        self.snickers_arm_side = ArmSide.LEFT if sa == "left" else ArmSide.RIGHT

        self.snickers_body_name = snickers_body_name
        if snickers_body_name not in getattr(env, "rigid_body_index", {}):
            raise RuntimeError(
                f"rigid body {snickers_body_name!r} not in scene — declare it under "
                f"the task yaml's `rigid_objects:` block (have: "
                f"{list(getattr(env, 'rigid_body_index', {}))})"
            )

        self.snickers_grasp_close_val = float(snickers_grasp_close_val)
        self.snickers_grasp_height_offset = float(snickers_grasp_height_offset)
        self.snickers_grasp_z = float(snickers_grasp_z)
        self.snickers_lift_z = float(snickers_lift_z)
        self.snickers_above_mouth_offset = float(snickers_above_mouth_offset)
        self.snickers_descend_offset = float(snickers_descend_offset)
        self.snickers_mouth_dx = float(snickers_mouth_dx)
        self.snickers_mouth_dy = float(snickers_mouth_dy)
        self.snickers_release_val = float(snickers_release_val)
        self.snickers_move_to_grasp_velocity_scale = float(snickers_move_to_grasp_velocity_scale)
        self.snickers_close_duration = float(snickers_close_duration)
        self.snickers_close_settle_time = float(snickers_close_settle_time)
        self.snickers_transport_velocity_scale = list(snickers_transport_velocity_scale)
        self.snickers_transport_settle_time = float(snickers_transport_settle_time)
        self.snickers_descend_velocity_scale = float(snickers_descend_velocity_scale)
        self.snickers_descend_settle_time = float(snickers_descend_settle_time)
        self.snickers_release_duration = float(snickers_release_duration)
        self.snickers_release_settle_time = float(snickers_release_settle_time)
        self.snickers_success_xy_radius = float(snickers_success_xy_radius)
        self.snickers_success_z_below_mouth = float(snickers_success_z_below_mouth)

        # Auto-derived per episode (in plan_episode); also re-evaluated in check_success.
        self._mouth_xy: Optional[Tuple[float, float]] = None
        self._mouth_z: Optional[float] = None
        self._strap_idx = self._load_strap_topology()

        logger.info(
            f"[put_v2] snickers arm={self.snickers_arm_str}  "
            f"success_xy_r={self.snickers_success_xy_radius:.3f} "
            f"below_mouth={self.snickers_success_z_below_mouth:.3f}"
        )

    # ----------------------------------------------------------------------
    # Helpers — strap topology / cloth-derived mouth
    # ----------------------------------------------------------------------
    def _load_strap_topology(self) -> Optional[dict]:
        """Glob the cloth dir for *_strap_topology.json. We only need
        ``body_indices`` here so _derive_mouth_from_cloth can exclude strap
        tips when computing the mouth — without the filter, asymmetric strap
        tips inflate mouth_z to ~0.35 (strap height) instead of ~0.27 (rim)."""
        cloth_dir = os.path.dirname(str(self.env.cloth_obj_path))
        candidates = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
        if not candidates:
            logger.warning(f"[put_v2] no *_strap_topology.json in {cloth_dir}")
            return None
        obj_stem = os.path.splitext(os.path.basename(str(self.env.cloth_obj_path)))[0]
        candidates.sort(key=lambda p: -len(os.path.commonprefix([
            os.path.basename(p), obj_stem,
        ])))
        topo_path = candidates[0]
        logger.info(f"[put_v2] using strap topology: {topo_path}")
        with open(topo_path) as f:
            topo = json.load(f)
        out = {}
        if "body_indices" in topo:
            out["body"] = np.asarray(topo["body_indices"], dtype=np.int64)
        logger.info(
            f"[put_v2] strap topology loaded: body={len(out['body']) if 'body' in out else 'n/a'}"
        )
        return out

    def _derive_mouth_from_cloth(self) -> Tuple[Tuple[float, float], float]:
        """Bag mouth xy/z derived from BAG BODY verts only (strap tips excluded
        via strap_topology.body_indices). Same logic as pack_snicker_v2."""
        world = np.asarray(self.env.get_cloth_position(), dtype=np.float64)

        if self._strap_idx is not None and "body" in self._strap_idx:
            body_idx = self._strap_idx["body"]
            sample = world[body_idx]
            src = "body_indices"
        else:
            sample = world
            src = "all (no topology)"
            logger.warning("[put_v2] no body_indices in topology — mouth_xy may be biased by strap tips")

        z_max = float(sample[:, 2].max())
        z_thresh = float(np.quantile(sample[:, 2], 0.90))
        mask = sample[:, 2] >= z_thresh
        top_xy = sample[mask, :2] if int(mask.sum()) >= 32 else sample[np.argsort(sample[:, 2])[-32:], :2]

        cx = float(top_xy[:, 0].mean()) + self.snickers_mouth_dx
        cy = float(top_xy[:, 1].mean()) + self.snickers_mouth_dy
        logger.info(
            f"[put_v2] derive_mouth ({src}): xy=({cx:+.3f},{cy:+.3f})  z_max={z_max:.3f}  "
            f"(dx,dy applied: {self.snickers_mouth_dx:+.3f},{self.snickers_mouth_dy:+.3f})"
        )
        return (cx, cy), z_max

    # ----------------------------------------------------------------------
    # Helpers — snickers / arm
    # ----------------------------------------------------------------------
    def _snickers_xpos(self) -> np.ndarray:
        return self.env.get_rigid_body_pose(self.snickers_body_name)[:3]

    def _arm_base(self) -> np.ndarray:
        base = self.env.left_base if self.snickers_arm_side == ArmSide.LEFT else self.env.right_base
        return np.asarray(base, dtype=np.float64)[:3]

    @staticmethod
    def _build_pose(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])

    # ----------------------------------------------------------------------
    # plan_episode — Stage A only
    # ----------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        # ---- A1: hold home, derive mouth ----
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        state = self.skills.planning_state
        plan_hold = self.skills.move_dual_arm_to(
            left_target_pose  = np.asarray(state.left_joints).copy(),
            right_target_pose = np.asarray(state.right_joints).copy(),
            goal_type     = "joint",
            left_gripper  = state.left_gripper,
            right_gripper = state.right_gripper,
        )
        yield ExecutionStep(plan=plan_hold, settle_time=0.5, description="hold_home")

        self._mouth_xy, self._mouth_z = self._derive_mouth_from_cloth()
        logger.info(f"[put_v2] mouth xy={self._mouth_xy} z={self._mouth_z:.4f}")

        # ---- snickers waypoints ----
        sh_pos = self._snickers_xpos()
        sh_top_z = sh_pos[2] + 0.0269
        grasp_pos = np.array(
            [sh_pos[0], sh_pos[1], sh_top_z + self.snickers_grasp_height_offset],
            dtype=np.float64,
        )
        base = self._arm_base()
        grasp_quat = self.skills.calculate_base_to_target_horizontal_quat(base, grasp_pos)
        grasp_pose = self._build_pose(grasp_pos, grasp_quat)
        grasp_pose[2] = self.snickers_grasp_z

        lift_pose = self._build_pose(grasp_pos, grasp_quat)
        lift_pose[2] = max(self.snickers_lift_z, self._mouth_z + 0.13)

        above_mouth_pos = np.array(
            [self._mouth_xy[0], self._mouth_xy[1], self._mouth_z + self.snickers_above_mouth_offset],
            dtype=np.float64,
        )
        above_mouth_quat = self.skills.calculate_base_to_target_horizontal_quat(base, above_mouth_pos)
        above_mouth_pose = self._build_pose(above_mouth_pos, above_mouth_quat)

        descend_pos = np.array(
            [self._mouth_xy[0], self._mouth_xy[1], self._mouth_z + self.snickers_descend_offset],
            dtype=np.float64,
        )
        descend_pose = self._build_pose(descend_pos, above_mouth_quat)

        self.task_meta = {
            "episode_id": episode_id,
            "arm": self.snickers_arm_str,
            "snickers_pos": sh_pos.tolist(),
            "mouth_xyz": [self._mouth_xy[0], self._mouth_xy[1], self._mouth_z],
            "grasp_pose": grasp_pose.tolist(),
            "lift_pose": lift_pose.tolist(),
            "above_mouth_pose": above_mouth_pose.tolist(),
            "descend_pose": descend_pose.tolist(),
            "description": (
                f"Single-arm ({self.snickers_arm_str}) pick-and-place: "
                f"snickers → bag mouth @"
                f"({self._mouth_xy[0]:.2f},{self._mouth_xy[1]:.2f},{self._mouth_z:.2f})"
            ),
        }

        # ---- A2: move to snickers grasp pose ----
        with self.recording("snickers_move_to_grasp"):
            plan = self.skills.move_to(
                target_pose=grasp_pose, arm=self.snickers_arm_side, gripper=1.0,
                ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 1, 0, 1], pose_constraint_frame="local",
                velocity_scale=self.snickers_move_to_grasp_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=0.5, description="snickers_move_to_grasp")

        # ---- A3: close gripper ----
        with self.recording("snickers_close"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.gripper_action(
                target_val=self.snickers_grasp_close_val, arm=self.snickers_arm_side,
                duration=self.snickers_close_duration,
            )
            yield ExecutionStep(plan=plan, settle_time=self.snickers_close_settle_time, description="snickers_close")

        # ---- A4: lift + waypoint above mouth ----
        with self.recording("snickers_lift_to_above_mouth"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            if self.snickers_arm_side == ArmSide.LEFT:
                self.skills.planning_state.left_gripper = self.snickers_grasp_close_val
            else:
                self.skills.planning_state.right_gripper = self.snickers_grasp_close_val
            plan = self.skills.move_to(
                target_pose=[lift_pose, above_mouth_pose],
                arm=self.snickers_arm_side, goal_type="waypoints",
                gripper=self.snickers_grasp_close_val, ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 0, 0, 0], pose_constraint_frame="local",
                # MUST be list, not tuple — _ensure_list_input only special-cases
                # list/ndarray; a tuple gets broadcast as a single value, yielding
                # "Unknown mode: ('joint', 'joint')" deep in the planner.
                interpolation_mode=["joint", "joint"],
                velocity_scale=self.snickers_transport_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.snickers_transport_settle_time,
                                description="snickers_lift_to_above_mouth")

        # ---- A5: descend into bag ----
        with self.recording("snickers_descend"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            if self.snickers_arm_side == ArmSide.LEFT:
                self.skills.planning_state.left_gripper = self.snickers_grasp_close_val
            else:
                self.skills.planning_state.right_gripper = self.snickers_grasp_close_val
            plan = self.skills.move_to(
                target_pose=descend_pose, arm=self.snickers_arm_side,
                gripper=self.snickers_grasp_close_val, ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 0, 0, 0], pose_constraint_frame="local",
                velocity_scale=self.snickers_descend_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.snickers_descend_settle_time,
                                description="snickers_descend")

        # ---- A6: release snickers ----
        with self.recording("snickers_release"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.gripper_action(
                target_val=self.snickers_release_val, arm=self.snickers_arm_side,
                duration=self.snickers_release_duration,
            )
            yield ExecutionStep(plan=plan, settle_time=self.snickers_release_settle_time, description="snickers_release")

        logger.success("[put_v2] Done.")

    # ----------------------------------------------------------------------
    # check_success — independent: only checks whether snickers landed inside.
    # ----------------------------------------------------------------------
    def check_success(self) -> bool:
        """Snickers landed inside bag mouth: xy distance to live mouth_xy
        within ``snickers_success_xy_radius`` AND snickers z is below
        ``mouth_z - snickers_success_z_below_mouth``."""
        try:
            sh = self._snickers_xpos()
        except Exception as e:
            logger.warning(f"[put_v2 check] snickers xpos failed: {e}")
            return False
        try:
            mouth_xy, mouth_z = self._derive_mouth_from_cloth()
        except Exception as e:
            logger.warning(f"[put_v2 check] mouth derivation failed: {e}")
            return False

        dxy = float(np.linalg.norm(sh[:2] - np.asarray(mouth_xy)))
        below = bool(sh[2] < mouth_z - self.snickers_success_z_below_mouth)
        within = bool(dxy < self.snickers_success_xy_radius)
        ok = bool(within and below)
        logger.info(
            f"[put_v2 check_success] snickers=({sh[0]:.3f},{sh[1]:.3f},{sh[2]:.3f}) "
            f"mouth_xy=({mouth_xy[0]:.3f},{mouth_xy[1]:.3f}) mouth_z={mouth_z:.3f} | "
            f"dxy={dxy:.3f} (thr={self.snickers_success_xy_radius:.3f}) within={within}, "
            f"below_mouth={below} (margin={self.snickers_success_z_below_mouth:.3f}) → {ok}"
        )
        return ok
