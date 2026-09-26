"""task/task_pack_snicker_v2.py

Pack a Snickers bar into a non-woven bag, then grasp the bag's straps and
lift / place. Independently maintained — does NOT import v1 / v4 task code.

Pipeline (high level):
  Stage A1 hold home + cloth/snickers settle
  Stage A2 single-arm move to snickers grasp pose (mid-air)
  Stage A3 close gripper on snickers
  Stage A4 lift + waypoint above bag mouth
  Stage A5 descend into bag
  Stage A6 release snickers
  Stage B0 dual-arm move to lift_silk preset (un-randomized)
  Stage B1 PRE  (IK with v5 seed)
  Stage B2 GRASP (IK with v5 seed)
  Stage B3 close grippers on straps
  Stage B4 LIFT (vertical, joint replay) + strap z snapshot for check_success
  Stage B5 PLACE (joint replay, hardcoded _*_PLACE_JOINTS)
  Stage B6 release grippers + return to lift_silk preset

Success: snickers landed inside bag mouth (xy radius + below mouth_z) AND
both straps' tip+mid mean(z) > strap_lift_thresh at LIFT snapshot.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Generator, Optional, Tuple


import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fabricbench.bench.common.types import ArmSide, ExecutionStep
from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.base_task import BaseTask

import mujoco

# ===========================================================================
# Bag-strap (v4) reference constants — copied verbatim, NOT imported, so this
# file is self-contained. To change the v5 seed / PLACE joints / reference
# quats, edit BOTH this file and task_grasp_bag_strap_single_v4.py.
# ===========================================================================
def _mirror_quat_y(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], q[2], -q[3]], dtype=np.float64)


_LEFT_PRE_SEED   = np.array([+0.3713, +2.0562, -1.5176, -1.3803, +1.2316, +0.8464])
_RIGHT_PRE_SEED  = np.array([-0.3081, +1.9306, -1.5293, +1.2006, +1.2280, -0.1168])
_LEFT_GRASP_SEED = np.array([+0.0986, +2.1238, -1.5176, -1.5683, +1.1509, +0.8464])
_RIGHT_GRASP_SEED= np.array([+0.0632, +1.9825, -1.5248, +1.2731, +1.0008, -0.1939])

_PRE_QUAT_LEFT   = np.array([+0.3006, -0.0571, +0.8060, -0.5067])
_GRASP_QUAT_LEFT = np.array([+0.2554, -0.0391, +0.7750, -0.5768])

_LEFT_PLACE_JOINTS  = np.array([-0.9789, +2.2190, -2.2725, +0.1054, +0.0253, +0.0604])
_RIGHT_PLACE_JOINTS = np.array([-0.3088, +1.0482, -0.9774, -0.3137, +0.1676, +0.1219])


class PackSnickerV2Task(BaseTask):
    """Snickers pick-and-place + bimanual bag-strap grasp + place + return-home.

    All tunables exposed via yaml ``task_params`` (snickers_* and bag_*).
    """

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        *,
        cfg: Optional[DictConfig] = None,
        collector=None,
        urdf_path: Optional[str] = None,
        # ============================================================
        # Snickers segment — single-arm pick & drop into bag.
        # ============================================================
        snickers_arm: str = "right",
        snickers_body_name: str = "snickers_body",
        snickers_grasp_close_val: float = 0.0,
        # bar top-of-mesh offset to gripper (negative = dive in past the top edge)
        snickers_grasp_height_offset: float = -0.018,
        snickers_grasp_z: float = 0.005,
        snickers_lift_z: float = 0.35,
        snickers_above_mouth_offset: float = 0.10,
        snickers_descend_offset: float = 0.08,
        # Optional fine-tune offsets added to auto-derived mouth_xy. Default 0,0
        # = use the auto-derive verbatim. Useful if you want to drop snickers
        # nearer to one strap (e.g. closer to base_pos than the centroid).
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
        # ============================================================
        # Bag-strap segment (v4 logic) — bimanual.
        # ============================================================
        bag_pre_offset_xyz:   Tuple[float, float, float] = (+0.080, 0.278, 0.205),
        bag_grasp_offset_xyz: Tuple[float, float, float] = (+0.095, 0.140, 0.198),
        bag_lift_height: float = 0.45,
        bag_left_dy:  float = 0.0,
        bag_right_dy: float = 0.0,
        bag_left_dz:  float = 0.0,
        bag_right_dz: float = 0.0,
        bag_open_gripper_value:  float = 1.0,
        bag_close_gripper_value: float = 0.0,
        bag_dual_to_home_velocity_scale: float = 0.40,
        bag_home_to_pre_velocity_scale:  float = 0.40,
        bag_pre_to_grasp_velocity_scale: float = 0.15,
        bag_lift_velocity_scale:         float = 0.20,
        bag_return_home_velocity_scale:  float = 0.40,
        bag_pre_settle:        float = 0.0,
        bag_grasp_settle:      float = 0.8,
        bag_close_duration:    float = 1.0,
        bag_post_close_settle: float = 0.5,
        bag_lift_settle:       float = 1.0,
        bag_return_home_settle: float = 0.5,
        # Success criterion (both must hold, snapshotted at LIFT end):
        #   1) both straps' tip + mid mean z > bag_strap_lift_thresh
        #   2) bag centroid z rises by > bag_lift_thresh vs the baseline taken
        #      right after Stage B0 (bag_dual_to_home), i.e. before any gripper
        #      touches the strap. Catches "strap lifted but bag stayed" = slip.
        bag_strap_lift_thresh: float = 0.35,
        bag_lift_thresh: float = 0.05,
    ):
        # ---- yaml overrides ----
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})

            def _xyz(key, default):
                v = tp.get(key, None)
                if v is None:
                    return tuple(float(x) for x in default)
                if hasattr(v, "__iter__"):
                    return tuple(float(x) for x in list(v))
                return tuple(float(x) for x in default)

            # snickers
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
            # bag
            bag_pre_offset_xyz   = _xyz("bag_pre_offset_xyz",   bag_pre_offset_xyz)
            bag_grasp_offset_xyz = _xyz("bag_grasp_offset_xyz", bag_grasp_offset_xyz)
            bag_lift_height = float(tp.get("bag_lift_height", bag_lift_height))
            bag_left_dy   = float(tp.get("bag_left_dy",   bag_left_dy))
            bag_right_dy  = float(tp.get("bag_right_dy",  bag_right_dy))
            bag_left_dz   = float(tp.get("bag_left_dz",   bag_left_dz))
            bag_right_dz  = float(tp.get("bag_right_dz",  bag_right_dz))
            bag_open_gripper_value  = float(tp.get("bag_open_gripper_value",  bag_open_gripper_value))
            bag_close_gripper_value = float(tp.get("bag_close_gripper_value", bag_close_gripper_value))
            bag_dual_to_home_velocity_scale = float(tp.get(
                "bag_dual_to_home_velocity_scale", bag_dual_to_home_velocity_scale))
            bag_home_to_pre_velocity_scale = float(tp.get(
                "bag_home_to_pre_velocity_scale", bag_home_to_pre_velocity_scale))
            bag_pre_to_grasp_velocity_scale = float(tp.get(
                "bag_pre_to_grasp_velocity_scale", bag_pre_to_grasp_velocity_scale))
            bag_lift_velocity_scale = float(tp.get("bag_lift_velocity_scale", bag_lift_velocity_scale))
            bag_return_home_velocity_scale = float(tp.get(
                "bag_return_home_velocity_scale", bag_return_home_velocity_scale))
            bag_pre_settle        = float(tp.get("bag_pre_settle",        bag_pre_settle))
            bag_grasp_settle      = float(tp.get("bag_grasp_settle",      bag_grasp_settle))
            bag_close_duration    = float(tp.get("bag_close_duration",    bag_close_duration))
            bag_post_close_settle = float(tp.get("bag_post_close_settle", bag_post_close_settle))
            bag_lift_settle       = float(tp.get("bag_lift_settle",       bag_lift_settle))
            bag_return_home_settle = float(tp.get("bag_return_home_settle", bag_return_home_settle))
            bag_strap_lift_thresh = float(tp.get("bag_strap_lift_thresh", bag_strap_lift_thresh))
            bag_lift_thresh = float(tp.get("bag_lift_thresh", bag_lift_thresh))

        super().__init__(env, planner_manager, collector, cfg)

        # ---- snickers state ----
        sa = snickers_arm.lower().strip()
        if sa not in ("left", "right"):
            raise ValueError(f"snickers_arm must be 'left'|'right', got {snickers_arm!r}")
        self.snickers_arm_str = sa
        self.snickers_arm_side = ArmSide.LEFT if sa == "left" else ArmSide.RIGHT

        self.snickers_body_name = snickers_body_name
        self.snickers_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, snickers_body_name)
        if self.snickers_body_id < 0:
            raise RuntimeError(f"body {snickers_body_name!r} not found in MuJoCo model")

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

        # ---- bag state ----
        self.bag_pre_offset   = np.asarray(bag_pre_offset_xyz,   dtype=np.float64)
        self.bag_grasp_offset = np.asarray(bag_grasp_offset_xyz, dtype=np.float64)
        self.bag_lift_height = float(bag_lift_height)
        self.bag_left_dy = float(bag_left_dy);  self.bag_right_dy = float(bag_right_dy)
        self.bag_left_dz = float(bag_left_dz);  self.bag_right_dz = float(bag_right_dz)
        self.bag_open_gripper  = float(bag_open_gripper_value)
        self.bag_close_gripper = float(bag_close_gripper_value)
        self.bag_dual_to_home_velocity_scale = float(bag_dual_to_home_velocity_scale)
        self.bag_home_to_pre_velocity_scale  = float(bag_home_to_pre_velocity_scale)
        self.bag_pre_to_grasp_velocity_scale = float(bag_pre_to_grasp_velocity_scale)
        self.bag_lift_velocity_scale         = float(bag_lift_velocity_scale)
        self.bag_return_home_velocity_scale  = float(bag_return_home_velocity_scale)
        self.bag_pre_settle        = float(bag_pre_settle)
        self.bag_grasp_settle      = float(bag_grasp_settle)
        self.bag_close_duration    = float(bag_close_duration)
        self.bag_post_close_settle = float(bag_post_close_settle)
        self.bag_lift_settle       = float(bag_lift_settle)
        self.bag_return_home_settle = float(bag_return_home_settle)
        self.bag_strap_lift_thresh = float(bag_strap_lift_thresh)
        self.bag_lift_thresh = float(bag_lift_thresh)
        self._initial_bag_centroid_z: Optional[float] = None

        # Pinocchio for bag IK (v5 seed)
        if urdf_path is None:
            urdf_path = os.path.join(
                project_root,
                "assets/robot/piper/urdf/piper_with_horizon_gripper.urdf",
            )
        self._kin = PinocchioKinematics(urdf_path, base_link="base_link", ee_link="tcp_link_center")

        # Auto-derived per episode (in plan_episode)
        self._mouth_xy: Optional[Tuple[float, float]] = None
        self._mouth_z: Optional[float] = None
        self._lift_snapshot: Optional[dict] = None
        self._snickers_drop_pose: Optional[dict] = None
        self._strap_idx = self._load_strap_topology()

        logger.info(
            f"[v2] snickers arm={self.snickers_arm_str}  bag PRE off={tuple(self.bag_pre_offset)} "
            f"GRASP off={tuple(self.bag_grasp_offset)}  LIFT z={self.bag_lift_height:.3f}  "
            f"strap_lift_thresh={self.bag_strap_lift_thresh:.3f}"
        )

    # ----------------------------------------------------------------------
    # Helpers — strap topology / home / bag base / cloth centroid
    # ----------------------------------------------------------------------
    def _load_strap_topology(self) -> Optional[dict]:
        """Glob the cloth dir for *_strap_topology.json — obj_path may point at
        source obj while topology is for the runtime_processed obj, so direct
        stem-rewriting is unreliable."""
        import glob
        if getattr(self.env, "cloth_asset", None) is None:
            logger.warning("[v2] env.cloth_asset is None — strap topology unavailable")
            return None
        cloth_dir = os.path.dirname(self.env.cloth_asset.obj_path)
        candidates = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
        if not candidates:
            logger.warning(f"[v2] no *_strap_topology.json in {cloth_dir}")
            return None
        obj_stem = os.path.splitext(os.path.basename(self.env.cloth_asset.obj_path))[0]
        candidates.sort(key=lambda p: -len(os.path.commonprefix([
            os.path.basename(p), obj_stem,
        ])))
        topo_path = candidates[0]
        logger.info(f"[v2] using strap topology: {topo_path}")
        with open(topo_path) as f:
            topo = json.load(f)
        out = {}
        for side in ("left_strap", "right_strap"):
            s = topo[side]
            out[side] = {
                "tip": np.asarray(s["tip_indices"], dtype=np.int64),
                "mid": np.asarray(s["mid_indices"], dtype=np.int64),
            }
        # Bag body verts (excludes both straps) — used by _derive_mouth_from_cloth
        # so the auto-mouth doesn't get pulled toward the asymmetric strap tips.
        if "body_indices" in topo:
            out["body"] = np.asarray(topo["body_indices"], dtype=np.int64)
        logger.info(
            f"[v2] strap topology loaded: "
            f"L tip={len(out['left_strap']['tip'])} mid={len(out['left_strap']['mid'])}  "
            f"R tip={len(out['right_strap']['tip'])} mid={len(out['right_strap']['mid'])}  "
            f"body={len(out['body']) if 'body' in out else 'n/a'}"
        )
        return out

    def _resolve_home_joints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Un-randomized lift_silk preset. env.home_q_* has noise baked in — avoid."""
        ri = getattr(self.env.randomizer, "robot_init", None)
        if ri is not None and getattr(ri, "_preset", None) is not None:
            return ri._preset["home_left"].copy(), ri._preset["home_right"].copy()
        logger.warning("[v2] robot_init preset missing — falling back to env.home_q_*")
        return np.asarray(self.env.home_q_left).copy(), np.asarray(self.env.home_q_right).copy()

    def _bag_base_pos(self) -> np.ndarray:
        try:
            ci = self.env.randomizer.cloth_init
            bp = ci._last_offsets.get("base_pos")
            if bp is not None:
                return np.asarray(bp, dtype=np.float64)
        except Exception:
            pass
        return np.array([0.40, 0.0, 0.0], dtype=np.float64)

    def _cloth_centroid_y(self) -> float:
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None or len(cloth_pos) == 0:
            return float(self._bag_base_pos()[1])
        return float(np.mean(np.asarray(cloth_pos)[:, 1]))

    def _derive_mouth_from_cloth(self) -> Tuple[Tuple[float, float], float]:
        """Bag mouth xy/z derived from BAG BODY verts only (strap tips excluded
        via strap_topology.body_indices). Without filtering, the asymmetric
        strap tips pull mouth_xy off-center and inflate mouth_z to strap-tip
        height (~0.35) instead of bag-rim height (~0.27).

        Strategy:
          - sample = body_indices' world positions (top 10% by z = rim region)
          - mouth_xy = mean(sample.xy) + (mouth_dx, mouth_dy) fine-tune
          - mouth_z  = max(sample.z)   ≈ rim height
        Falls back to all-vert top-10% only if topology missing.
        """
        live = self.env.get_cloth_position()
        if live is None or len(live) == 0:
            asset = self.env.cloth_asset
            verts = np.asarray(asset.vertices, dtype=np.float64)
            default_pos = np.asarray(asset.default_pos, dtype=np.float64).reshape(3)
            qw, qx, qy, qz = (float(v) for v in np.asarray(asset.default_quat).reshape(4))
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=np.float64)
            world = verts @ R.T + default_pos
        else:
            world = np.asarray(live, dtype=np.float64)

        # Restrict to bag body verts if topology supplies the partition.
        if self._strap_idx is not None and "body" in self._strap_idx:
            body_idx = self._strap_idx["body"]
            sample = world[body_idx]
            src = "body_indices"
        else:
            sample = world
            src = "all (no topology)"
            logger.warning("[v2] no body_indices in topology — mouth_xy may be biased by strap tips")

        z_max = float(sample[:, 2].max())
        z_thresh = float(np.quantile(sample[:, 2], 0.90))
        mask = sample[:, 2] >= z_thresh
        top_xy = sample[mask, :2] if int(mask.sum()) >= 32 else sample[np.argsort(sample[:, 2])[-32:], :2]

        cx = float(top_xy[:, 0].mean()) + self.snickers_mouth_dx
        cy = float(top_xy[:, 1].mean()) + self.snickers_mouth_dy
        logger.info(
            f"[v2] derive_mouth ({src}): xy=({cx:+.3f},{cy:+.3f})  z_max={z_max:.3f}  "
            f"(dx,dy applied: {self.snickers_mouth_dx:+.3f},{self.snickers_mouth_dy:+.3f})"
        )
        return (cx, cy), z_max

    # ----------------------------------------------------------------------
    # Helpers — bag IK target builders
    # ----------------------------------------------------------------------
    def _build_world_targets(
        self,
        offset_xyz: np.ndarray, quat_left: np.ndarray,
        l_dy: float = 0.0, r_dy: float = 0.0,
        l_dz: float = 0.0, r_dz: float = 0.0,
        y_ref: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bag = self._bag_base_pos()
        if y_ref is None:
            y_ref = self._cloth_centroid_y()
        l_pos = np.array([
            bag[0] + offset_xyz[0],
            y_ref  + abs(offset_xyz[1]) + l_dy,
            bag[2] + offset_xyz[2] + l_dz,
        ], dtype=np.float64)
        r_pos = np.array([
            bag[0] + offset_xyz[0],
            y_ref  - abs(offset_xyz[1]) + r_dy,
            bag[2] + offset_xyz[2] + r_dz,
        ], dtype=np.float64)
        return (np.concatenate([l_pos, quat_left]),
                np.concatenate([r_pos, _mirror_quat_y(quat_left)]))

    def _solve_ik_pair(
        self, left_world: np.ndarray, right_world: np.ndarray,
        l_seed: np.ndarray, r_seed: np.ndarray, tag: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        l_base = np.asarray(self.env.left_base[:3])
        r_base = np.asarray(self.env.right_base[:3])
        l_target = np.concatenate([left_world[:3]  - l_base, left_world[3:]])
        r_target = np.concatenate([right_world[:3] - r_base, right_world[3:]])
        ik_kwargs = dict(
            pose_constraint=[1, 1, 1, 1, 0, 1],
            constraint_frame="local",
            threshold=5e-3,
            max_restarts=30,
        )
        l_q = self._kin.solve_ik(l_target, seed_joints=l_seed, **ik_kwargs)
        r_q = self._kin.solve_ik(r_target, seed_joints=r_seed, **ik_kwargs)
        if l_q is None: raise RuntimeError(f"[v2] LEFT IK failed at {tag}: pos={l_target[:3]}")
        if r_q is None: raise RuntimeError(f"[v2] RIGHT IK failed at {tag}: pos={r_target[:3]}")
        return l_q[:6], r_q[:6]

    # ----------------------------------------------------------------------
    # Helpers — snickers
    # ----------------------------------------------------------------------
    def _snickers_xpos(self) -> np.ndarray:
        mujoco.mj_forward(self.env.model, self.env.data)
        return np.asarray(self.env.data.xpos[self.snickers_body_id], dtype=np.float64).copy()

    def _arm_base(self) -> np.ndarray:
        base = self.env.left_base if self.snickers_arm_side == ArmSide.LEFT else self.env.right_base
        return np.asarray(base, dtype=np.float64)[:3]

    @staticmethod
    def _build_pose(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])

    def _bag_centroid_z(self) -> Optional[float]:
        """Mean z over all current cloth verts, or None if cloth unavailable."""
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None or len(cloth_pos) == 0:
            return None
        return float(np.mean(np.asarray(cloth_pos)[:, 2]))

    def _snapshot_initial_bag_centroid(self) -> None:
        """Baseline for the bag-lift-delta check. Call right after the bag is
        in place but BEFORE any gripper touches a strap (i.e. after B0
        bag_dual_to_home settle, after snickers section is done)."""
        z = self._bag_centroid_z()
        self._initial_bag_centroid_z = z
        if z is not None:
            logger.info(f"[v2] initial bag centroid z (pre-grasp baseline) = {z:.3f}")

    def _snapshot_strap_z(self) -> None:
        """Snapshot at end of Stage B4 LIFT (before B5 PLACE drops bag). Records
        per-strap tip/mid z + whole-bag centroid z + the pre-grasp baseline."""
        if self._strap_idx is None:
            self._lift_snapshot = None
            return
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None or len(cloth_pos) == 0:
            self._lift_snapshot = None
            return
        snap = {}
        for side in ("left_strap", "right_strap"):
            snap[f"{side}_tip_z"] = float(np.mean(cloth_pos[self._strap_idx[side]["tip"], 2]))
            snap[f"{side}_mid_z"] = float(np.mean(cloth_pos[self._strap_idx[side]["mid"], 2]))
        snap["bag_centroid_z"] = float(np.mean(np.asarray(cloth_pos)[:, 2]))
        snap["bag_centroid_z_initial"] = (
            float(self._initial_bag_centroid_z)
            if self._initial_bag_centroid_z is not None else None
        )
        # Snickers + mouth at LIFT moment. check_success uses these instead
        # of "current" values so place/release moving the bag away doesn't
        # invalidate a successful lift.
        try:
            mouth_xy, mouth_z = self._derive_mouth_from_cloth()
            snap["mouth_xy"] = (float(mouth_xy[0]), float(mouth_xy[1]))
            snap["mouth_z"] = float(mouth_z)
        except Exception:
            snap["mouth_xy"] = None
            snap["mouth_z"] = None
        try:
            sh = self._snickers_xpos()
            snap["snickers_xpos"] = (float(sh[0]), float(sh[1]), float(sh[2]))
        except Exception:
            snap["snickers_xpos"] = None
        self._lift_snapshot = snap
        delta_str = (
            f"Δ={snap['bag_centroid_z'] - snap['bag_centroid_z_initial']:+.3f}"
            if snap["bag_centroid_z_initial"] is not None else "Δ=n/a"
        )
        logger.info(
            f"[v2] LIFT snapshot strap z: "
            f"L tip={snap['left_strap_tip_z']:.3f} mid={snap['left_strap_mid_z']:.3f}  "
            f"R tip={snap['right_strap_tip_z']:.3f} mid={snap['right_strap_mid_z']:.3f}  "
            f"strap_thr={self.bag_strap_lift_thresh:.3f}  "
            f"bag_z={snap['bag_centroid_z']:.3f} ({delta_str}, thr={self.bag_lift_thresh:.3f})"
        )

    # ----------------------------------------------------------------------
    # plan_episode
    # ----------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        # ============================================================
        # Stage A: Snickers pick & place
        # ============================================================
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
        logger.info(f"[v2] mouth xy={self._mouth_xy} z={self._mouth_z:.4f}")

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

        # Snapshot snickers final pose for success check.
        self._snickers_drop_pose = {
            "xy":   self._snickers_xpos()[:2].tolist(),
            "z":    float(self._snickers_xpos()[2]),
        }

        # ============================================================
        # Stage B: Bag-strap grasp & lift & place (v4 logic, inline)
        # ============================================================
        # ---- B0: dual return to lift_silk preset (un-randomized) ----
        # Goes straight from snickers release to dual-arm home — no separate
        # single-arm retreat. The dual_to_home move plans both arms together,
        # the snickers-arm naturally clears the bag mouth en route.
        home_l, home_r = self._resolve_home_joints()
        with self.recording("bag_dual_to_home"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=home_l, right_target_pose=home_r,
                goal_type="joint",
                left_gripper=self.bag_open_gripper, right_gripper=self.bag_open_gripper,
                velocity_scale=self.bag_dual_to_home_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=0.5, description="bag_dual_to_home")

        # Baseline for bag-lift-delta success check — bag has been jostled by
        # snickers placement but is now at rest, grippers still empty/away.
        # Anything moving the bag centroid up from here onward is "real lift".
        self._snapshot_initial_bag_centroid()

        # Snapshot y_ref once after settling, so PRE/GRASP/LIFT share it.
        y_ref = self._cloth_centroid_y()
        bag_xy0 = self._bag_base_pos()
        logger.info(f"[v2] y_ref={y_ref:+.4f}, bag spawn=({bag_xy0[0]:+.3f},{bag_xy0[1]:+.3f},{bag_xy0[2]:+.3f})")

        pre_l_w, pre_r_w = self._build_world_targets(self.bag_pre_offset, _PRE_QUAT_LEFT, y_ref=y_ref)
        gr_l_w,  gr_r_w  = self._build_world_targets(
            self.bag_grasp_offset, _GRASP_QUAT_LEFT,
            l_dy=self.bag_left_dy, r_dy=self.bag_right_dy,
            l_dz=self.bag_left_dz, r_dz=self.bag_right_dz,
            y_ref=y_ref,
        )

        lift_l_pos = np.array([gr_l_w[0], gr_l_w[1], self.bag_lift_height])
        lift_r_pos = np.array([gr_r_w[0], gr_r_w[1], self.bag_lift_height])
        lift_quat_l = self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.left_base)[:3], lift_l_pos)
        lift_quat_r = self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.right_base)[:3], lift_r_pos)
        lift_l_w = np.concatenate([lift_l_pos, lift_quat_l])
        lift_r_w = np.concatenate([lift_r_pos, lift_quat_r])

        l_q_pre,  r_q_pre  = self._solve_ik_pair(pre_l_w,  pre_r_w,  _LEFT_PRE_SEED,   _RIGHT_PRE_SEED,   tag="PRE")
        l_q_gr,   r_q_gr   = self._solve_ik_pair(gr_l_w,   gr_r_w,   _LEFT_GRASP_SEED, _RIGHT_GRASP_SEED, tag="GRASP")
        l_q_lift, r_q_lift = self._solve_ik_pair(lift_l_w, lift_r_w, l_q_gr,           r_q_gr,            tag="LIFT")
        l_q_place, r_q_place = _LEFT_PLACE_JOINTS.copy(), _RIGHT_PLACE_JOINTS.copy()

        self.task_meta = {
            "episode_id": episode_id,
            "snickers_drop": self._snickers_drop_pose,
            "mouth_xyz": [self._mouth_xy[0], self._mouth_xy[1], self._mouth_z],
            "bag_pos": bag_xy0.tolist(),
            "bag_y_ref": float(y_ref),
            "bag_pre_world_left":   pre_l_w.tolist(),
            "bag_pre_world_right":  pre_r_w.tolist(),
            "bag_grasp_world_left":  gr_l_w.tolist(),
            "bag_grasp_world_right": gr_r_w.tolist(),
            "bag_lift_world_left":   lift_l_w.tolist(),
            "bag_lift_world_right":  lift_r_w.tolist(),
            "description": "v2 snickers pack + bimanual bag-strap grasp+lift+place",
        }

        # ---- B1: PRE ----
        with self.recording("bag_move_to_pre_grasp"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=l_q_pre, right_target_pose=r_q_pre,
                goal_type="joint",
                left_gripper=self.bag_open_gripper, right_gripper=self.bag_open_gripper,
                velocity_scale=self.bag_home_to_pre_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_pre_settle, description="bag_move_to_pre_grasp")

        # ---- B2: GRASP ----
        with self.recording("bag_move_to_grasp"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=l_q_gr, right_target_pose=r_q_gr,
                goal_type="joint",
                left_gripper=self.bag_open_gripper, right_gripper=self.bag_open_gripper,
                velocity_scale=self.bag_pre_to_grasp_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_grasp_settle, description="bag_move_to_grasp")

        # ---- B3: close ----
        with self.recording("bag_close_grippers"):
            plan = self.skills.gripper_bimanual_action(
                left_target_val=self.bag_close_gripper, right_target_val=self.bag_close_gripper,
                duration=self.bag_close_duration,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_post_close_settle, description="bag_close_grippers")

        # ---- B4: LIFT (vertical) ----
        # Kinematically attach snickers to the closest cloth vertex so it
        # rides with the bag during lift/place. Style3D pins rigid bodies,
        # so without this the bag passes around snickers and leaves it on
        # the table. Detached before bag_release in B6.
        if hasattr(self.env, "attach_rigid_to_cloth"):
            try:
                self.env.attach_rigid_to_cloth("snickers_body")
            except Exception as e:
                logger.warning(f"[v2] attach_rigid_to_cloth failed: {e}")
        with self.recording("bag_lift"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=l_q_lift, right_target_pose=r_q_lift,
                goal_type="joint",
                left_gripper=self.bag_close_gripper, right_gripper=self.bag_close_gripper,
                velocity_scale=self.bag_lift_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_lift_settle, description="bag_lift")

        self._snapshot_strap_z()
        if self._lift_snapshot is not None:
            self.task_meta["lift_snapshot"] = dict(self._lift_snapshot)

        # Early exit: success is judged at LIFT (verdict frozen here). If the
        # snapshot doesn't pass, no point running place / release — bail out.
        if not self.check_success():
            logger.warning("[v2] LIFT-time check failed — skipping place/release/return_home")
            if hasattr(self.env, "detach_rigid_from_cloth"):
                self.env.detach_rigid_from_cloth("snickers_body")
            return

        # ---- B5: PLACE (joint replay) ----
        with self.recording("bag_place"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=l_q_place, right_target_pose=r_q_place,
                goal_type="joint",
                left_gripper=self.bag_close_gripper, right_gripper=self.bag_close_gripper,
                velocity_scale=self.bag_lift_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_lift_settle, description="bag_place")

        # ---- B6: release + return home ----
        # Detach snickers from cloth before opening the bag so it falls into
        # the placed bag under physics, not glued to the now-released cloth.
        if hasattr(self.env, "detach_rigid_from_cloth"):
            self.env.detach_rigid_from_cloth("snickers_body")
        with self.recording("bag_release"):
            plan = self.skills.gripper_bimanual_action(
                left_target_val=self.bag_open_gripper, right_target_val=self.bag_open_gripper,
                duration=self.bag_close_duration,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_post_close_settle, description="bag_release")

        with self.recording("bag_return_home"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.move_dual_arm_to(
                left_target_pose=home_l, right_target_pose=home_r,
                goal_type="joint",
                left_gripper=self.bag_open_gripper, right_gripper=self.bag_open_gripper,
                velocity_scale=self.bag_return_home_velocity_scale,
            )
            yield ExecutionStep(plan=plan, settle_time=self.bag_return_home_settle, description="bag_return_home")

        logger.success("[v2] Done.")

    # ----------------------------------------------------------------------
    # check_success
    # ----------------------------------------------------------------------
    def check_success(self) -> bool:
        """Verdict is *frozen at LIFT moment* (not at end of episode):
        (1) Snickers (LIFT-snapshot xy/z) landed inside bag mouth
            (LIFT-snapshot mouth_xy + mouth_z).
        (2) Both straps' tip + mid mean z > bag_strap_lift_thresh.
        (3) Bag centroid z rose by > bag_lift_thresh vs pre-grasp baseline.
        Using the snapshot avoids losing a successful grasp just because
        Stage B5 PLACE moves the bag away from its initial mouth position.
        """
        snap = self._lift_snapshot
        if snap is None:
            logger.warning("[v2 check] no LIFT snapshot — TOTAL=False")
            return False

        # (1) snickers vs mouth (both at LIFT moment)
        sh = snap.get("snickers_xpos")
        m_xy = snap.get("mouth_xy")
        m_z = snap.get("mouth_z")
        if sh is None or m_xy is None or m_z is None:
            logger.warning("[v2 check] missing snickers/mouth in snapshot")
            snickers_ok = False
            dxy = float("nan"); below = False
        else:
            dxy = float(np.linalg.norm(np.asarray(sh[:2]) - np.asarray(m_xy)))
            below = bool(sh[2] < m_z - self.snickers_success_z_below_mouth)
            snickers_ok = bool(dxy < self.snickers_success_xy_radius and below)

        # (2) strap z
        s_thr = self.bag_strap_lift_thresh
        L_ok = (snap["left_strap_tip_z"]  > s_thr) and (snap["left_strap_mid_z"]  > s_thr)
        R_ok = (snap["right_strap_tip_z"] > s_thr) and (snap["right_strap_mid_z"] > s_thr)

        # (3) bag centroid Δz
        z0 = snap.get("bag_centroid_z_initial")
        z1 = snap.get("bag_centroid_z")
        if z0 is None or z1 is None:
            bag_ok = False
            bag_delta = float("nan")
        else:
            bag_delta = z1 - z0
            bag_ok = bool(L_ok and R_ok and (bag_delta > self.bag_lift_thresh))

        result = bool(snickers_ok and bag_ok)
        logger.info(
            f"[v2 check_success @LIFT] snickers: dxy={dxy:.3f} "
            f"(thr={self.snickers_success_xy_radius:.3f}), below_mouth={below} → {snickers_ok}  |  "
            f"bag: strap_thr={s_thr:.3f} L={L_ok} R={R_ok}, "
            f"bag_Δz={bag_delta:+.3f} (thr={self.bag_lift_thresh:.3f}) → {bag_ok}  |  TOTAL={result}"
        )
        return result
