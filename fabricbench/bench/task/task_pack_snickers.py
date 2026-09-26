"""task/task_pack_snickers.py

Long-horizon composite: pack a Snickers bar into the bag mouth (Stage A) then
grasp the bag's two straps and lift/place it (Stage B). Ported 1:1 from
deformable_bench ``task_pack_snicker_v2.py``; self-contained (does NOT import
the atomic put_snickers / grasp_bag_strap tasks, mirroring deformable_bench).

Pipeline:
  A1 hold home + cloth/snickers settle, derive bag mouth
  A2 single-arm move to snickers grasp pose
  A3 close gripper on snickers
  A4 lift + waypoint above bag mouth
  A5 descend into bag
  A6 release snickers
  B0 dual-arm move to lift_silk preset
  B1 PRE  (IK, v5 seed)
  B2 GRASP (IK, v5 seed)
  B3 close grippers on straps
  B4 LIFT (joint replay from IK) + strap/bag z snapshot for check_success
  B5 PLACE (joint replay, hardcoded _*_PLACE_JOINTS)
  B6 release + return to lift_silk preset

Success (both required): snickers landed inside bag mouth (A) AND both straps'
tip+mid mean z cleared the lift threshold with the bag body actually rising (B).

control_mode: pd — Stage A needs real contact forces on the free rigid bar.
Stage B grasps the (cloth) strap under the same PD co-sim.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Generator, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from fabricbench.bench.common import checks
from fabricbench.bench.common.types import ArmSide, ExecutionStep
from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.base_task import BaseTask
from fabricbench.bench.task.randomization import _ROBOT_INIT_PRESETS
# metric only (not task logic — the task itself stays self-contained):
# one source of truth for "strap actually pinched, not hooked" across
# collect (both bag tasks) and closed-loop eval.
from fabricbench.bench.task.task_grasp_bag_strap import strap_pinch_metrics


def _mirror_quat_y(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], q[2], -q[3]], dtype=np.float64)


# --- Bag-strap (v4) reference constants, verbatim from deformable_bench ------
_LEFT_PRE_SEED   = np.array([+0.3713, +2.0562, -1.5176, -1.3803, +1.2316, +0.8464])
_RIGHT_PRE_SEED  = np.array([-0.3081, +1.9306, -1.5293, +1.2006, +1.2280, -0.1168])
_LEFT_GRASP_SEED = np.array([+0.0986, +2.1238, -1.5176, -1.5683, +1.1509, +0.8464])
_RIGHT_GRASP_SEED = np.array([+0.0632, +1.9825, -1.5248, +1.2731, +1.0008, -0.1939])
_PRE_QUAT_LEFT   = np.array([+0.3006, -0.0571, +0.8060, -0.5067])
_GRASP_QUAT_LEFT = np.array([+0.2554, -0.0391, +0.7750, -0.5768])
_LEFT_PLACE_JOINTS  = np.array([-0.9789, +2.2190, -2.2725, +0.1054, +0.0253, +0.0604])
_RIGHT_PLACE_JOINTS = np.array([-0.3088, +1.0482, -0.9774, -0.3137, +0.1676, +0.1219])


class PackSnickersTask(BaseTask):
    """Snickers pick-and-place + bimanual bag-strap grasp/lift/place."""

    def __init__(self, env, planner_manager: PlannerManager, *,
                 cfg: Optional[DictConfig] = None, collector=None,
                 urdf_path: Optional[str] = None):
        super().__init__(env, planner_manager, collector, cfg)
        tp = dict(cfg.get("task_params", {}) or {}) if (cfg is not None and hasattr(cfg, "get")) else {}

        def f(k, d):
            return float(tp.get(k, d))

        # ---- Stage A: snickers ----
        self.snickers_arm_str = str(tp.get("snickers_arm", "right")).lower().strip()
        self.snickers_arm_side = ArmSide.LEFT if self.snickers_arm_str == "left" else ArmSide.RIGHT
        self.snickers_body_name = str(tp.get("snickers_body_name", "snickers_body"))
        if self.snickers_body_name not in getattr(env, "rigid_body_index", {}):
            raise RuntimeError(
                f"rigid body {self.snickers_body_name!r} not in scene — declare it under "
                f"the task yaml's rigid_objects: block (have: {list(getattr(env, 'rigid_body_index', {}))})")
        self.snickers_grasp_close_val = f("snickers_grasp_close_val", 0.0)
        self.snickers_grasp_height_offset = f("snickers_grasp_height_offset", -0.018)
        self.snickers_grasp_z = f("snickers_grasp_z", 0.005)
        self.snickers_lift_z = f("snickers_lift_z", 0.35)
        self.snickers_above_mouth_offset = f("snickers_above_mouth_offset", 0.10)
        self.snickers_descend_offset = f("snickers_descend_offset", 0.05)
        self.snickers_mouth_dx = f("snickers_mouth_dx", 0.0)
        self.snickers_mouth_dy = f("snickers_mouth_dy", 0.0)
        self.snickers_release_val = f("snickers_release_val", 1.0)
        self.snickers_move_to_grasp_velocity_scale = f("snickers_move_to_grasp_velocity_scale", 0.15)
        self.snickers_close_duration = f("snickers_close_duration", 0.4)
        self.snickers_close_settle_time = f("snickers_close_settle_time", 0.1)
        _trans = tp.get("snickers_transport_velocity_scale", (0.2, 0.1))
        self.snickers_transport_velocity_scale = [float(v) for v in _trans]
        self.snickers_transport_settle_time = f("snickers_transport_settle_time", 0.3)
        self.snickers_descend_velocity_scale = f("snickers_descend_velocity_scale", 0.4)
        self.snickers_descend_settle_time = f("snickers_descend_settle_time", 0.3)
        self.snickers_release_duration = f("snickers_release_duration", 0.6)
        self.snickers_release_settle_time = f("snickers_release_settle_time", 0.0)
        self.snickers_success_xy_radius = f("snickers_success_xy_radius", 0.12)
        self.snickers_success_z_below_mouth = f("snickers_success_z_below_mouth", 0.02)

        # ---- Stage B: bag-strap ----
        self.bag_pre_offset = np.asarray(
            [float(v) for v in tp.get("bag_pre_offset_xyz", (0.080, 0.278, 0.205))], dtype=np.float64)
        self.bag_grasp_offset = np.asarray(
            [float(v) for v in tp.get("bag_grasp_offset_xyz", (0.095, 0.120, 0.198))], dtype=np.float64)
        self.bag_pre_approach_dz = f("bag_pre_approach_dz", 0.06)
        self.bag_lift_height = f("bag_lift_height", 0.45)
        self.bag_left_dy = f("bag_left_dy", 0.0);  self.bag_right_dy = f("bag_right_dy", 0.0)
        self.bag_left_dz = f("bag_left_dz", 0.0);  self.bag_right_dz = f("bag_right_dz", 0.0)
        self.bag_open = f("bag_open_gripper_value", 1.0)
        self.bag_close = f("bag_close_gripper_value", 0.01)
        self.bag_dual_to_home_velocity_scale = f("bag_dual_to_home_velocity_scale", 0.40)
        self.bag_home_to_pre_velocity_scale = f("bag_home_to_pre_velocity_scale", 0.40)
        self.bag_pre_to_grasp_velocity_scale = f("bag_pre_to_grasp_velocity_scale", 0.15)
        self.bag_lift_velocity_scale = f("bag_lift_velocity_scale", 0.20)
        self.bag_return_home_velocity_scale = f("bag_return_home_velocity_scale", 0.80)
        self.bag_pre_settle = f("bag_pre_settle", 0.0)
        self.bag_grasp_settle = f("bag_grasp_settle", 0.8)
        self.bag_close_duration = f("bag_close_duration", 1.0)
        self.bag_post_close_settle = f("bag_post_close_settle", 0.5)
        self.bag_lift_settle = f("bag_lift_settle", 1.0)
        self.bag_return_home_settle = f("bag_return_home_settle", 0.0)
        self.bag_strap_lift_thresh = f("bag_strap_lift_thresh", 0.35)
        self.bag_lift_thresh = f("bag_lift_thresh", 0.05)
        self.strap_grasp_max_dist = f("strap_grasp_max_dist", 0.02)
        # B5 carry: translate BOTH grippers by place_dy keeping the B4 grip
        # geometry (separation/quats), then lower to place_z and release.
        # 0 disables -> legacy hardcoded PLACE_JOINTS (which narrowed the
        # gripper separation 0.31->0.17 with a 37/23cm asymmetric shift, so
        # the flexible straps re-seated and the bag followed only ~56%).
        self.bag_place_dy = f("bag_place_dy", -0.15)
        self.bag_place_z = f("bag_place_z", 0.34)
        self.bag_place_hover = f("bag_place_hover", 0.04)
        # B2.5 measure-and-correct (m). Default 0 = OFF (user 2026-07-15:
        # corrective moves rejected — demos must be one natural motion, failed
        # episodes are simply dropped at collection). Static bag_*_dy/dz absorb
        # the systematic aim bias; the residual ±2cm approach-disturbance term
        # is run-to-run noise (same seed reruns differ by 2cm) and costs ~1/3
        # of episodes. Set cap >0 (e.g. 0.06) to re-enable trigger-gated
        # correction: only an arm whose arrival miss > trigger gets corrected
        # (measured: 6 arms -> 1 correction, strict-judge 3/3).
        self.bag_grasp_correct_max = f("bag_grasp_correct_max", 0.0)
        self.bag_grasp_correct_trigger = f("bag_grasp_correct_trigger", 0.02)
        # Plan-time aim snap cap (m), default OFF: snapping the target onto the
        # measured strap changes the whole approach sweep and can knock the
        # strap away (ep0 R: 0.002 static -> 0.044 snapped). Kept for A/B.
        self.bag_aim_snap_max = f("bag_aim_snap_max", 0.0)

        self._mouth_xy: Optional[Tuple[float, float]] = None
        self._mouth_z: Optional[float] = None
        self._snickers_ok = False
        self._lift_snapshot: Optional[dict] = None
        self._hold_snapshot: Optional[dict] = None
        self._initial_bag_centroid_z: Optional[float] = None
        self._strap_idx = self._load_strap_topology()

        if urdf_path is None:
            urdf_path = env.robot.urdf
        self._kin = PinocchioKinematics(urdf_path, base_link="base_link", ee_link="tcp_link_center")
        logger.info(f"[pack] snickers arm={self.snickers_arm_str}  strap_thr={self.bag_strap_lift_thresh:.3f}")

    # ======================================================================
    # Topology / mouth / snickers helpers (Stage A)
    # ======================================================================
    def _load_strap_topology(self) -> Optional[dict]:
        cloth_dir = os.path.dirname(str(self.env.cloth_obj_path))
        cands = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
        if not cands:
            logger.warning(f"[pack] no *_strap_topology.json in {cloth_dir}")
            return None
        obj_stem = os.path.splitext(os.path.basename(str(self.env.cloth_obj_path)))[0]
        cands.sort(key=lambda p: -len(os.path.commonprefix([os.path.basename(p), obj_stem])))
        with open(cands[0]) as fh:
            topo = json.load(fh)
        logger.info(f"[pack] using strap topology: {cands[0]}")
        out = {"body": np.asarray(topo["body_indices"], dtype=np.int64)} if "body_indices" in topo else {}
        for side in ("left_strap", "right_strap"):
            s = topo[side]
            out[side] = {"tip": np.asarray(s["tip_indices"], dtype=np.int64),
                         "mid": np.asarray(s["mid_indices"], dtype=np.int64)}
        return out

    def _derive_mouth_from_cloth(self) -> Tuple[Tuple[float, float], float]:
        world = np.asarray(self.env.get_cloth_position(), dtype=np.float64)
        if self._strap_idx is not None and "body" in self._strap_idx:
            sample = world[self._strap_idx["body"]]
        else:
            sample = world
        z_max = float(sample[:, 2].max())
        z_thr = float(np.quantile(sample[:, 2], 0.90))
        mask = sample[:, 2] >= z_thr
        top_xy = sample[mask, :2] if int(mask.sum()) >= 32 else sample[np.argsort(sample[:, 2])[-32:], :2]
        cx = float(top_xy[:, 0].mean()) + self.snickers_mouth_dx
        cy = float(top_xy[:, 1].mean()) + self.snickers_mouth_dy
        return (cx, cy), z_max

    def _snickers_xpos(self) -> np.ndarray:
        return self.env.get_rigid_body_pose(self.snickers_body_name)[:3]

    def _arm_base(self) -> np.ndarray:
        base = self.env.left_base if self.snickers_arm_side == ArmSide.LEFT else self.env.right_base
        return np.asarray(base, dtype=np.float64)[:3]

    @staticmethod
    def _pose(pos, quat) -> np.ndarray:
        return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])

    # ======================================================================
    # Strap helpers (Stage B)
    # ======================================================================
    def _bag_centroid_z(self) -> Optional[float]:
        c = self.env.get_cloth_position()
        return None if c is None or len(c) == 0 else float(np.mean(np.asarray(c)[:, 2]))

    def _cloth_centroid_y(self) -> float:
        c = self.env.get_cloth_position()
        return float(self.env.cloth_base_pos[1]) if c is None or len(c) == 0 else float(np.mean(np.asarray(c)[:, 1]))

    def _snapshot_strap_z(self) -> None:
        if self._strap_idx is None:
            self._lift_snapshot = None
            return
        c = self.env.get_cloth_position()
        if c is None or len(c) == 0:
            self._lift_snapshot = None
            return
        snap = {}
        for side in ("left_strap", "right_strap"):
            snap[f"{side}_tip_z"] = float(np.mean(c[self._strap_idx[side]["tip"], 2]))
            snap[f"{side}_mid_z"] = float(np.mean(c[self._strap_idx[side]["mid"], 2]))
            arm = side.split("_")[0]
            snap[f"{side}_pinch"] = strap_pinch_metrics(self.env, c, self._strap_idx[side], arm)
        snap["bag_centroid_z"] = float(np.mean(np.asarray(c)[:, 2]))
        snap["bag_centroid_z_initial"] = (float(self._initial_bag_centroid_z)
                                          if self._initial_bag_centroid_z is not None else None)
        self._lift_snapshot = snap
        d = (f"Δ={snap['bag_centroid_z'] - snap['bag_centroid_z_initial']:+.3f}"
             if snap["bag_centroid_z_initial"] is not None else "Δ=n/a")
        logger.info(
            f"[pack] LIFT snapshot: L tip={snap['left_strap_tip_z']:.3f} mid={snap['left_strap_mid_z']:.3f} "
            f"R tip={snap['right_strap_tip_z']:.3f} mid={snap['right_strap_mid_z']:.3f} "
            f"strap_thr={self.bag_strap_lift_thresh:.3f} bag_z={snap['bag_centroid_z']:.3f} ({d})")
        for side in ("left_strap", "right_strap"):
            pm = snap[f"{side}_pinch"]
            logger.info(
                f"[pack] pinch[{side}]: min_dist={pm['min_dist']:.4f} t={pm['t']:+.4f} "
                f"rho={pm['rho']:.4f} gap={pm['gap']:.3f} finger_sep={pm['finger_sep']:.4f} "
                f"(max_dist={self.strap_grasp_max_dist:.3f})")

    def _snapshot_hold(self) -> None:
        """HOLD checkpoint: taken at the END of the B5 carry (``B5_place_bag``),
        i.e. after the bag has been carried AWAY from its initial mouth position
        and is hovering above the place point, but BEFORE ``B5b_place_down``
        lowers it. Answers the one question the LIFT snapshot cannot: did the
        grasp survive the carry, or did the bag fall out mid-way?

        Records the SAME instrument the LIFT pinch test uses
        (``strap_pinch_metrics.min_dist`` = TCP pinch centre -> nearest strap
        vertex). That quantity is HAND-RELATIVE: carrying the bag moves strap
        and TCP together and leaves it unchanged, so this cannot re-introduce
        the world-frame false negative the LIFT snapshot was built to avoid
        (snickers vs the bag's *initial* mouth xy). It only grows when the
        strap leaves the fingers.
        """
        if self._strap_idx is None:
            self._hold_snapshot = None
            return
        c = self.env.get_cloth_position()
        if c is None or len(c) == 0:
            self._hold_snapshot = None
            return
        snap = {}
        for side in ("left_strap", "right_strap"):
            snap[f"{side}_tip_z"] = float(np.mean(c[self._strap_idx[side]["tip"], 2]))
            snap[f"{side}_mid_z"] = float(np.mean(c[self._strap_idx[side]["mid"], 2]))
            snap[f"{side}_pinch"] = strap_pinch_metrics(self.env, c, self._strap_idx[side],
                                                        side.split("_")[0])
        snap["bag_centroid_z"] = float(np.mean(np.asarray(c)[:, 2]))
        snap["bag_centroid_xy"] = (float(np.mean(np.asarray(c)[:, 0])),
                                   float(np.mean(np.asarray(c)[:, 1])))
        self._hold_snapshot = snap
        logger.info(
            f"[pack] HOLD snapshot (end of B5 carry): "
            f"L pinch={snap['left_strap_pinch']['min_dist']:.4f} "
            f"R pinch={snap['right_strap_pinch']['min_dist']:.4f} "
            f"(max_dist={self.strap_grasp_max_dist:.3f})  "
            f"bag_z={snap['bag_centroid_z']:.3f}")

    def _resolve_home_joints(self) -> Tuple[np.ndarray, np.ndarray]:
        rand = self.cfg.get("randomization", {}) if hasattr(self.cfg, "get") else {}
        name = (rand.get("robot_init", {}) or {}).get("start_pose")
        preset = _ROBOT_INIT_PRESETS.get(name) if name else None
        if preset is not None:
            return preset["home_left"].copy(), preset["home_right"].copy()
        return np.asarray(self.env.home_q_left).copy(), np.asarray(self.env.home_q_right).copy()

    def _build_world_targets(self, offset, quat_left, l_dy=0.0, r_dy=0.0, l_dz=0.0, r_dz=0.0, y_ref=None):
        """deformable_bench's exact symbolic planning: x/z ref = bag OBJ-origin
        (= env.cloth_base_pos with the bag placed origin-anchored, matching
        deformable's base_pos); y ref = cloth centroid; |Δy| mirrored L/R.

        The planar offsets are rotated by the episode's randomized bag yaw
        (cfg.cloth_rot_z): the straps rotate with the bag, and at strap radius
        the ±5° init randomization swings them ±1.3 cm — larger than the
        ~1.8 cm firm-pinch window, so an unrotated formula cannot hit both
        arms across episodes (measured 2026-07-16: left residual spread 3.1 cm,
        100% flung when the target overshoots outward). y_ref (live centroid)
        still absorbs translation + residual drift."""
        bag = self.env.cloth_base_pos
        if y_ref is None:
            y_ref = self._cloth_centroid_y()
        theta = np.deg2rad(float(getattr(self.env.cfg, "cloth_rot_z", 0.0)))
        c, s = np.cos(theta), np.sin(theta)

        def rot(ox, oy):
            return c * ox - s * oy, s * ox + c * oy

        lx, ly = rot(offset[0], +abs(offset[1]) + l_dy)
        rx, ry = rot(offset[0], -abs(offset[1]) + r_dy)
        l_pos = np.array([bag[0] + lx, y_ref + ly, bag[2] + offset[2] + l_dz])
        r_pos = np.array([bag[0] + rx, y_ref + ry, bag[2] + offset[2] + r_dz])
        return self._pose(l_pos, quat_left), self._pose(r_pos, _mirror_quat_y(quat_left))

    def _solve_ik_pair(self, lw, rw, ls, rs, tag=""):
        lb = np.asarray(self.env.left_base[:3]);  rb = np.asarray(self.env.right_base[:3])
        lt = np.concatenate([lw[:3] - lb, lw[3:]]);  rt = np.concatenate([rw[:3] - rb, rw[3:]])
        kw = dict(pose_constraint=[1, 1, 1, 1, 0, 1], constraint_frame="local", threshold=5e-3, max_restarts=30)
        lq = self._kin.solve_ik(lt, seed_joints=ls, **kw)
        rq = self._kin.solve_ik(rt, seed_joints=rs, **kw)
        if lq is None:
            raise RuntimeError(f"[pack] LEFT IK failed at {tag} (target={lt[:3]})")
        if rq is None:
            raise RuntimeError(f"[pack] RIGHT IK failed at {tag} (target={rt[:3]})")
        return lq[:6], rq[:6]

    @staticmethod
    def _branch_ok(lq: np.ndarray, rq: np.ndarray) -> Tuple[bool, bool]:
        """Wrist-branch health of a free-roll IK solution pair: the healthy
        branch sits at j4 ≈ −1.4 (left) / +1.4 (right), the mirrored one at
        ≈ ±0.7 with opposite sign. ±0.5 splits them cleanly."""
        return bool(lq[3] < -0.5), bool(rq[3] > 0.5)

    def _solve_pre_branch_guarded(self, y_ref: float) -> Tuple[np.ndarray, np.ndarray]:
        """PRE IK with a wrist-branch guard (adaptive lateral pull-in).

        The free-roll IK ([1,1,1,1,0,1]) has two mirror wrist branches that
        merge near a workspace boundary (local_docs/pack-snickers-ik-singularity.md).
        When bag lean (y_ref drift) pushes a PRE target past that boundary the
        healthy branch has no solution, and DLS lands on the flipped one — the
        arm twists over and the approach is ruined. PRE is an empty-air
        approach waypoint, so the offending arm's target may be pulled
        laterally inward (toward the bag) in 1 cm steps until the healthy
        branch returns. Keeps the widest feasible PRE (wide = better grasp,
        §6) instead of a fixed shrunken offset; bag placement and GRASP
        targets are untouched.
        """
        pull_l = pull_r = 0.0
        lq = rq = pre_r = None
        for _ in range(5):
            pre_l, pre_r = self._build_world_targets(
                self.bag_pre_offset, _PRE_QUAT_LEFT, l_dy=-pull_l, r_dy=+pull_r, y_ref=y_ref)
            lq, rq = self._solve_ik_pair(pre_l, pre_r, _LEFT_PRE_SEED, _RIGHT_PRE_SEED, tag="PRE")
            l_ok, r_ok = self._branch_ok(lq, rq)
            if l_ok and r_ok:
                break
            if not l_ok:
                pull_l += 0.01
            if not r_ok:
                pull_r += 0.01
        rby = float(self.env.right_base[1])
        logger.info(f"[pack] PRE branch: y_ref={y_ref:+.4f} pull(L,R)=({pull_l:.3f},{pull_r:.3f}) "
                    f"tgt_r_base_y={float(pre_r[1]) - rby:+.4f} L_j4={float(lq[3]):+.3f} R_j4={float(rq[3]):+.3f}")
        l_ok, r_ok = self._branch_ok(lq, rq)
        if not (l_ok and r_ok):
            logger.warning(f"[pack] PRE wrist STILL flipped after max pull-in "
                           f"(L_j4={float(lq[3]):+.3f} R_j4={float(rq[3]):+.3f}) — approach may twist")
        return lq, rq

    # ======================================================================
    # plan_episode
    # ======================================================================
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        # ================= Stage A: snickers pick & drop =================
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        st = self.skills.planning_state
        plan_hold = self.skills.move_dual_arm_to(
            left_target_pose=np.asarray(st.left_joints).copy(),
            right_target_pose=np.asarray(st.right_joints).copy(), goal_type="joint",
            left_gripper=st.left_gripper, right_gripper=st.right_gripper)
        yield ExecutionStep(plan=plan_hold, settle_time=0.5, description="A1_hold_home")

        self._mouth_xy, self._mouth_z = self._derive_mouth_from_cloth()
        logger.info(f"[pack] mouth xy={self._mouth_xy} z={self._mouth_z:.4f}")

        sh_pos = self._snickers_xpos()
        sh_top_z = sh_pos[2] + 0.0269
        grasp_pos = np.array([sh_pos[0], sh_pos[1], sh_top_z + self.snickers_grasp_height_offset])
        base = self._arm_base()
        grasp_quat = self.skills.calculate_base_to_target_horizontal_quat(base, grasp_pos)
        grasp_pose = self._pose(grasp_pos, grasp_quat);  grasp_pose[2] = self.snickers_grasp_z
        lift_pose = self._pose(grasp_pos, grasp_quat);   lift_pose[2] = max(self.snickers_lift_z, self._mouth_z + 0.13)
        above_pos = np.array([self._mouth_xy[0], self._mouth_xy[1], self._mouth_z + self.snickers_above_mouth_offset])
        above_quat = self.skills.calculate_base_to_target_horizontal_quat(base, above_pos)
        above_pose = self._pose(above_pos, above_quat)
        descend_pos = np.array([self._mouth_xy[0], self._mouth_xy[1], self._mouth_z + self.snickers_descend_offset])
        descend_pose = self._pose(descend_pos, above_quat)
        logger.info(f"[pack] A-snickers sh_pos={sh_pos.round(3)} grasp_pose={grasp_pose[:3].round(3)} "
                    f"lift={lift_pose[:3].round(3)} above={above_pose[:3].round(3)} descend={descend_pose[:3].round(3)}")

        with self.recording("A2_snickers_move_to_grasp"):
            plan = self.skills.move_to(
                target_pose=grasp_pose, arm=self.snickers_arm_side, gripper=1.0, ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 1, 0, 1], pose_constraint_frame="local",
                velocity_scale=self.snickers_move_to_grasp_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=0.5, description="A2_snickers_move_to_grasp")

        with self.recording("A3_snickers_close"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.gripper_action(target_val=self.snickers_grasp_close_val,
                                              arm=self.snickers_arm_side, duration=self.snickers_close_duration)
            yield ExecutionStep(plan=plan, settle_time=self.snickers_close_settle_time, description="A3_snickers_close")

        with self.recording("A4_snickers_lift_to_above_mouth"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            if self.snickers_arm_side == ArmSide.LEFT:
                self.skills.planning_state.left_gripper = self.snickers_grasp_close_val
            else:
                self.skills.planning_state.right_gripper = self.snickers_grasp_close_val
            plan = self.skills.move_to(
                target_pose=[lift_pose, above_pose], arm=self.snickers_arm_side, goal_type="waypoints",
                gripper=self.snickers_grasp_close_val, ee_link="tcp_link_bottom",
                pose_constraint=[1, 1, 1, 0, 0, 0], pose_constraint_frame="local",
                interpolation_mode=["joint", "joint"], velocity_scale=self.snickers_transport_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.snickers_transport_settle_time,
                                description="A4_snickers_lift_to_above_mouth")

        with self.recording("A5_snickers_descend"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            if self.snickers_arm_side == ArmSide.LEFT:
                self.skills.planning_state.left_gripper = self.snickers_grasp_close_val
            else:
                self.skills.planning_state.right_gripper = self.snickers_grasp_close_val
            plan = self.skills.move_to(
                target_pose=descend_pose, arm=self.snickers_arm_side, gripper=self.snickers_grasp_close_val,
                ee_link="tcp_link_bottom", pose_constraint=[1, 1, 1, 0, 0, 0], pose_constraint_frame="local",
                velocity_scale=self.snickers_descend_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.snickers_descend_settle_time, description="A5_snickers_descend")

        with self.recording("A6_snickers_release"):
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            plan = self.skills.gripper_action(target_val=self.snickers_release_val,
                                              arm=self.snickers_arm_side, duration=self.snickers_release_duration)
            yield ExecutionStep(plan=plan, settle_time=self.snickers_release_settle_time, description="A6_snickers_release")

        # snapshot snickers-in-mouth result (Stage A success is frozen here,
        # before Stage B disturbs the bag)
        self._snickers_ok = self._check_snickers_in_mouth()

        # ================= Stage B: bag-strap grasp & lift =================
        # B0: dual-arm back to the un-randomized lift_silk preset
        home_l, home_r = self._resolve_home_joints()
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("B0_dual_to_home"):
            plan_b0 = self.skills.move_dual_arm_to(
                left_target_pose=home_l, right_target_pose=home_r, goal_type="joint",
                left_gripper=self.bag_open, right_gripper=self.bag_open,
                velocity_scale=self.bag_dual_to_home_velocity_scale)
            yield ExecutionStep(plan=plan_b0, settle_time=0.5, description="B0_dual_to_home")

        # baseline bag centroid AFTER B0 settle, before strap grasp touches it
        self._initial_bag_centroid_z = self._bag_centroid_z()
        if self._initial_bag_centroid_z is not None:
            logger.info(f"[pack] initial bag centroid z (B0 baseline) = {self._initial_bag_centroid_z:.3f}")

        y_ref = self._cloth_centroid_y()
        gr_l, gr_r = self._build_world_targets(
            self.bag_grasp_offset, _GRASP_QUAT_LEFT, l_dy=self.bag_left_dy, r_dy=self.bag_right_dy,
            l_dz=self.bag_left_dz, r_dz=self.bag_right_dz, y_ref=y_ref)

        # Plan-time aim snap: pull each grasp target onto the nearest measured
        # strap vertex (symbolic target + bag_*_dy/dz acts as the prior that
        # picks the right part of the loop). The bag is settled and untouched
        # during B1/B2, so measuring here is as good as measuring on arrival —
        # but the approach stays ONE natural motion (no corrective re-aim,
        # user 2026-07-15). Residual after static offsets alone is ±1-2cm,
        # which straddles the firm-pinch margin; the snap removes it.
        c = self.env.get_cloth_position()
        if (self.bag_aim_snap_max > 0 and self._strap_idx is not None
                and c is not None and len(c)):
            for side, tgt in (("left_strap", gr_l), ("right_strap", gr_r)):
                verts = c[np.concatenate([self._strap_idx[side]["tip"], self._strap_idx[side]["mid"]])]
                d = np.linalg.norm(verts - tgt[:3], axis=1)
                rel = verts[int(np.argmin(d))] - tgt[:3]
                miss = float(np.linalg.norm(rel))
                if miss > self.bag_aim_snap_max:
                    logger.warning(f"[pack] aim-snap[{side}]: strap {miss:.3f}m from prior "
                                   f"(> cap {self.bag_aim_snap_max:.3f}) — capping")
                    rel = rel * (self.bag_aim_snap_max / miss)
                tgt[:3] += rel
                logger.info(f"[pack] aim-snap[{side}]: prior_miss={miss:.4f} apply={rel.round(4)}")

        lift_l_pos = np.array([gr_l[0], gr_l[1], self.bag_lift_height])
        lift_r_pos = np.array([gr_r[0], gr_r[1], self.bag_lift_height])
        lift_l = np.concatenate([lift_l_pos, self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.left_base)[:3], lift_l_pos)])
        lift_r = np.concatenate([lift_r_pos, self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.right_base)[:3], lift_r_pos)])

        lq_pre, rq_pre = self._solve_pre_branch_guarded(y_ref)
        lq_gr, rq_gr = self._solve_ik_pair(gr_l, gr_r, _LEFT_GRASP_SEED, _RIGHT_GRASP_SEED, tag="GRASP")
        gl_ok, gr_ok = self._branch_ok(lq_gr, rq_gr)
        if not (gl_ok and gr_ok):
            logger.warning(f"[pack] GRASP wrist flipped (L_j4={float(lq_gr[3]):+.3f} "
                           f"R_j4={float(rq_gr[3]):+.3f}) — grasp target is at the real strap, "
                           f"cannot pull in; expect a twisted approach")
        lq_lift, rq_lift = self._solve_ik_pair(lift_l, lift_r, lq_gr, rq_gr, tag="LIFT")
        lq_place, rq_place = _LEFT_PLACE_JOINTS.copy(), _RIGHT_PLACE_JOINTS.copy()
        logger.info(f"[pack] bag_base={self.env.cloth_base_pos.round(3)} gr_world_L={gr_l[:3].round(3)}")
        logger.info(f"[pack] B-joint PRE  L={lq_pre.round(3)} R={rq_pre.round(3)}")
        logger.info(f"[pack] B-joint GRSP L={lq_gr.round(3)} R={rq_gr.round(3)}")
        logger.info(f"[pack] B-joint LIFT L={lq_lift.round(3)} R={rq_lift.round(3)}")

        # B1: home -> PRE
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("B1_move_to_pre_grasp"):
            plan = self.skills.move_dual_arm_to(
                left_target_pose=lq_pre, right_target_pose=rq_pre, goal_type="joint",
                left_gripper=self.bag_open, right_gripper=self.bag_open,
                velocity_scale=self.bag_home_to_pre_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.bag_pre_settle, description="B1_move_to_pre_grasp")

        # B2: PRE -> GRASP
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("B2_move_to_grasp"):
            plan = self.skills.move_dual_arm_to(
                left_target_pose=lq_gr, right_target_pose=rq_gr, goal_type="joint",
                left_gripper=self.bag_open, right_gripper=self.bag_open,
                velocity_scale=self.bag_pre_to_grasp_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.bag_grasp_settle, description="B2_move_to_grasp")

        # Arrival-miss diagnostic (always on, read-only): world-frame vector from
        # each TCP to its nearest strap vertex, right after arriving at GRASP and
        # BEFORE closing. dy/dz map directly to grasp target y/z, so averaging
        # this over episodes tells us how much more static offset to fold in
        # (the "aim inward from the start" experiment).
        c0 = self.env.get_cloth_position()
        if self._strap_idx is not None and c0 is not None and len(c0):
            for side, arm in (("left_strap", "left"), ("right_strap", "right")):
                eef = self.env.left_eef_pose if arm == "left" else self.env.right_eef_pose
                verts = c0[np.concatenate([self._strap_idx[side]["tip"], self._strap_idx[side]["mid"]])]
                tcp = np.asarray(eef[:3], dtype=np.float64)
                rel = verts[int(np.argmin(np.linalg.norm(verts - tcp, axis=1)))] - tcp
                logger.info(f"[pack] arrival_miss[{side}]: world_dxyz=({rel[0]:+.4f},{rel[1]:+.4f},"
                            f"{rel[2]:+.4f}) |={float(np.linalg.norm(rel)):.4f}")

        # B2.5: measure-and-correct. The symbolic GRASP target chases the strap
        # with fixed offsets, but per-episode drape (the free bar stirs the
        # right side) leaves the strap up to ~5cm from the pinch point — a
        # shallow hook that slips off in B5. So after arriving, re-aim the TCP
        # at the actually-measured nearest strap vertex and make a small
        # corrective move before closing. Adaptive by construction; no
        # per-episode constant can do this.
        c = self.env.get_cloth_position()
        if (self.bag_grasp_correct_max > 0 and self._strap_idx is not None
                and c is not None and len(c)):
            corr = {}
            n_triggered = 0
            for side, arm in (("left_strap", "left"), ("right_strap", "right")):
                eef = self.env.left_eef_pose if arm == "left" else self.env.right_eef_pose
                verts = c[np.concatenate([self._strap_idx[side]["tip"], self._strap_idx[side]["mid"]])]
                tcp = np.asarray(eef[:3], dtype=np.float64)
                d = np.linalg.norm(verts - tcp, axis=1)
                rel = verts[int(np.argmin(d))] - tcp
                miss = float(np.linalg.norm(rel))
                if miss <= self.bag_grasp_correct_trigger:
                    # close enough to pinch firmly — leave this arm untouched
                    rel = np.zeros(3)
                else:
                    n_triggered += 1
                    if miss > self.bag_grasp_correct_max:
                        rel = rel * (self.bag_grasp_correct_max / miss)
                corr[arm] = (rel, np.asarray(eef[:3], dtype=np.float64))
                logger.info(f"[pack] B2.5 correct[{side}]: miss={miss:.4f} "
                            f"(trigger={self.bag_grasp_correct_trigger:.3f}) apply={rel.round(4)}")
            if n_triggered > 0:
                gl2 = self._pose(corr["left"][1] + corr["left"][0], gr_l[3:])
                gr2 = self._pose(corr["right"][1] + corr["right"][0], gr_r[3:])
                try:
                    lq_gr2, rq_gr2 = self._solve_ik_pair(gl2, gr2, lq_gr, rq_gr, tag="GRASP-CORRECT")
                    c_ok_l, c_ok_r = self._branch_ok(lq_gr2, rq_gr2)
                    if c_ok_l and c_ok_r:
                        lq_gr, rq_gr = lq_gr2, rq_gr2
                        # keep LIFT straight above the corrected grasp
                        lift_l_pos = np.array([gl2[0], gl2[1], self.bag_lift_height])
                        lift_r_pos = np.array([gr2[0], gr2[1], self.bag_lift_height])
                        lift_l = np.concatenate([lift_l_pos, self.skills.calculate_base_to_target_horizontal_quat(
                            np.asarray(self.env.left_base)[:3], lift_l_pos)])
                        lift_r = np.concatenate([lift_r_pos, self.skills.calculate_base_to_target_horizontal_quat(
                            np.asarray(self.env.right_base)[:3], lift_r_pos)])
                        lq_lift, rq_lift = self._solve_ik_pair(lift_l, lift_r, lq_gr, rq_gr, tag="LIFT-CORRECT")
                        self.skills.sync_state(self.env.get_observation(include_visuals=False))
                        with self.recording("B2b_correct_grasp"):
                            plan = self.skills.move_dual_arm_to(
                                left_target_pose=lq_gr, right_target_pose=rq_gr, goal_type="joint",
                                left_gripper=self.bag_open, right_gripper=self.bag_open,
                                velocity_scale=self.bag_pre_to_grasp_velocity_scale)
                            yield ExecutionStep(plan=plan, settle_time=self.bag_grasp_settle,
                                                description="B2b_correct_grasp")
                    else:
                        logger.warning("[pack] B2.5 corrected IK flipped branch — keeping original grasp")
                except RuntimeError as e:
                    logger.warning(f"[pack] B2.5 corrective IK failed — keeping original grasp: {e}")

        # B3: close on straps
        with self.recording("B3_close_grippers"):
            plan = self.skills.gripper_bimanual_action(
                left_target_val=self.bag_close, right_target_val=self.bag_close, duration=self.bag_close_duration)
            yield ExecutionStep(plan=plan, settle_time=self.bag_post_close_settle, description="B3_close_grippers")

        # close-time diagnostic: nearest strap vertex in the gripper LOCAL
        # frame (z = along fingers, distal positive). Firm pinch = small |z|;
        # shallow (slips in B5) = strap out near the pad tips. Drives the
        # bag_*_dy/dz aim correction (to-optimize §8 right-strap flakiness).
        c = self.env.get_cloth_position()
        if self._strap_idx is not None and c is not None and len(c):
            for side, arm in (("left_strap", "left"), ("right_strap", "right")):
                pm = strap_pinch_metrics(self.env, c, self._strap_idx[side], arm)
                eef = self.env.left_eef_pose if arm == "left" else self.env.right_eef_pose
                verts = c[np.concatenate([self._strap_idx[side]["tip"], self._strap_idx[side]["mid"]])]
                tcp = np.asarray(eef[:3], dtype=np.float64)
                rel = verts[int(np.argmin(np.linalg.norm(verts - tcp, axis=1)))] - tcp
                w, x, y, z = [float(v) for v in eef[3:7]]
                R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                              [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                              [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
                loc = R.T @ rel
                logger.info(f"[pack] close[{side}]: min_dist={pm['min_dist']:.4f} "
                            f"local_xyz=({loc[0]:+.4f},{loc[1]:+.4f},{loc[2]:+.4f})")

        # B4: LIFT
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("B4_lift_bag"):
            plan = self.skills.move_dual_arm_to(
                left_target_pose=lq_lift, right_target_pose=rq_lift, goal_type="joint",
                left_gripper=self.bag_close, right_gripper=self.bag_close,
                velocity_scale=self.bag_lift_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.bag_lift_settle, description="B4_lift_bag")

        self._snapshot_strap_z()
        if self._lift_snapshot is not None:
            self.task_meta["lift_snapshot"] = dict(self._lift_snapshot)

        # B5: PLACE — carry sideways preserving the B4 grip geometry (both
        # grippers translate together, straps stay taut, bag follows ~1:1),
        # then lower to place. Falls back to the legacy PLACE_JOINTS replay
        # if the carry IK fails or flips a wrist branch.
        placed = False
        if abs(self.bag_place_dy) > 1e-6:
            try:
                # Carry DESCENDS while translating (like the legacy motion):
                # the lateral reach envelope is much wider at lower z — at lift
                # height (0.45) the left arm cannot reach dy=-0.30. But it
                # stops a hover above place height with the bag still airborne:
                # the bag swings pendulum-like behind the fast hands, and if it
                # touches down mid-swing the (now sticky) ground friction pins
                # it 10cm short. Hover + settle lets it swing back under the
                # grippers, then B5b lowers it the last few cm. The retry loop
                # shrinks dy on IK failure too, not just on a branch flip.
                dy = self.bag_place_dy
                hover_z = self.bag_place_z + self.bag_place_hover
                lq_carry = rq_carry = None
                for _ in range(4):
                    off = np.array([0.0, dy, 0.0])
                    carry_l = self._pose(np.asarray(lift_l[:3]) + off, lift_l[3:])
                    carry_r = self._pose(np.asarray(lift_r[:3]) + off, lift_r[3:])
                    carry_l[2] = hover_z
                    carry_r[2] = hover_z
                    try:
                        lq_carry, rq_carry = self._solve_ik_pair(
                            carry_l, carry_r, lq_lift, rq_lift, tag="CARRY")
                    except RuntimeError as e:
                        logger.info(f"[pack] B5 carry dy={dy:+.3f}: IK miss ({e})")
                        dy = dy + (0.05 if dy < 0 else -0.05)
                        continue
                    c_l, c_r = self._branch_ok(lq_carry, rq_carry)
                    if c_l and c_r:
                        break
                    logger.info(f"[pack] B5 carry dy={dy:+.3f}: branch flip L={c_l} R={c_r}")
                    dy = dy + (0.05 if dy < 0 else -0.05)
                if lq_carry is None:
                    raise RuntimeError(f"[pack] B5 carry IK failed down to dy={dy:+.3f}")
                logger.info(f"[pack] B5 carry: dy={dy:+.3f} L_j4={float(lq_carry[3]):+.3f} "
                            f"R_j4={float(rq_carry[3]):+.3f}")
                down_l = carry_l.copy(); down_l[2] = self.bag_place_z
                down_r = carry_r.copy(); down_r[2] = self.bag_place_z
                lq_down, rq_down = self._solve_ik_pair(down_l, down_r, lq_carry, rq_carry, tag="PLACE-DOWN")
                self.skills.sync_state(self.env.get_observation(include_visuals=False))
                with self.recording("B5_place_bag"):
                    plan = self.skills.move_dual_arm_to(
                        left_target_pose=lq_carry, right_target_pose=rq_carry, goal_type="joint",
                        left_gripper=self.bag_close, right_gripper=self.bag_close,
                        velocity_scale=self.bag_lift_velocity_scale)
                    yield ExecutionStep(plan=plan, settle_time=0.8, description="B5_place_bag")
                self._snapshot_hold()
                self.skills.sync_state(self.env.get_observation(include_visuals=False))
                with self.recording("B5b_place_down"):
                    plan = self.skills.move_dual_arm_to(
                        left_target_pose=lq_down, right_target_pose=rq_down, goal_type="joint",
                        left_gripper=self.bag_close, right_gripper=self.bag_close,
                        velocity_scale=self.bag_lift_velocity_scale)
                    yield ExecutionStep(plan=plan, settle_time=self.bag_lift_settle,
                                        description="B5b_place_down")
                placed = True
            except RuntimeError as e:
                logger.warning(f"[pack] B5 carry IK failed — falling back to PLACE_JOINTS: {e}")
        if not placed:
            self.skills.sync_state(self.env.get_observation(include_visuals=False))
            with self.recording("B5_place_bag"):
                plan = self.skills.move_dual_arm_to(
                    left_target_pose=lq_place, right_target_pose=rq_place, goal_type="joint",
                    left_gripper=self.bag_close, right_gripper=self.bag_close,
                    velocity_scale=self.bag_lift_velocity_scale)
                yield ExecutionStep(plan=plan, settle_time=self.bag_lift_settle, description="B5_place_bag")
            self._snapshot_hold()

        if self._hold_snapshot is not None:
            self.task_meta["hold_snapshot"] = dict(self._hold_snapshot)

        # B6: release + return home
        with self.recording("B6_release_grippers"):
            plan = self.skills.gripper_bimanual_action(
                left_target_val=self.bag_open, right_target_val=self.bag_open, duration=self.bag_close_duration)
            yield ExecutionStep(plan=plan, settle_time=self.bag_post_close_settle, description="B6_release_grippers")

        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("B6_return_home"):
            plan = self.skills.move_dual_arm_to(
                left_target_pose=home_l, right_target_pose=home_r, goal_type="joint",
                left_gripper=self.bag_open, right_gripper=self.bag_open,
                velocity_scale=self.bag_return_home_velocity_scale)
            yield ExecutionStep(plan=plan, settle_time=self.bag_return_home_settle, description="B6_return_home")

        logger.success("[pack] Done.")

    # ======================================================================
    # success
    # ======================================================================
    def _check_snickers_in_mouth(self) -> bool:
        try:
            sh = self._snickers_xpos()
            mouth_xy, mouth_z = self._derive_mouth_from_cloth()
        except Exception as e:
            logger.warning(f"[pack] snickers-in-mouth check failed: {e}")
            return False
        dxy = float(np.linalg.norm(sh[:2] - np.asarray(mouth_xy)))
        ok = bool(dxy < self.snickers_success_xy_radius and sh[2] < mouth_z - self.snickers_success_z_below_mouth)
        logger.info(f"[pack] snickers-in-mouth: sh=({sh[0]:.3f},{sh[1]:.3f},{sh[2]:.3f}) "
                    f"mouth=({mouth_xy[0]:.3f},{mouth_xy[1]:.3f},{mouth_z:.3f}) dxy={dxy:.3f} → {ok}")
        return ok

    def check_success(self) -> bool:
        """Composite success: snickers landed in mouth (Stage A, frozen at A6)
        AND both straps lifted with the bag body rising (Stage B, at B4)
        AND the grasp was still HELD at the end of the B5 carry (B5 HOLD).

        (1)-(4) are frozen at LIFT on purpose: they judge "did it get picked
        up", and freezing them keeps B5 PLACE (which carries the bag ~0.24 m
        away from its initial mouth position) from failing a good lift.
        (5) HOLD closes the blind spot that freezing opened — "picked up, then
        dropped mid-carry" — WITHOUT re-opening the false negative, because it
        only reads the hand-relative pinch distance, never a world position.
        """
        snap = self._lift_snapshot
        if snap is None:
            logger.warning("[pack check_success] no LIFT snapshot — returning False")
            return False
        s_thr = self.bag_strap_lift_thresh
        g_thr = self.strap_grasp_max_dist
        L_up = (snap["left_strap_tip_z"] > s_thr) and (snap["left_strap_mid_z"] > s_thr)
        R_up = (snap["right_strap_tip_z"] > s_thr) and (snap["right_strap_mid_z"] > s_thr)
        # pinched at the TCP, not merely hooked over a finger/wrist (a hooked
        # strap also clears the z threshold and carries the bag)
        L_pinch = snap["left_strap_pinch"]["min_dist"] < g_thr
        R_pinch = snap["right_strap_pinch"]["min_dist"] < g_thr
        L_ok = L_up and L_pinch
        R_ok = R_up and R_pinch
        z0, z1 = snap.get("bag_centroid_z_initial"), snap.get("bag_centroid_z")
        if z0 is None or z1 is None:
            logger.warning("[pack check_success] missing bag centroid samples — returning False")
            return False
        bag_ok = (z1 - z0) > self.bag_lift_thresh

        # (5) HOLD — still pinched at the end of the B5 carry (see
        # _snapshot_hold). Same instrument and same threshold as (3); no new
        # parameter, and nothing world-frame, so B5 PLACE cannot trip it.
        hsnap = self._hold_snapshot
        if hsnap is None:
            logger.warning("[pack check_success] no HOLD snapshot (B5 carry never "
                           "sampled) — hold=False")
            hold_L = hold_R = False
            hold_str = "hold=—(no snapshot)"
        else:
            hold_L = hsnap["left_strap_pinch"]["min_dist"] < g_thr
            hold_R = hsnap["right_strap_pinch"]["min_dist"] < g_thr
            hold_str = (f"hold L={hold_L}({hsnap['left_strap_pinch']['min_dist']:.4f}) "
                        f"R={hold_R}({hsnap['right_strap_pinch']['min_dist']:.4f})"
                        f"<{g_thr:.3f}")
        hold_ok = bool(hold_L and hold_R)

        result = bool(self._snickers_ok and L_ok and R_ok and bag_ok and hold_ok)
        logger.info(
            f"[pack check_success] snickers_in_mouth={self._snickers_ok}  "
            f"L={L_ok}(up={L_up} pinch={snap['left_strap_pinch']['min_dist']:.4f}<{g_thr:.3f}) "
            f"R={R_ok}(up={R_up} pinch={snap['right_strap_pinch']['min_dist']:.4f}<{g_thr:.3f}) "
            f"bag_Δz={z1 - z0:+.3f}(thr={self.bag_lift_thresh:.3f}) | "
            f"{hold_str} → hold_ok={hold_ok} → {result}")
        return result


# ===========================================================================
# Closed-loop evaluator (eval_sim registers this for pack_snickers).
# Constructed with `spec` only; env arrives via observe(env) — strap topology
# loads lazily on the first poll (same pattern as BagStrapEvaluator).
# ===========================================================================
def _load_pack_topology(cloth_obj_path) -> Optional[dict]:
    """{left_strap,right_strap:{tip,mid}, body} vertex indices from the
    ``*_strap_topology.json`` next to the bag OBJ. None if absent."""
    cloth_dir = os.path.dirname(str(cloth_obj_path))
    cands = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
    if not cands:
        logger.warning(f"[pack-eval] no *_strap_topology.json in {cloth_dir}")
        return None
    stem = os.path.splitext(os.path.basename(str(cloth_obj_path)))[0]
    cands.sort(key=lambda p: -len(os.path.commonprefix([os.path.basename(p), stem])))
    with open(cands[0]) as fh:
        topo = json.load(fh)
    out = {side: {"tip": np.asarray(topo[side]["tip_indices"], dtype=np.int64),
                  "mid": np.asarray(topo[side]["mid_indices"], dtype=np.int64)}
           for side in ("left_strap", "right_strap")}
    if "body_indices" in topo:
        out["body"] = np.asarray(topo["body_indices"], dtype=np.int64)
    logger.info(f"[pack-eval] topology: {cands[0]}")
    return out


class PackSnickersEvaluator:
    """Closed-loop staged success/score for pack_snickers (pick bar -> drop in
    bag -> bimanual strap grasp -> lift).

    Per-poll atoms (~1s, baselines on the first poll, sustained = held >=2
    consecutive polls at ANY point — every stage is a mid-episode WINDOW):
      bar_up    bar z rose >= bar_lift_margin        (stage A: bar was picked)
      in_bag    bar within snickers_success_xy_radius of the live bag-mouth
                centroid AND below mouth rim          (stage B: bar in the bag)
      l_ok/r_ok strap rose >= strap_lift_margin AND pinched at the TCP
                (strap_pinch_metrics < strap_grasp_max_dist) — hooked straps
                that rise without being pinched do NOT count
      carry     bag centroid rose >= bag_lift_margin AND in_bag still true
                (the bag is lifted WITH the bar riding inside)

    Score (staged, sums to 100):
      A sustained(bar_up)=20, B sustained(in_bag)=30,
      C1 sustained(l_ok)=20, C2 sustained(r_ok)=20, C3 sustained(carry)=10.
    SR = sustained(in_bag) AND sustained(l_ok & r_ok & carry): putting the bar
    in the bag is a NECESSARY condition — a perfect strap-lift of an empty bag
    is False (user 2026-07-16). Raw per-poll series kept in eval_metrics.
    """

    early_exit = True   # stop at the first held full-success window

    def __init__(self, spec):
        self.spec = spec
        p = spec.task_params
        self.body_name = str(p.get("snickers_body_name", "snickers_body"))
        self.in_radius = float(p.get("snickers_success_xy_radius", 0.12))
        self.z_below_mouth = float(p.get("snickers_success_z_below_mouth", 0.02))
        self.bar_lift_margin = float(p.get("bar_lift_margin", 0.10))
        self.strap_margin = float(p.get("strap_lift_margin", 0.10))
        self.bag_margin = float(p.get("bag_lift_margin", p.get("bag_lift_thresh", 0.05)))
        self.grasp_max_dist = float(p.get("strap_grasp_max_dist", 0.02))
        self._idx: Optional[dict] = None
        self._base: Optional[dict] = None
        self._hist: list[dict] = []

    def reset_eval(self) -> None:
        self._idx = None
        self._base = None
        self._hist = []

    def _strap_z(self, cloth: np.ndarray, side: str) -> float:
        return float(np.mean(np.concatenate(
            [cloth[self._idx[side]["tip"], 2], cloth[self._idx[side]["mid"], 2]])))

    def _mouth(self, cloth: np.ndarray) -> Tuple[np.ndarray, float]:
        sample = cloth[self._idx["body"]] if "body" in (self._idx or {}) else cloth
        z_thr = float(np.quantile(sample[:, 2], 0.90))
        mask = sample[:, 2] >= z_thr
        top = sample[mask] if int(mask.sum()) >= 32 else sample[np.argsort(sample[:, 2])[-32:]]
        return top[:, :2].mean(axis=0), float(sample[:, 2].max())

    def observe(self, env) -> None:
        if self._idx is None:
            self._idx = _load_pack_topology(env.cloth_obj_path)
        cloth = env.get_cloth_position()
        if cloth is None or self._idx is None:
            self._hist.append({k: False for k in ("bar_up", "in_bag", "l", "r", "carry")})
            return
        bar = env.get_rigid_body_pose(self.body_name)[:3]
        zl, zr = self._strap_z(cloth, "left_strap"), self._strap_z(cloth, "right_strap")
        zbag = float(np.mean(cloth[:, 2]))
        if self._base is None:
            self._base = {"bar": round(float(bar[2]), 4), "l": round(zl, 4),
                          "r": round(zr, 4), "bag": round(zbag, 4)}
        mouth_xy, mouth_z = self._mouth(cloth)
        dxy = float(np.linalg.norm(bar[:2] - mouth_xy))
        pd_l = strap_pinch_metrics(env, cloth, self._idx["left_strap"], "left")["min_dist"]
        pd_r = strap_pinch_metrics(env, cloth, self._idx["right_strap"], "right")["min_dist"]
        dz_bag = zbag - self._base["bag"]
        in_bag = dxy < self.in_radius and bar[2] < mouth_z - self.z_below_mouth
        rec = {
            "bar_up": (float(bar[2]) - self._base["bar"]) > self.bar_lift_margin,
            "in_bag": bool(in_bag),
            "l": (zl - self._base["l"]) > self.strap_margin and pd_l < self.grasp_max_dist,
            "r": (zr - self._base["r"]) > self.strap_margin and pd_r < self.grasp_max_dist,
            "carry": dz_bag > self.bag_margin and bool(in_bag),
            "bar_z": round(float(bar[2]), 4), "dxy_mouth": round(dxy, 4),
            "dz_l": round(zl - self._base["l"], 4), "dz_r": round(zr - self._base["r"], 4),
            "dz_bag": round(dz_bag, 4), "pinch_l": round(pd_l, 4), "pinch_r": round(pd_r, 4),
        }
        self._hist.append(rec)
        logger.info(f"[pack-eval] poll#{len(self._hist)}: bar_up={rec['bar_up']} "
                    f"in_bag={rec['in_bag']}(dxy={dxy:.3f}) "
                    f"L={rec['l']}(dz={rec['dz_l']:+.3f} pinch={pd_l:.3f}) "
                    f"R={rec['r']}(dz={rec['dz_r']:+.3f} pinch={pd_r:.3f}) "
                    f"carry={rec['carry']}(dz_bag={dz_bag:+.3f})")

    def check_success(self) -> bool:
        """SR: bar placed in the bag AND the bag lifted by both pinched straps
        with the bar riding inside, each held ~1s at some point."""
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        lifted = [h["l"] and h["r"] and h["carry"] for h in self._hist]
        return checks.sustained(ser("in_bag")) and checks.sustained(lifted)

    def get_score(self) -> float:
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        stage = lambda k, pts: pts if checks.sustained(ser(k)) else 0  # noqa: E731
        return float(stage("bar_up", 20) + stage("in_bag", 30)
                     + stage("l", 20) + stage("r", 20) + stage("carry", 10))

    def eval_metrics(self) -> dict:
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        return {
            "stages": {k: checks.sustained(ser(k))
                       for k in ("bar_up", "in_bag", "l", "r", "carry")},
            "score_split": {"A_bar_up": 20 * checks.sustained(ser("bar_up")),
                            "B_in_bag": 30 * checks.sustained(ser("in_bag")),
                            "C1_left": 20 * checks.sustained(ser("l")),
                            "C2_right": 20 * checks.sustained(ser("r")),
                            "C3_carry": 10 * checks.sustained(ser("carry"))},
            "thresholds": {"bar_lift_margin": self.bar_lift_margin,
                           "in_radius": self.in_radius, "z_below_mouth": self.z_below_mouth,
                           "strap_margin": self.strap_margin, "bag_margin": self.bag_margin,
                           "grasp_max_dist": self.grasp_max_dist},
            "baseline": self._base, "n_polls": len(self._hist),
            "series": self._hist,
        }
