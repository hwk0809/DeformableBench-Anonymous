"""task/task_grasp_bag_strap.py

Bimanual bag-strap grasp & lift (short-horizon). Ported 1:1 from
deformable_bench ``task_grasp_bag_strap_single_v4.py`` (the "v4 symmetric
world-pose-driven" version); only the env accessors are swapped for the newton
``DeformableEnv`` interface — all offsets, IK seeds, PLACE joints and success
thresholds are the verbatim deformable_bench defaults.

Pipeline:
  Stage 0  hold home + bag settle
  Stage 1  PRE   : IK (v5 seed) -> joint goal, grippers open
  Stage 2  GRASP : IK (v5 seed) -> joint goal, grippers still open
  Stage 3  close grippers on both straps
  Stage 4  LIFT  : pure vertical, grasp xy held, z -> lift_height. Snapshot
                   strap tip/mid z + bag centroid z right after (for success).
  Stage 5  PLACE : joint replay (_*_PLACE_JOINTS, hardcoded — that pose is IK
                   singular, real-robot solved it with a specific tilt quat).
  Stage 6  release + return to the un-randomized lift_silk preset.

Grasping the (cloth) strap -> control_mode: kinematic, same as the other
bimanual cloth-grasp tasks (grasp_silk / fling); the §5 motion-history fix
carries the strap with the fingers.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Generator, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from deformablebench.bench.common import checks
from deformablebench.bench.common.types import ExecutionStep
from deformablebench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
from deformablebench.bench.robot.planning.planner_manager import PlannerManager
from deformablebench.bench.task.base_task import BaseTask
from deformablebench.bench.task.randomization import _ROBOT_INIT_PRESETS


def _mirror_quat_y(q: np.ndarray) -> np.ndarray:
    """Reflect a quat (w,x,y,z) across the world y=0 plane: (w,-x,y,-z)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], q[2], -q[3]], dtype=np.float64)


# ===========================================================================
# Reference values from real-robot v5 hand-tuning (FK on v5 working joints ->
# world EE pose; LEFT/mirror(RIGHT) midpoint). Verbatim deformable_bench v4.
# ===========================================================================
_LEFT_PRE_SEED   = np.array([+0.3713, +2.0562, -1.5176, -1.3803, +1.2316, +0.8464])
_RIGHT_PRE_SEED  = np.array([-0.3081, +1.9306, -1.5293, +1.2006, +1.2280, -0.1168])
_LEFT_GRASP_SEED = np.array([+0.0986, +2.1238, -1.5176, -1.5683, +1.1509, +0.8464])
_RIGHT_GRASP_SEED = np.array([+0.0632, +1.9825, -1.5248, +1.2731, +1.0008, -0.1939])

_PRE_QUAT_LEFT   = np.array([+0.3006, -0.0571, +0.8060, -0.5067])
_GRASP_QUAT_LEFT = np.array([+0.2554, -0.0391, +0.7750, -0.5768])

_PRE_OFFSET_DEFAULT   = (+0.080, 0.278, 0.205)
_GRASP_OFFSET_DEFAULT = (+0.095, 0.140, 0.198)

_LEFT_PLACE_JOINTS  = np.array([-0.9789, +2.2190, -2.2725, +0.1054, +0.0253, +0.0604])
_RIGHT_PLACE_JOINTS = np.array([-0.3088, +1.0482, -0.9774, -0.3137, +0.1676, +0.1219])


class BagStrapGraspTask(BaseTask):
    """Symmetric IK-driven bimanual bag-strap grasp + lift + place."""

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        *,
        cfg: Optional[DictConfig] = None,
        collector=None,
        urdf_path: Optional[str] = None,
        pre_offset_xyz:   Tuple[float, float, float] = _PRE_OFFSET_DEFAULT,
        grasp_offset_xyz: Tuple[float, float, float] = _GRASP_OFFSET_DEFAULT,
        pre_approach_dz: float = 0.06,   # PRE hovers this far above the strap mids
        lift_height:    float = 0.45,
        left_dy:  float = 0.0,
        right_dy: float = 0.0,
        left_dz:  float = 0.0,
        right_dz: float = 0.0,
        open_gripper_value:  float = 1.0,
        close_gripper_value: float = 0.0,
        home_to_pre_velocity_scale:  float = 0.40,
        pre_to_grasp_velocity_scale: float = 0.15,
        lift_velocity_scale:         float = 0.30,
        return_home_velocity_scale:  float = 0.40,
        bag_settle_duration: float = 1.5,
        pre_settle:          float = 1.0,
        grasp_settle:        float = 0.8,
        close_duration:      float = 1.0,
        post_close_settle:   float = 0.5,
        lift_settle:         float = 1.0,
        return_home_settle:  float = 0.5,
        strap_lift_thresh: float = 0.35,
        bag_lift_thresh:   float = 0.05,
        strap_grasp_max_dist: float = 0.02,
        grasp_correct_max: float = 0.0,   # 0 = no corrective move (see pack task note)
        aim_yaw_from_strap: bool = False,  # see _measured_bag_yaw
        strap_yaw_rest_deg: float = 0.0,   # this asset's reading at zero rotation
    ):
        # ---- yaml overrides (yaml > kwargs > module defaults) ----
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})

            def _xyz(key, default):
                v = tp.get(key, None)
                if v is None:
                    return tuple(float(x) for x in default)
                if hasattr(v, "__iter__"):
                    return tuple(float(x) for x in list(v))
                return tuple(float(x) for x in default)

            pre_offset_xyz   = _xyz("pre_offset_xyz",   pre_offset_xyz)
            grasp_offset_xyz = _xyz("grasp_offset_xyz", grasp_offset_xyz)
            pre_approach_dz = float(tp.get("pre_approach_dz", pre_approach_dz))
            lift_height = float(tp.get("lift_height", lift_height))
            left_dy   = float(tp.get("left_dy",   left_dy))
            right_dy  = float(tp.get("right_dy",  right_dy))
            left_dz   = float(tp.get("left_dz",   left_dz))
            right_dz  = float(tp.get("right_dz",  right_dz))
            open_gripper_value  = float(tp.get("open_gripper_value",  open_gripper_value))
            close_gripper_value = float(tp.get("close_gripper_value", close_gripper_value))
            home_to_pre_velocity_scale  = float(tp.get("home_to_pre_velocity_scale",  home_to_pre_velocity_scale))
            pre_to_grasp_velocity_scale = float(tp.get("pre_to_grasp_velocity_scale", pre_to_grasp_velocity_scale))
            lift_velocity_scale = float(tp.get("lift_velocity_scale", lift_velocity_scale))
            return_home_velocity_scale = float(tp.get("return_home_velocity_scale", return_home_velocity_scale))
            bag_settle_duration = float(tp.get("bag_settle_duration", bag_settle_duration))
            pre_settle        = float(tp.get("pre_settle",        pre_settle))
            grasp_settle      = float(tp.get("grasp_settle",      grasp_settle))
            close_duration    = float(tp.get("close_duration",    close_duration))
            post_close_settle = float(tp.get("post_close_settle", post_close_settle))
            lift_settle       = float(tp.get("lift_settle",       lift_settle))
            return_home_settle = float(tp.get("return_home_settle", return_home_settle))
            strap_lift_thresh = float(tp.get("strap_lift_thresh", strap_lift_thresh))
            bag_lift_thresh   = float(tp.get("bag_lift_thresh",   bag_lift_thresh))
            strap_grasp_max_dist = float(tp.get("strap_grasp_max_dist", strap_grasp_max_dist))
            grasp_correct_max = float(tp.get("grasp_correct_max", grasp_correct_max))
            aim_yaw_from_strap = bool(tp.get("aim_yaw_from_strap", aim_yaw_from_strap))
            strap_yaw_rest_deg = float(tp.get("strap_yaw_rest_deg", strap_yaw_rest_deg))

        super().__init__(env, planner_manager, collector, cfg)

        self.pre_offset   = np.asarray(pre_offset_xyz,   dtype=np.float64)
        self.grasp_offset = np.asarray(grasp_offset_xyz, dtype=np.float64)
        self.pre_approach_dz = float(pre_approach_dz)
        self.lift_height  = float(lift_height)
        self.left_dy  = float(left_dy);  self.right_dy = float(right_dy)
        self.left_dz  = float(left_dz);  self.right_dz = float(right_dz)
        self.open_gripper  = float(open_gripper_value)
        self.close_gripper = float(close_gripper_value)
        self.home_to_pre_velocity_scale  = float(home_to_pre_velocity_scale)
        self.pre_to_grasp_velocity_scale = float(pre_to_grasp_velocity_scale)
        self.lift_velocity_scale         = float(lift_velocity_scale)
        self.return_home_velocity_scale  = float(return_home_velocity_scale)
        self.bag_settle_duration = float(bag_settle_duration)
        self.pre_settle    = float(pre_settle)
        self.grasp_settle  = float(grasp_settle)
        self.close_duration    = float(close_duration)
        self.post_close_settle = float(post_close_settle)
        self.lift_settle   = float(lift_settle)
        self.return_home_settle = float(return_home_settle)
        self.strap_lift_thresh = float(strap_lift_thresh)
        self.bag_lift_thresh   = float(bag_lift_thresh)
        self.strap_grasp_max_dist = float(strap_grasp_max_dist)
        self.grasp_correct_max = float(grasp_correct_max)
        self.aim_yaw_from_strap = bool(aim_yaw_from_strap)
        self.strap_yaw_rest_deg = float(strap_yaw_rest_deg)
        self._lift_snapshot: Optional[dict] = None
        self._initial_bag_centroid_z: Optional[float] = None
        self._strap_idx = self._load_strap_topology()

        if urdf_path is None:
            urdf_path = env.robot.urdf
        self._kin = PinocchioKinematics(
            urdf_path, base_link="base_link", ee_link="tcp_link_center")

        logger.info(
            f"[strap] PRE offset={tuple(self.pre_offset)}; GRASP offset={tuple(self.grasp_offset)}; "
            f"LIFT z={self.lift_height:.3f}; strap_lift_thresh={self.strap_lift_thresh:.3f}"
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _load_strap_topology(self) -> Optional[dict]:
        """tip/mid indices for both straps from *_strap_topology.json in the
        cloth asset dir. None -> check_success falls back to always-False."""
        cloth_dir = os.path.dirname(str(self.env.cloth_obj_path))
        candidates = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
        if not candidates:
            logger.warning(f"[strap] no *_strap_topology.json in {cloth_dir}")
            return None
        obj_stem = os.path.splitext(os.path.basename(str(self.env.cloth_obj_path)))[0]
        candidates.sort(key=lambda p: -len(os.path.commonprefix([
            os.path.basename(p), obj_stem,
        ])))
        topo_path = candidates[0]
        logger.info(f"[strap] using strap topology: {topo_path}")
        with open(topo_path) as f:
            topo = json.load(f)
        out = {}
        for side in ("left_strap", "right_strap"):
            s = topo[side]
            out[side] = {
                "tip": np.asarray(s["tip_indices"], dtype=np.int64),
                "mid": np.asarray(s["mid_indices"], dtype=np.int64),
            }
        logger.info(
            f"[strap] topology loaded: "
            f"L tip={len(out['left_strap']['tip'])} mid={len(out['left_strap']['mid'])}  "
            f"R tip={len(out['right_strap']['tip'])} mid={len(out['right_strap']['mid'])}"
        )
        return out

    def _bag_centroid_z(self) -> Optional[float]:
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None or len(cloth_pos) == 0:
            return None
        return float(np.mean(np.asarray(cloth_pos)[:, 2]))

    def _snapshot_initial_bag_centroid(self) -> None:
        z = self._bag_centroid_z()
        self._initial_bag_centroid_z = z
        if z is not None:
            logger.info(f"[strap] initial bag centroid z (post-settle) = {z:.3f}")

    def _snapshot_strap_z(self) -> None:
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
            arm = side.split("_")[0]
            pm = strap_pinch_metrics(self.env, cloth_pos, self._strap_idx[side], arm)
            snap[f"{side}_pinch"] = pm
        snap["bag_centroid_z"] = float(np.mean(np.asarray(cloth_pos)[:, 2]))
        snap["bag_centroid_z_initial"] = (
            float(self._initial_bag_centroid_z)
            if self._initial_bag_centroid_z is not None else None
        )
        self._lift_snapshot = snap
        delta_str = (
            f"Δ={snap['bag_centroid_z'] - snap['bag_centroid_z_initial']:+.3f}"
            if snap["bag_centroid_z_initial"] is not None else "Δ=n/a"
        )
        logger.info(
            f"[strap] LIFT snapshot: "
            f"L tip={snap['left_strap_tip_z']:.3f} mid={snap['left_strap_mid_z']:.3f}  "
            f"R tip={snap['right_strap_tip_z']:.3f} mid={snap['right_strap_mid_z']:.3f}  "
            f"strap_thr={self.strap_lift_thresh:.3f}  "
            f"bag_z={snap['bag_centroid_z']:.3f} ({delta_str}, thr={self.bag_lift_thresh:.3f})"
        )
        for side in ("left_strap", "right_strap"):
            pm = snap[f"{side}_pinch"]
            logger.info(
                f"[strap] pinch[{side}]: min_dist={pm['min_dist']:.4f} t={pm['t']:+.4f} "
                f"rho={pm['rho']:.4f} gap={pm['gap']:.3f} finger_sep={pm['finger_sep']:.4f} "
                f"(max_dist={self.strap_grasp_max_dist:.3f})")

    def _resolve_home_joints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *un-randomized* start-pose preset joints (return-home
        target). Reads the preset name from the task's randomization cfg so it
        doesn't depend on the noised env.home_q_*."""
        rand = self.cfg.get("randomization", {}) if hasattr(self.cfg, "get") else {}
        name = (rand.get("robot_init", {}) or {}).get("start_pose")
        preset = _ROBOT_INIT_PRESETS.get(name) if name else None
        if preset is not None:
            return preset["home_left"].copy(), preset["home_right"].copy()
        logger.warning("[strap] robot_init preset missing — falling back to env.home_q_*")
        return np.asarray(self.env.home_q_left).copy(), np.asarray(self.env.home_q_right).copy()

    def _cloth_centroid_y(self) -> float:
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None or len(cloth_pos) == 0:
            return float(self.env.cloth_base_pos[1])
        return float(np.mean(np.asarray(cloth_pos)[:, 1]))

    def _bag_base(self) -> np.ndarray:
        """deformable_bench's ``base_pos`` = the bag OBJ-origin world position.
        deformablebench now anchors all cloth by OBJ origin (builder), so env.cloth_base_pos
        IS the OBJ origin = deformable's base_pos directly."""
        return self.env.cloth_base_pos

    def _measured_bag_yaw(self) -> float:
        """Bag yaw [rad] read off the two strap-mid centroids (0 = asset rest pose).

        The symbolic targets are placement-tracking but orientation-BLIND: they add
        a fixed ``±|Δy|`` about the bag origin, so a yawed bag swings its straps out
        from under the aim by ``r*sin(yaw)`` with ``r`` = the strap's ~0.13 m lever
        arm. Measured on the shipped ±5 deg range: aim error along x =
        -2.31 mm/deg * yaw (r = -0.993, 8 episodes), i.e. ±11 mm across the range —
        wider than the jaw's capture window, and the sole cause of the episodes the
        task drops (all four failures sat at yaw >= +4.3 deg).

        Strap mids rather than the body centroid: the centroid is yaw-invariant by
        construction (the bag rotates about it), the straps are what carries the
        signal. Vertex sets come from the asset's ``*_strap_topology.json``.

        ``strap_yaw_rest_deg`` is the reading this asset gives at zero commanded
        rotation — the settled bag's straps are not exactly parallel to y, and the
        shipped offsets were calibrated against that pose. Measured for
        nonwoven_small_6: reading - commanded = +3.53 deg, constant to 0.05 deg
        over 6 episodes spanning -1.9..+5.0 deg. Subtracting it makes this a pure
        yaw CORRECTION; leaving it in would also bake a fixed ~8 mm aim shift.
        """
        cloth = self.env.get_cloth_position()
        if self._strap_idx is None or cloth is None or len(cloth) == 0:
            return 0.0
        v = (cloth[self._strap_idx["left_strap"]["mid"]].mean(axis=0)
             - cloth[self._strap_idx["right_strap"]["mid"]].mean(axis=0))
        # rest pose: left mid is +y of right mid, so yaw = angle of v off +y
        return float(np.arctan2(-v[0], v[1])) - np.deg2rad(self.strap_yaw_rest_deg)

    @staticmethod
    def _rotz(v: np.ndarray, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float64)

    @staticmethod
    def _rotz_quat(q: np.ndarray, yaw: float) -> np.ndarray:
        """Pre-rotate a (w,x,y,z) quat by ``yaw`` about the WORLD z axis."""
        c, s = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        w, x, y, z = (float(t) for t in q)
        return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w],
                        dtype=np.float64)

    def _build_world_targets(
        self, offset_xyz, quat_left,
        l_dy=0.0, r_dy=0.0, l_dz=0.0, r_dz=0.0, y_ref=None, yaw=0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(left_world_pose, right_world_pose), 7D each — deformable_bench's exact
        symbolic planning. x/z ref = bag OBJ-origin (_bag_base); y ref = cloth
        centroid (absorbs the v5 asymmetric y bias). |Δy| mirrored L/R.

        ``yaw`` (rad, see ``_measured_bag_yaw``) turns the offset vector and the
        approach quats about the vertical through that reference point — the same
        axis the builder spins the cloth about — so the pair of targets rides with
        a rotated bag. ``yaw=0`` reproduces the original formula bit for bit."""
        bag = self._bag_base()
        if y_ref is None:
            y_ref = self._cloth_centroid_y()
        ref = np.array([bag[0], y_ref, bag[2]], dtype=np.float64)
        # -0.01 on the left hand's x: kept from ce45a11, folded into the offset
        # vector so it rides the yaw like the rest of the aim (at yaw=0 this is
        # bit-for-bit that commit's formula).
        # -0.01 = the left hand's reach trim (it lands ~1 cm deeper than the right
        # against the same strap). It rides INSIDE the rotated vector, so it stays
        # a trim relative to the bag rather than a world-x shift.
        l_pos = ref + self._rotz(
            np.array([offset_xyz[0] - 0.01, abs(offset_xyz[1]) + l_dy, offset_xyz[2] + l_dz]), yaw)
        r_pos = ref + self._rotz(
            np.array([offset_xyz[0], -abs(offset_xyz[1]) + r_dy, offset_xyz[2] + r_dz]), yaw)
        # Mirror FIRST, then spin both: a yawed bag turns the two approach frames
        # the same way. Mirroring a pre-rotated quat would turn the right hand by
        # -yaw instead.
        q_l, q_r = np.asarray(quat_left, dtype=np.float64), _mirror_quat_y(quat_left)
        if yaw:
            q_l, q_r = self._rotz_quat(q_l, yaw), self._rotz_quat(q_r, yaw)
        return np.concatenate([l_pos, q_l]), np.concatenate([r_pos, q_r])

    def _solve_ik_pair(self, left_world, right_world, l_seed, r_seed, tag="") -> Tuple[np.ndarray, np.ndarray]:
        l_base_xyz = np.asarray(self.env.left_base[:3])
        r_base_xyz = np.asarray(self.env.right_base[:3])
        l_target_base = np.concatenate([left_world[:3]  - l_base_xyz, left_world[3:]])
        r_target_base = np.concatenate([right_world[:3] - r_base_xyz, right_world[3:]])
        ik_kwargs = dict(pose_constraint=[1, 1, 1, 1, 0, 1], constraint_frame="local",
                         threshold=5e-3, max_restarts=30)
        l_q = self._kin.solve_ik(l_target_base, seed_joints=l_seed, **ik_kwargs)
        r_q = self._kin.solve_ik(r_target_base, seed_joints=r_seed, **ik_kwargs)
        if l_q is None:
            raise RuntimeError(f"[strap] LEFT IK failed at {tag} (target={l_target_base[:3]})")
        if r_q is None:
            raise RuntimeError(f"[strap] RIGHT IK failed at {tag} (target={r_target_base[:3]})")
        return l_q[:6], r_q[:6]

    # ----------------------------------------------------------------------
    # plan_episode
    # ----------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        # ---- Stage 0: hold home + bag settle ----
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        state = self.skills.planning_state
        plan_home = self.skills.move_dual_arm_to(
            left_target_pose  = np.asarray(state.left_joints).copy(),
            right_target_pose = np.asarray(state.right_joints).copy(),
            goal_type     = "joint",
            left_gripper  = state.left_gripper,
            right_gripper = state.right_gripper,
        )
        yield ExecutionStep(plan=plan_home, settle_time=self.bag_settle_duration,
                            description="bag_settle")

        self._snapshot_initial_bag_centroid()

        y_ref = self._cloth_centroid_y()
        bag_xy0 = self.env.cloth_base_pos
        logger.info(
            f"[strap] y_ref (cloth centroid)={y_ref:+.4f}, "
            f"bag spawn=({bag_xy0[0]:+.3f},{bag_xy0[1]:+.3f},{bag_xy0[2]:+.3f})"
        )

        yaw = self._measured_bag_yaw() if self.aim_yaw_from_strap else 0.0
        if self.aim_yaw_from_strap:
            logger.info(f"[strap] measured bag yaw = {np.degrees(yaw):+.2f} deg "
                        f"(targets rotated about the bag origin)")

        pre_l_w, pre_r_w = self._build_world_targets(
            self.pre_offset, _PRE_QUAT_LEFT, y_ref=y_ref, yaw=yaw)
        gr_l_w, gr_r_w = self._build_world_targets(
            self.grasp_offset, _GRASP_QUAT_LEFT,
            l_dy=self.left_dy, r_dy=self.right_dy, l_dz=self.left_dz, r_dz=self.right_dz,
            y_ref=y_ref, yaw=yaw)

        lift_l_pos = np.array([gr_l_w[0], gr_l_w[1], self.lift_height])
        lift_r_pos = np.array([gr_r_w[0], gr_r_w[1], self.lift_height])
        lift_quat_l = self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.left_base)[:3], lift_l_pos)
        lift_quat_r = self.skills.calculate_base_to_target_horizontal_quat(
            np.asarray(self.env.right_base)[:3], lift_r_pos)
        lift_l_w = np.concatenate([lift_l_pos, lift_quat_l])
        lift_r_w = np.concatenate([lift_r_pos, lift_quat_r])

        l_q_pre,  r_q_pre  = self._solve_ik_pair(pre_l_w, pre_r_w, _LEFT_PRE_SEED, _RIGHT_PRE_SEED, tag="PRE")
        l_q_gr,   r_q_gr   = self._solve_ik_pair(gr_l_w, gr_r_w, _LEFT_GRASP_SEED, _RIGHT_GRASP_SEED, tag="GRASP")
        l_q_lift, r_q_lift = self._solve_ik_pair(lift_l_w, lift_r_w, l_q_gr, r_q_gr, tag="LIFT")
        l_q_place, r_q_place = _LEFT_PLACE_JOINTS.copy(), _RIGHT_PLACE_JOINTS.copy()

        self.task_meta = {
            "episode_id": episode_id,
            "bag_pos": self.env.cloth_base_pos.tolist(),
            "grasp_world_left": gr_l_w.tolist(), "grasp_world_right": gr_r_w.tolist(),
            "lift_world_left": lift_l_w.tolist(), "lift_world_right": lift_r_w.tolist(),
            "description": "symmetric IK-driven bimanual bag-strap grasp",
        }
        logger.info(f"[strap] joint plan: PRE L={l_q_pre.round(3)} R={r_q_pre.round(3)}")
        logger.info(f"[strap] joint plan: GRSP L={l_q_gr.round(3)} R={r_q_gr.round(3)}")

        # ---- Stage 1: home -> PRE ----
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("move_to_pre_grasp"):
            plan_pre = self.skills.move_dual_arm_to(
                left_target_pose=l_q_pre, right_target_pose=r_q_pre, goal_type="joint",
                left_gripper=self.open_gripper, right_gripper=self.open_gripper,
                velocity_scale=self.home_to_pre_velocity_scale)
            yield ExecutionStep(plan=plan_pre, settle_time=self.pre_settle, description="move_to_pre_grasp")

        # ---- Stage 2: PRE -> GRASP ----
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("move_to_grasp"):
            plan_grasp = self.skills.move_dual_arm_to(
                left_target_pose=l_q_gr, right_target_pose=r_q_gr, goal_type="joint",
                left_gripper=self.open_gripper, right_gripper=self.open_gripper,
                velocity_scale=self.pre_to_grasp_velocity_scale)
            yield ExecutionStep(plan=plan_grasp, settle_time=self.grasp_settle, description="move_to_grasp")

        # ---- Stage 2.5: measure-and-correct (same as pack's B2.5) ----
        # The symbolic GRASP target chases the strap with fixed offsets; the
        # per-episode drape can leave the strap several cm from the pinch
        # point (a shallow hook that slips later). Re-aim the TCP at the
        # actually-measured nearest strap vertex before closing.
        c = self.env.get_cloth_position()
        if (self.grasp_correct_max > 0 and self._strap_idx is not None
                and c is not None and len(c)):
            corr = {}
            for side, arm in (("left_strap", "left"), ("right_strap", "right")):
                eef = self.env.left_eef_pose if arm == "left" else self.env.right_eef_pose
                verts = c[np.concatenate([self._strap_idx[side]["tip"], self._strap_idx[side]["mid"]])]
                tcp = np.asarray(eef[:3], dtype=np.float64)
                rel = verts[int(np.argmin(np.linalg.norm(verts - tcp, axis=1)))] - tcp
                miss = float(np.linalg.norm(rel))
                if miss > self.grasp_correct_max:
                    rel = rel * (self.grasp_correct_max / miss)
                corr[arm] = (rel, tcp)
                logger.info(f"[strap] S2.5 correct[{side}]: miss={miss:.4f} apply={rel.round(4)}")
            if max(float(np.linalg.norm(v[0])) for v in corr.values()) > 0.005:
                gl2 = np.concatenate([corr["left"][1] + corr["left"][0], gr_l_w[3:]])
                gr2 = np.concatenate([corr["right"][1] + corr["right"][0], gr_r_w[3:]])
                try:
                    l_q_gr, r_q_gr = self._solve_ik_pair(gl2, gr2, l_q_gr, r_q_gr, tag="GRASP-CORRECT")
                    lift_l_pos = np.array([gl2[0], gl2[1], self.lift_height])
                    lift_r_pos = np.array([gr2[0], gr2[1], self.lift_height])
                    lift_l_w = np.concatenate([lift_l_pos, self.skills.calculate_base_to_target_horizontal_quat(
                        np.asarray(self.env.left_base)[:3], lift_l_pos)])
                    lift_r_w = np.concatenate([lift_r_pos, self.skills.calculate_base_to_target_horizontal_quat(
                        np.asarray(self.env.right_base)[:3], lift_r_pos)])
                    l_q_lift, r_q_lift = self._solve_ik_pair(lift_l_w, lift_r_w, l_q_gr, r_q_gr, tag="LIFT-CORRECT")
                    self.skills.sync_state(self.env.get_observation(include_visuals=False))
                    with self.recording("correct_grasp"):
                        plan_corr = self.skills.move_dual_arm_to(
                            left_target_pose=l_q_gr, right_target_pose=r_q_gr, goal_type="joint",
                            left_gripper=self.open_gripper, right_gripper=self.open_gripper,
                            velocity_scale=self.pre_to_grasp_velocity_scale)
                        yield ExecutionStep(plan=plan_corr, settle_time=self.grasp_settle,
                                            description="correct_grasp")
                except RuntimeError as e:
                    logger.warning(f"[strap] S2.5 corrective IK failed — keeping original grasp: {e}")

        # ---- Stage 3: close grippers ----
        with self.recording("close_grippers"):
            plan_close = self.skills.gripper_bimanual_action(
                left_target_val=self.close_gripper, right_target_val=self.close_gripper,
                duration=self.close_duration)
            yield ExecutionStep(plan=plan_close, settle_time=self.post_close_settle, description="close_grippers")

        # ---- Stage 4: LIFT ----
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("lift_bag"):
            plan_lift = self.skills.move_dual_arm_to(
                left_target_pose=l_q_lift, right_target_pose=r_q_lift, goal_type="joint",
                left_gripper=self.close_gripper, right_gripper=self.close_gripper,
                velocity_scale=self.lift_velocity_scale)
            yield ExecutionStep(plan=plan_lift, settle_time=self.lift_settle, description="lift_bag")

        self._snapshot_strap_z()
        if self._lift_snapshot is not None:
            self.task_meta["lift_snapshot"] = dict(self._lift_snapshot)

        # ---- Stage 5: PLACE (joint replay) ----
        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("place_bag"):
            plan_place = self.skills.move_dual_arm_to(
                left_target_pose=l_q_place, right_target_pose=r_q_place, goal_type="joint",
                left_gripper=self.close_gripper, right_gripper=self.close_gripper,
                velocity_scale=self.lift_velocity_scale)
            yield ExecutionStep(plan=plan_place, settle_time=self.lift_settle, description="place_bag")

        # ---- Stage 6: release + return home ----
        home_l, home_r = self._resolve_home_joints()
        with self.recording("release_grippers"):
            plan_release = self.skills.gripper_bimanual_action(
                left_target_val=self.open_gripper, right_target_val=self.open_gripper,
                duration=self.close_duration)
            yield ExecutionStep(plan=plan_release, settle_time=self.post_close_settle, description="release_grippers")

        self.skills.sync_state(self.env.get_observation(include_visuals=False))
        with self.recording("return_home"):
            plan_rh = self.skills.move_dual_arm_to(
                left_target_pose=home_l, right_target_pose=home_r, goal_type="joint",
                left_gripper=self.open_gripper, right_gripper=self.open_gripper,
                velocity_scale=self.return_home_velocity_scale)
            yield ExecutionStep(plan=plan_rh, settle_time=self.return_home_settle, description="return_home")

        logger.success("[strap] Done.")

    def check_success(self) -> bool:
        """Both straps' tip+mid mean z clear strap_lift_thresh AND each strap is
        actually PINCHED at the TCP (excludes hooked/poked straps, which also
        rise) AND bag centroid z rose by >= bag_lift_thresh from post-settle
        baseline (catches strap lifted but bag stayed / fingers slipped).
        Sampled right after LIFT."""
        snap = self._lift_snapshot
        if snap is None:
            logger.warning("[strap check_success] no LIFT snapshot — returning False")
            return False
        s_thr = self.strap_lift_thresh
        g_thr = self.strap_grasp_max_dist
        L_up = (snap["left_strap_tip_z"]  > s_thr) and (snap["left_strap_mid_z"]  > s_thr)
        R_up = (snap["right_strap_tip_z"] > s_thr) and (snap["right_strap_mid_z"] > s_thr)
        L_pinch = snap["left_strap_pinch"]["min_dist"] < g_thr
        R_pinch = snap["right_strap_pinch"]["min_dist"] < g_thr
        L_ok = L_up and L_pinch
        R_ok = R_up and R_pinch
        z0 = snap.get("bag_centroid_z_initial")
        z1 = snap.get("bag_centroid_z")
        if z0 is None or z1 is None:
            logger.warning("[strap check_success] missing bag centroid samples — returning False")
            return False
        bag_delta = z1 - z0
        bag_ok = bag_delta > self.bag_lift_thresh
        result = bool(L_ok and R_ok and bag_ok)
        logger.info(
            f"[strap check_success] strap_thr={s_thr:.3f} grasp_max_dist={g_thr:.3f} "
            f"bag_thr={self.bag_lift_thresh:.3f}  "
            f"L: tip={snap['left_strap_tip_z']:.3f} mid={snap['left_strap_mid_z']:.3f} "
            f"pinch={snap['left_strap_pinch']['min_dist']:.4f} → {L_ok}  "
            f"R: tip={snap['right_strap_tip_z']:.3f} mid={snap['right_strap_mid_z']:.3f} "
            f"pinch={snap['right_strap_pinch']['min_dist']:.4f} → {R_ok}  "
            f"bag_Δz={bag_delta:+.3f} → {bag_ok}  → {result}"
        )
        return result


# ===========================================================================
# Closed-loop evaluator (eval_sim registers this for grasp_bag_strap).
# Separate from the collect task above: constructed with `spec` only, env
# arrives via observe(env) — so strap topology is loaded lazily on the first
# poll (same pattern as FlattenSilkEvaluator).
# ===========================================================================
def load_strap_indices(cloth_obj_path) -> Optional[dict]:
    """{left_strap,right_strap} -> {tip,mid} vertex indices from the
    ``*_strap_topology.json`` baked next to the bag OBJ. None if absent."""
    cloth_dir = os.path.dirname(str(cloth_obj_path))
    cands = sorted(glob.glob(os.path.join(cloth_dir, "*_strap_topology.json")))
    if not cands:
        logger.warning(f"[strap-eval] no *_strap_topology.json in {cloth_dir}")
        return None
    stem = os.path.splitext(os.path.basename(str(cloth_obj_path)))[0]
    cands.sort(key=lambda p: -len(os.path.commonprefix([os.path.basename(p), stem])))
    with open(cands[0]) as f:
        topo = json.load(f)
    logger.info(f"[strap-eval] topology: {cands[0]}")
    return {side: {"tip": np.asarray(topo[side]["tip_indices"], dtype=np.int64),
                   "mid": np.asarray(topo[side]["mid_indices"], dtype=np.int64)}
            for side in ("left_strap", "right_strap")}


def strap_pinch_metrics(env, cloth: np.ndarray, idx_side: dict, side: str) -> dict:
    """Is the strap actually PINCHED between the finger pads (vs merely hooked)?

    A hooked/poked strap also clears the z-threshold (it hangs off a finger or
    the wrist and rises with the arm), so strap height alone over-reports
    success. Discriminator: the strap must pass through the pinch point.

    Decomposes the nearest strap vertex relative to the TCP pinch centre
    (``env.*_eef_pose`` = gripper base + eef_offset_z, i.e. between the pads)
    along the finger-finger closing axis:
      min_dist  distance of nearest strap vertex to the pinch centre
      t         along the closing axis (pinched -> within the gap, |t| small)
      rho       perpendicular to it (hooked over a finger -> large)
      gap       actual gripper open fraction (PD mode: > commanded close if blocked)
    Verdict uses min_dist only (calibrated); t/rho/gap are logged for diagnosis.
    """
    verts = cloth[np.concatenate([idx_side["tip"], idx_side["mid"]])]
    eef = env.left_eef_pose if side == "left" else env.right_eef_pose
    tcp = np.asarray(eef[:3], dtype=np.float64)
    d = np.linalg.norm(verts - tcp, axis=1)
    j = int(np.argmin(d))
    p_a, p_b = env.gripper_finger_positions(side)
    axis = p_b - p_a
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    rel = verts[j] - tcp
    t = float(rel @ axis)
    rho = float(np.linalg.norm(rel - t * axis))
    gap = float(env.left_gripper_val if side == "left" else env.right_gripper_val)
    return {"min_dist": float(d[j]), "t": t, "rho": rho, "gap": gap, "finger_sep": n}


class BagStrapEvaluator:
    """Closed-loop success/score for grasp_bag_strap (bimanual strap grasp+lift).

    Design (with user 2026-07-12; pinch added 2026-07-15):
      * Per-poll atoms, RELATIVE to a post-settle baseline captured on the first
        poll: left strap rose, right strap rose, bag rose.
      * 2026-07-12 assumed "a strap only rises if a gripper grabbed it" and
        skipped gripper-proximity. Counterexample observed 2026-07-15: a strap
        HOOKED over a finger/wrist (poked, not pinched) also rises and carries
        the bag. So each side's atom now additionally requires the strap to
        pass through the TCP pinch point (strap_pinch_metrics min_dist <
        strap_grasp_max_dist) — not loose whole-gripper proximity, which stays
        ambiguous on a ~1020-vertex strap loop.
      * SR = sustained(left_up AND right_up AND bag_up): all three held for >=2
        consecutive polls (~1s) at ANY point. The demo lifts then places back +
        releases, so the lift is a mid-episode WINDOW, never the end state —
        sustained scans for that window; early_exit stops at the first hit (judged
        at the lift, not at release / return-home).
      * Score = per-side sustained(50)/ever(25)/never(0), summed -> 0/25/50/75/100
        (judges LEFT and RIGHT like grasp_silk). bag-lift gates SR only.
      * Full per-poll series kept in eval_metrics (raw, recomposable offline).
    """

    early_exit = True   # stop at the first held lift window (before place/release)

    def __init__(self, spec):
        self.spec = spec
        p = spec.task_params
        # relative-to-baseline margins (meters). strap default: clearly off-table.
        self.strap_margin = float(p.get("strap_lift_margin", 0.15))
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

    def observe(self, env) -> None:
        if self._idx is None:
            self._idx = load_strap_indices(env.cloth_obj_path)
        cloth = env.get_cloth_position()
        if cloth is None or self._idx is None:
            self._hist.append({"left": False, "right": False, "bag": False,
                               "dz_l": 0.0, "dz_r": 0.0, "dz_bag": 0.0})
            return
        zl, zr, zb = self._strap_z(cloth, "left_strap"), self._strap_z(cloth, "right_strap"), \
            float(np.mean(cloth[:, 2]))
        if self._base is None:
            self._base = {"l": round(zl, 4), "r": round(zr, 4), "bag": round(zb, 4)}
        dz_l, dz_r, dz_bag = zl - self._base["l"], zr - self._base["r"], zb - self._base["bag"]
        pd_l = strap_pinch_metrics(env, cloth, self._idx["left_strap"], "left")["min_dist"]
        pd_r = strap_pinch_metrics(env, cloth, self._idx["right_strap"], "right")["min_dist"]
        rec = {"left": dz_l > self.strap_margin and pd_l < self.grasp_max_dist,
               "right": dz_r > self.strap_margin and pd_r < self.grasp_max_dist,
               "bag": dz_bag > self.bag_margin,
               "dz_l": round(dz_l, 4), "dz_r": round(dz_r, 4), "dz_bag": round(dz_bag, 4),
               "pinch_l": round(pd_l, 4), "pinch_r": round(pd_r, 4)}
        self._hist.append(rec)
        logger.info(f"[strap-eval] poll#{len(self._hist)}: "
                    f"dz L={dz_l:+.3f} pinch={pd_l:.3f} ({rec['left']}) "
                    f"R={dz_r:+.3f} pinch={pd_r:.3f} ({rec['right']}) "
                    f"bag={dz_bag:+.3f}({rec['bag']})")

    def check_success(self) -> bool:
        """SR: both straps lifted AND bag lifted, held ~1s (sustained) at any point."""
        return checks.sustained([h["left"] and h["right"] and h["bag"] for h in self._hist])

    def get_score(self) -> float:
        """Per side: sustained-held=50 / rose-then-slipped(ever)=25 / never=0, summed."""
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        side = lambda k: 50 if checks.sustained(ser(k)) else 25 if checks.ever(ser(k)) else 0  # noqa: E731
        return float(side("left") + side("right"))

    def eval_metrics(self) -> dict:
        ser = lambda k: [h[k] for h in self._hist]  # noqa: E731
        mx = lambda k: round(max((h[k] for h in self._hist), default=0.0), 4)  # noqa: E731
        return {
            "left_ever": checks.ever(ser("left")), "left_sustained": checks.sustained(ser("left")),
            "right_ever": checks.ever(ser("right")), "right_sustained": checks.sustained(ser("right")),
            "bag_ever": checks.ever(ser("bag")),
            "max_dz_left": mx("dz_l"), "max_dz_right": mx("dz_r"), "max_dz_bag": mx("dz_bag"),
            "strap_margin": self.strap_margin, "bag_margin": self.bag_margin,
            "baseline": self._base, "n_polls": len(self._hist),
            "series": self._hist,   # full per-poll raw record (recomposable offline)
        }
