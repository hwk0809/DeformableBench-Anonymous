"""Flatten tshirt: 双臂固定点抓取 (left_shoulder + left_hem) + 单次 fling 展平.

对齐 deformable_bench ``task_flatten_tshirt_v3.py`` (FlattenTshirtV3Task, blue_tshirt_9k):
策略 = 固定抓 left_shoulder + left_hem 两点 → 撑开抬起 → 单次 fling → coverage 判据.
初始态由 yaml ``two_point_swing_drop`` 随机化 (高处生成→双点 pin→拉出→贴桌抖落→释放).

与 flatten_silk 的区别: silk 是"候选选点 + 近端 fling/远端 spread"多轮;tshirt 这里是
固定两点单次 fling (对齐 deformable 的 tshirt 策略), 简单直接.

Usage:
    .venv/bin/python -m fabricbench.bench.run_task --task flatten_tshirt --viewer null
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Generator, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from fabricbench.bench.common.asset import _load_obj_preserve_vertex_order
from fabricbench.bench.common.types import ExecutionStep, ObservationMessage
from fabricbench.bench.common.metric_utils import (
    flatten_metrics, get_covered_area_from_particles)
from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.base_task import BaseTask


# ======================================================================
# Eval-only scorer (mirrors FlattenSilkEvaluator's contract: observe() every
# poll, then check_success()/get_score() after the driver releases + settles).
#
# Design (locked with user 2026-07-16, validated on 9 scripted-expert seeds +
# a crumpled failure baseline; raw particle dumps preserved):
#   * TWO complementary signals — coverage (footprint, catches "balled up") and
#     weighted_distance (ICP-to-canonical shape, catches "spread but folded",
#     incl. a single tucked sleeve that coverage misses). ClothFunnels/UniFolding
#     ``deformable_distance``. IoU is recorded as a diagnostic only (no rotation
#     alignment -> noisy here; weighted_distance is the rotation-aligned version).
#   * score = 100*(w_cov*coverage_norm + (1-w_cov)*shape_norm), graded 0..100.
#   * success = settled (mean_z<τz) AND coverage>=τcov AND weighted_distance<=τwd
#     — both gates, so neither a ball (low cov) nor a fold (high weightD) passes.
#   * All raw metrics recorded per poll (recomposable offline); init+final+
#     canonical particles dumped by the eval driver for full offline re-scoring.
# ======================================================================
class FlattenTshirtEvaluator:
    CLOTH_PARTICLE_RADIUS = 0.005
    early_exit = True         # stop policy actions on the first successful poll

    def __init__(self, spec):
        self.spec = spec
        p = dict(getattr(spec, "task_params", {}) or {})
        # success gates
        self.tau_cov = float(p.get("flatten_success_coverage", 0.60))
        self.tau_wd = float(p.get("flatten_success_weightd", 0.25))
        self.tau_z = float(p.get("flatten_success_mean_z", 0.03))
        # score normalization / weighting
        self.w_cov = float(p.get("flatten_score_w_cov", 0.4))
        self.cov_lo = float(p.get("flatten_cov_lo", 0.25))
        self.cov_hi = float(p.get("flatten_cov_hi", 0.78))
        self.wd_hi = float(p.get("flatten_wd_hi", 0.36))
        self.wd_lo = float(p.get("flatten_wd_lo", 0.196))
        settle_sec = float(p.get("terminal_settle_sec", 2.0))
        self.terminal_settle_ticks = max(1, int(round(spec.control_fps * settle_sec)))
        self._canonical: Optional[np.ndarray] = None
        self.reset_eval()

    def reset_eval(self) -> None:
        self._series: list[dict] = []
        self._init_particles: Optional[np.ndarray] = None
        self._final_particles: Optional[np.ndarray] = None

    def _canon(self, env) -> np.ndarray:
        if self._canonical is None:
            verts, _ = _load_obj_preserve_vertex_order(env.cloth_obj_path)
            self._canonical = np.asarray(verts, dtype=np.float64)
        return self._canonical

    def _measure(self, env):
        c = env.get_cloth_position()
        if c is None:
            return None
        c = np.asarray(c, dtype=np.float64)
        m = flatten_metrics(c, self._canon(env), self.CLOTH_PARTICLE_RADIUS)
        m = {k: round(float(v), 4) for k, v in m.items()}
        m["max_z"] = round(float(c[:, 2].max()), 4)
        m["mean_z"] = round(float(c[:, 2].mean()), 4)
        return m, c

    def observe(self, env) -> None:
        r = self._measure(env)
        if r is None:
            return
        m, c = r
        if self._init_particles is None:
            self._init_particles = c.copy()
        self._final_particles = c
        self._series.append(m)

    def _score_from(self, m: dict) -> float:
        cn = float(np.clip((m["coverage"] - self.cov_lo) / (self.cov_hi - self.cov_lo), 0, 1))
        sn = float(np.clip((self.wd_hi - m["weighted_distance"]) / (self.wd_hi - self.wd_lo), 0, 1))
        return round(100.0 * (self.w_cov * cn + (1.0 - self.w_cov) * sn), 1)

    def check_success(self) -> bool:
        if not self._series:
            return False
        m = self._series[-1]
        return (m["mean_z"] < self.tau_z
                and m["coverage"] >= self.tau_cov
                and m["weighted_distance"] <= self.tau_wd)

    def get_score(self) -> float:
        return self._score_from(self._series[-1]) if self._series else 0.0

    def eval_metrics(self) -> dict:
        last = self._series[-1] if self._series else {}
        return {
            "final": last,
            "score": self.get_score(),
            "thresholds": {"cov": self.tau_cov, "weightd": self.tau_wd, "mean_z": self.tau_z},
            "series": self._series,
        }

    def eval_raw(self) -> dict:
        """Raw particle dump for full offline re-scoring (written by eval driver)."""
        return {"init": self._init_particles, "final": self._final_particles,
                "canonical": self._canonical}


@dataclass
class GraspState:
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    left_pos: Optional[np.ndarray] = None
    right_pos: Optional[np.ndarray] = None
    current_kp_dist: Optional[float] = None


class FlattenTshirtTask(BaseTask):
    """Grasp left_shoulder + left_hem (fixed) and do a single fling."""

    GRASP_Z_OFFSET = -0.03
    GRASP_Z_OFFSET_STEP = -0.005
    GRASP_INWARD_OFFSET = 0.02

    LEFT_KP_NAME = "left_shoulder"
    RIGHT_KP_NAME = "left_hem"

    def __init__(self, env, planner_manager: PlannerManager,
                 collector=None, cfg: Optional[DictConfig] = None):
        super().__init__(env, planner_manager, collector, cfg)

        tp = {}
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})

        def _f(name, d):
            return float(tp.get(name, d))

        self.grasp_val = _f("grasp_close_val", 0.0)
        # R13d 判决性实验：合爪时长（= 合爪速度的唯一来源，见
        # closed_loop_skills.robust_bimanual_grasp 的同名参数）。默认 0.8s
        # 与 skill 签名默认值一致 ⇒ 不配这一项时默认路径逐位不变。
        self.grasp_settle_time = _f("grasp_settle_time", 0.8)
        self.lift_h = _f("lift_height", 0.4)
        # Grasp depth below the keypoint, and the IK constraint mask used while
        # descending. Both were hardcoded; they are the two knobs that decide
        # whether the blade is commanded THROUGH the table (kinematic passes
        # through, PD stalls on it -- see docs/local_docs/PD-CONTROL.md).
        # Defaults keep the historical behaviour bit-for-bit.
        self.grasp_z_offset = _f("grasp_z_offset", self.GRASP_Z_OFFSET)
        # T2 台架参数：抓取目标 z 的下限（默认 0.0 = 桌面，逐位不变）。禁入生产 yaml。
        self.grasp_z_floor = _f("grasp_z_floor", 0.0)
        self.grasp_z_offset_step = _f("grasp_z_offset_retry_step", self.GRASP_Z_OFFSET_STEP)
        self.grasp_pose_constraint = list(
            tp.get("grasp_pose_constraint", [1, 1, 1, 0, 0, 0]))
        self.grasp_pose_constraint_frame = str(
            tp.get("grasp_pose_constraint_frame", "world"))
        self.success_thresh = _f("success_thresh", 0.15)
        self.success_coverage = _f("success_coverage", 0.5)

        self.fling_forward_amplitude = _f("fling_forward_amplitude", 0.33)
        self.fling_back_amplitude = _f("fling_back_amplitude", 0.04)
        self.fling_height = _f("fling_height", 0.10)
        self.fling_drag_dist = _f("fling_drag_dist", 0.15)
        self.fling_mid_low_height = _f("fling_mid_low_height", 0.20)
        self.fling_low_height = _f("fling_low_height", 0.03)
        self.fling_vel_scale = _f("fling_vel_scale", 1.0)
        self.fling_accel_scale = _f("fling_accel_scale", 2.0)
        self.lift_y_half_min = _f("lift_y_half_min", 0.17)
        self.lift_y_half_max = _f("lift_y_half_max", 0.30)

        # --- release/retreat separation (docs/GRIPPER-CONTACT.md R11) ---------
        # Historically the jaw command and the go-home retreat are ONE plan, and
        # planner_manager._append_gripper_state ramps the command linearly across
        # the whole trajectory (`g_pos = np.linspace(start, end, T)`). The blades
        # therefore stay clamped on the cloth for the first ~half of the retreat
        # and drag it upward at 6-9 N before the gap finally clears the fabric.
        # Every release-phase reading (3a lag, 3b residual, the release-window
        # lift) is measured through that drag, so it cannot separate "the cloth
        # sticks to the blade" from "the blade carried it away while opening".
        #
        # release_before_home=True inserts a gripper-only step (plan_gripper
        # holds the arm joints stationary) plus a dwell BEFORE move_to_home, so
        # the jaw opens with the arm parked. Any cloth rise after that is real
        # sticking. Default False keeps the historical behaviour untouched.
        self.release_before_home = bool(tp.get("release_before_home", False))
        self.release_open_sec = _f("release_open_sec", 0.5)
        self.release_dwell_sec = _f("release_dwell_sec", 1.0)

        # keypoint indices from asset_meta (single source). fabricbench's
        # env.cloth_keypoints returns {name: index} directly.
        keypoints = self.env.cloth_keypoints or {}
        if self.LEFT_KP_NAME not in keypoints or self.RIGHT_KP_NAME not in keypoints:
            raise ValueError(
                f"cloth asset missing required keypoints "
                f"'{self.LEFT_KP_NAME}' / '{self.RIGHT_KP_NAME}'; "
                f"available: {list(keypoints.keys())}")
        self._kp_a = int(keypoints[self.LEFT_KP_NAME])
        self._kp_b = int(keypoints[self.RIGHT_KP_NAME])

        # canonical (flat rest) mesh: preserve OBJ vertex order because keypoint
        # indices in asset_meta.yaml refer to raw OBJ vertex ids.
        verts, _ = _load_obj_preserve_vertex_order(self.env.cloth_obj_path)
        self._canonical_verts = np.asarray(verts, dtype=np.float64)
        self._canonical_kp_dist = float(
            np.linalg.norm(
                self._canonical_verts[self._kp_a][:2]
                - self._canonical_verts[self._kp_b][:2]))
        self.init_covered_area: Optional[float] = None
        self.grasp_state = GraspState()
        logger.info(
            f"[FlattenTshirt] canonical kp dist "
            f"({self.LEFT_KP_NAME}<->{self.RIGHT_KP_NAME}) = "
            f"{self._canonical_kp_dist:.3f}m  lift_h={self.lift_h}")

    # ------------------------------------------------------------------
    # Grasp targets (called by robust_bimanual_grasp)
    # ------------------------------------------------------------------
    def _make_grasp_pose(self, target_pos: np.ndarray, arm_side: str,
                         z_offset: Optional[float] = None,
                         inward_xy: Optional[np.ndarray] = None) -> np.ndarray:
        pos = target_pos.copy()
        if z_offset is None:
            z_offset = self.grasp_z_offset
        # T2 台架参数（默认 0.0 ⇒ 与原来逐位相同）：允许把抓取目标 z 压到桌面以下，
        # 用来扫刀压深 c。**禁止进生产 yaml**，只在 T 系列台架配置里出现。
        pos[2] = max(pos[2] + z_offset, self.grasp_z_floor)
        if inward_xy is not None:
            pos[0] += float(inward_xy[0])
            pos[1] += float(inward_xy[1])
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        quat = self.skills.calculate_base_to_target_horizontal_quat(base, pos)
        return np.concatenate([pos, quat])

    def _calculate_grasp_targets(
        self, obs: ObservationMessage,
        specific_indices: Optional[Tuple[int, int]] = None,
        z_offset: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        cloth_pos = obs.pcd_cloth_truth

        if specific_indices is None:
            pa = cloth_pos[self._kp_a]
            pb = cloth_pos[self._kp_b]
            # world-y: larger y -> left arm
            if pa[1] >= pb[1]:
                l_idx, r_idx = self._kp_a, self._kp_b
            else:
                l_idx, r_idx = self._kp_b, self._kp_a
            self.grasp_state.current_kp_dist = float(np.linalg.norm(pa[:2] - pb[:2]))
        else:
            l_idx, r_idx = specific_indices
            logger.info(f"[FlattenTshirt] retry with locked L={l_idx}, R={r_idx}")

        centroid_xy = cloth_pos[:, :2].mean(axis=0)

        def _inward(kp_pos: np.ndarray) -> np.ndarray:
            delta = centroid_xy - kp_pos[:2]
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6 or self.GRASP_INWARD_OFFSET <= 0.0:
                return np.zeros(2)
            return delta / norm * self.GRASP_INWARD_OFFSET

        pose_l = self._make_grasp_pose(
            cloth_pos[l_idx], "left", z_offset=z_offset, inward_xy=_inward(cloth_pos[l_idx]))
        pose_r = self._make_grasp_pose(
            cloth_pos[r_idx], "right", z_offset=z_offset, inward_xy=_inward(cloth_pos[r_idx]))

        self.grasp_state.left_idx = l_idx
        self.grasp_state.right_idx = r_idx
        self.grasp_state.left_pos = pose_l
        self.grasp_state.right_pos = pose_r
        logger.info(
            f"[FlattenTshirt] grasp targets: L={l_idx} {pose_l[:3].round(3)}, "
            f"R={r_idx} {pose_r[:3].round(3)}, "
            f"current_kp_dist={self.grasp_state.current_kp_dist} "
            f"(canonical={self._canonical_kp_dist:.3f}m)")
        return pose_l, pose_r, l_idx, r_idx

    # ------------------------------------------------------------------
    # plan_episode
    # ------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        logger.info(f"[FlattenTshirt] Starting episode {episode_id}")

        snaps = getattr(self.env, "initial_cloth_states", None)
        ref = (np.asarray(snaps[0]) if snaps
               else self._canonical_verts)
        self.init_covered_area = get_covered_area_from_particles(ref, 0.005)
        logger.info(f"[FlattenTshirt] init_covered_area={self.init_covered_area:.5f}")

        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        with self.recording("grasp"):
            success = yield from self.skills.robust_bimanual_grasp(
                get_targets_func=self._calculate_grasp_targets,
                max_retries=3,
                close_gripper_value=self.grasp_val,
                grasp_settle_time=self.grasp_settle_time,
                grasp_depth=self.grasp_z_offset,
                retry_depth_step=self.grasp_z_offset_step,
                pose_constraint=self.grasp_pose_constraint,
                pose_constraint_frame=self.grasp_pose_constraint_frame,
                test_lift_height=0.15,
                check_threshold_height=0.08,
                ee_link="tcp_link_bottom",
                ground_check=False,
                velocity_scale=0.4,
            )
            if not success:
                logger.error("[FlattenTshirt] bimanual grasp failed")
                return

            lift_y = float(np.clip(self._canonical_kp_dist / 2.0 - 0.01,
                                   self.lift_y_half_min, self.lift_y_half_max))
            target_left_lift = [0.2, lift_y - 0.02, self.lift_h]
            target_right_lift = [0.2, -lift_y + 0.02, self.lift_h]
            logger.info(f"[FlattenTshirt] lift Y=+/-{lift_y:.3f}m")
            plan_lift = self.skills.move_dual_arm_to(
                left_target_pose=target_left_lift,
                right_target_pose=target_right_lift,
                pose_constraint=[1, 1, 1, 0, 0, 0])
            yield ExecutionStep(plan=plan_lift, settle_time=0.1, description="lift")

            if not self._check_grasp_success():
                logger.error("[FlattenTshirt] grasp check failed after lift")
                return
            logger.success("[FlattenTshirt] grasp + lift completed")

        with self.recording("fling"):
            fling_pos_left = list(target_left_lift)
            fling_pos_right = list(target_right_lift)
            fling_pos_left[1] = self._canonical_kp_dist / 2 - 0.035
            fling_pos_right[1] = -self._canonical_kp_dist / 2 + 0.035

            forward_q = np.array([0.707106781, 0.0, 0.707106781, 0.0])
            target_left_fling = np.concatenate([fling_pos_left, forward_q])
            target_right_fling = np.concatenate([fling_pos_right, forward_q])
            yield from self.skills.fling(
                target_left_fling, target_right_fling,
                fling_forward_amplitude=self.fling_forward_amplitude,
                fling_back_amplitude=self.fling_back_amplitude,
                fling_height=self.fling_height,
                drag_dist=self.fling_drag_dist,
                mid_low_height=self.fling_mid_low_height,
                low_height=self.fling_low_height,
                fling_vel_scale=self.fling_vel_scale,
                fling_accel_scale=self.fling_accel_scale,
            )

            if self.release_before_home:
                plan_open = self.skills.gripper_bimanual_action(
                    1.0, 1.0, duration=self.release_open_sec)
                if not plan_open.success:
                    logger.error("[FlattenTshirt] release-before-home: gripper plan failed")
                    return
                logger.info(f"[FlattenTshirt] release-before-home: open {self.release_open_sec}s "
                            f"+ dwell {self.release_dwell_sec}s, arm parked")
                yield ExecutionStep(plan=plan_open,
                                    settle_time=self.release_dwell_sec,
                                    description="release_open")

            plan_home = self.skills.move_to_home()
            yield ExecutionStep(plan=plan_home, settle_time=0.0, description="go_home")

            self._log_coverage("after_fling")

    # ------------------------------------------------------------------
    # success
    # ------------------------------------------------------------------
    def _check_grasp_success(self) -> bool:
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None:
            return False
        z_mean = float(np.mean(cloth_pos[:, 2]))
        ok = z_mean > self.success_thresh
        logger.debug(f"[FlattenTshirt] grasp check: mean_z={z_mean:.4f}m "
                     f"thresh={self.success_thresh} ok={ok}")
        return ok

    def _log_coverage(self, label: str) -> float:
        curr = self.env.get_cloth_position()
        if curr is None or self.init_covered_area in (None, 0):
            return 0.0
        area = get_covered_area_from_particles(np.asarray(curr), 0.005)
        cov = area / self.init_covered_area
        logger.info(f"[FlattenTshirt] {label} coverage: {cov:.4f}")
        return cov

    def check_success(self) -> bool:
        # v3: delegate to the yaml success_check (coverage>=0.7 AND mean_z_below)
        if self._success_checker is not None:
            ok = bool(self._success_checker.check())
            logger.info(f"[FlattenTshirt] YAML success_check: {ok}")
            return ok
        if self.init_covered_area is None:
            return False
        cov = self._log_coverage("check_success")
        return cov > self.success_coverage
