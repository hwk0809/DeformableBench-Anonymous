"""Flatten silk: 双臂抓角 fling/spread 展平 (easy/hard 难度由 yaml 随机化预设区分).

选点: canonical 角点对过 4 条 filter (reachable / on_surface / occlusion /
multi-layer), 优先近角对, 按模式分支:
  - near (近角对) → fling: pullback → swing → release, 参数对齐真机 fling 弧形
  - far  (远角对) → spread: 高位过中部 (X=0.40, z>0.35, 与 lift X=0.20 错开
        便于策略学习) → 前推到远端低位释放 (X=0.55, z=0.05),
        让被折住的远端铺回去, 暴露近端 → 下个 stage 自然 fling 近端
    (抓远端直接 fling 会展不平: 远端被折后又被甩回近端方向, cloth 翻面/堆叠)

silk 35×35, cloth_centroid≈(0.40,0), 远 corner≈X=0.575, 近 corner≈X=0.225.

Usage:
    .venv/bin/python -m deformablebench.bench.run_task --task flatten_silk_easy --viewer null
"""

import argparse
import os
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import trimesh
from loguru import logger
from omegaconf import DictConfig
from scipy.spatial import ConvexHull, cKDTree

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deformablebench.bench.common.types import ExecutionStep, ObservationMessage
from deformablebench.bench.common.metric_utils import get_covered_area_from_particles
from deformablebench.bench.robot.planning.planner_manager import PlannerManager
from deformablebench.bench.task.base_task import BaseTask


# ======================================================================
# Eval-only scorer (RoboTwin/RoboDojo style, mirrors SilkGraspBimanualTask's
# evaluator half). The eval driver (eval_sim) instantiates this with the spec,
# calls observe() every poll, then — after releasing the grippers and letting
# the cloth settle — calls get_score()/check_success(). It is deliberately
# lightweight (no env/planner in __init__) so importing it is cheap.
# ======================================================================
class FlattenSilkEvaluator:
    """Terminal-settle coverage scorer for flatten_silk.

    Design (locked with user 2026-07-12):
      * Score = normalized coverage (展平程度), NOT a binary threshold.
      * WHEN measured: at the END, after the driver opens both grippers and lets
        the cloth fall & settle on the table — so a cloth still stretched taut in
        the air (large XY projection but not flat) can never be mis-scored as
        "flat". The on-ground guard (max particle z < GROUND_MAX_Z) is the
        anti-misjudgment mechanism.
      * Coverage is RECORDED AT EVERY POLL as a full series (kept in
        eval_metrics), so the score formula can be changed offline / re-derived
        without re-running the episode.

    The driver drives the terminal settle itself (it owns env+bridge, so the
    settle is captured into the saved video); this evaluator only exposes
    ``terminal_settle_ticks`` and ``early_exit=False`` and measures.
    """

    GROUND_MAX_Z = 0.05         # settled == highest particle within 5cm of table
    CLOTH_PARTICLE_RADIUS = 0.005

    # tell the driver: run the FULL episode (no mid-episode early-exit), then
    # release+settle for this many control ticks before the final measurement.
    early_exit = False

    def __init__(self, spec):
        self.spec = spec
        p = spec.task_params
        self.threshold = float(p.get("success_coverage", 0.8))
        settle_sec = float(p.get("terminal_settle_sec", 2.0))
        self.terminal_settle_ticks = max(1, int(round(spec.control_fps * settle_sec)))
        self._series: list[dict] = []
        self._init_area: Optional[float] = None

    def reset_eval(self) -> None:
        self._series = []
        self._init_area = None

    def _reference_area(self, env) -> float:
        """Flat reference area = covered area of the cloth's canonical (rest/flat)
        mesh — exactly what FlattenSilkTask.plan_episode uses. The eval path has no
        scripted task, so ``env.init_covered_area`` is unset; derive it from the
        flat OBJ directly instead of dividing by zero."""
        area = float(getattr(env, "init_covered_area", 0.0) or 0.0)
        if area > 1e-9:
            return area
        verts = np.asarray(trimesh.load(env.cloth_obj_path, process=False,
                                        force="mesh").vertices)
        return float(get_covered_area_from_particles(verts, self.CLOTH_PARTICLE_RADIUS))

    def _measure(self, env) -> Optional[dict]:
        c = env.get_cloth_position()
        if c is None:
            return None
        if self._init_area is None:
            self._init_area = self._reference_area(env)
        area = get_covered_area_from_particles(c, self.CLOTH_PARTICLE_RADIUS)
        cov = area / self._init_area if self._init_area and self._init_area > 1e-9 else 0.0
        return {
            "coverage": round(min(cov, 1.0), 4),   # clamped, used for score
            "coverage_raw": round(cov, 4),         # unclamped (can exceed 1.0)
            "max_z": round(float(c[:, 2].max()), 4),
            "mean_z": round(float(c[:, 2].mean()), 4),
            "std_z": round(float(c[:, 2].std()), 4),
        }

    def observe(self, env) -> None:
        rec = self._measure(env)
        if rec is not None:
            self._series.append(rec)

    def _on_ground(self, rec: dict) -> bool:
        return rec["max_z"] < self.GROUND_MAX_Z

    def check_success(self) -> bool:
        """Success = the FINAL (settled) coverage is on-ground and >= threshold.
        Meaningful only after the terminal settle (last record == settled state)."""
        if not self._series:
            return False
        last = self._series[-1]
        return self._on_ground(last) and last["coverage"] >= self.threshold

    def get_score(self) -> float:
        """Score 0..100 = final settled coverage. The full per-poll series lives
        in eval_metrics so any other formula is recomposable offline."""
        if not self._series:
            return 0.0
        return round(self._series[-1]["coverage"] * 100.0, 1)

    def eval_metrics(self) -> dict:
        onground = [r["coverage"] for r in self._series if self._on_ground(r)]
        return {
            "coverage_final": self._series[-1]["coverage"] if self._series else 0.0,
            "coverage_max": round(max((r["coverage"] for r in self._series), default=0.0), 4),
            "coverage_max_onground": round(max(onground), 4) if onground else 0.0,
            "settled_max_z": self._series[-1]["max_z"] if self._series else None,
            "threshold": self.threshold,
            "coverage_series": self._series,   # full per-poll record (recomposable)
        }


# ======================================================================
# Constants
# ======================================================================
DEFAULT_FLING_CFG: Dict[str, Any] = {
    # 2026-04-26: waypoint 分离 z_max 与 peak-v 位置, 精确对齐真机 fling 弧形.
    # 真机: z_max @ (0.447, 0.434), peak v @ (0.48, 0.32), release @ (0.29, 0.06)
    # 关键: seg2 (swing_top→retract) 是 "前伸+下压" (跟真机 fling 的核心抛物动作一致),
    #       不是 "收回"! 先前的 (0.48, 0.44)→(0.46, 0.35) 把 z_max 和 x_max 合一, 导致
    #       seg2 几乎不动, v_peak 卡在 0° 纯前方向, 布料没被下抛.
    # 测试: v_peak 1.06 (真 1.13), st→rel 1.64s (真 1.68 的 98%), v_late 0.43 (真 0.46), avg_ang -56° (真 -76°)
    "pullback_x": 0.10,     "pullback_z": 0.45,
    "swing_top_x": 0.48,    "swing_top_z": 0.50,   # was (0.48, 0.44); 真机 z_max
    "retract_x": 0.52,      "retract_z": 0.30,     # was (0.46, 0.35); 真机 peak v/x_max
    "release_x": 0.27,      "release_z": 0.05,
    "pullback_vel_scale": 0.4,
    # per-segment: [start→swing_top, swing_top→retract, retract→release]
    # 最后段 15% 限速匹配真机慢减速段 (|a|≈1.5 vs 加速段 ≈3.5)
    "swing_vel_scales": [1.0, 0.10, 0.10],
    "swing_accel_scale": 2.0,
}
FLING_DEBUG_DIR = "/tmp/fling_debug_flatten_silk"

# spread cfg: lift step (单独) + spread sweep (waypoints 2 点) + release
#   lift     (0.50, 0.40): 单独 step, 远端原地升 z (X=0.50 接近抓取位置, 不扯布料)
#   spread sweep waypoints:
#     retract (0.40, 0.30): 拉回中部稍低 (在 fling swing_top X=0.40 同位置停住)
#     down    (0.55, 0.05): 推到远端低位释放
# 跟 fling 区分:
#   spread lift (0.50, 0.40) vs fling lift (0.20, 0.40) — 同 z, X 差 30cm
#   spread retract (0.40, 0.30) vs fling swing_top (0.40, 0.43) — 同 X, z 差 13cm + 速度 (停住 vs 高速)
#   spread down (0.55, 0.05) vs fling release (0.27, 0.05) — 同 z, X 差 28cm
DEFAULT_SPREAD_CFG: Dict[str, Any] = {
    "lift_x": 0.50,        "lift_z": 0.40,
    "lift_vel_scale": 0.4,
    "lift_settle": 0.3,
    "retract_x": 0.30,     "retract_z": 0.4,
    "down_x": 0.55,        "down_z": 0.05,
    # per-segment vel: [start→retract, retract→down], 全程慢 (不像 fling 高速)
    "sweep_vel_scale": [0.4, 0.8],
    "sweep_accel_scale": 1.0,
    "sweep_settle": 0.5,
    "release_duration": 0.3,
    "release_settle": 0.5,
}

# Corner keypoint vertex indices come from the cloth asset_meta.yaml (single
# source), read per-instance in _load_mesh_data as ``self.corner_indices`` /
# ``self.all_corner_set``. Keys: LU/RU/RD/LD (U=远 +X, D=近 -X, L=左 +Y, R=右 -Y).
# Not hardcoded — they must track whichever mesh the asset currently uses.
# canonical 邻接关系 (修正命名后语义清晰):
#   ("LU", "RU") = 远左-远右 = 远端两角对  (cy 跨 → y-span pass) → spread 候选
#   ("LD", "RD") = 近左-近右 = 近端两角对  (cy 跨 → y-span pass) → fling 候选
#   ("LU", "LD") = 远左-近左 = 左侧两角对  (cy 同 +, normal y-span FAIL)
#   ("RU", "RD") = 远右-近右 = 右侧两角对  (cy 同 -, normal y-span FAIL)
ADJACENT_EDGES = [("LU", "RU"), ("LD", "RD"), ("LU", "LD"), ("RU", "RD")]


class InitReachableError(RuntimeError):
    """Raised when _verify_init_reachable assertion fails."""


class FlattenSilkTask(BaseTask):
    """双臂角点选点 + 近端 fling / 远端 spread (高位过中部→前推释放) 展平."""

    GRASP_Z_OFFSET = -0.02           # Boss: 压下去 1.5cm
    GRASP_Z_OFFSET_STEP = -0.005
    # GRASP_INWARD_OFFSET = 0.015       # hull point 沿内法向偏移 (yaml grasp.inward_offset 可覆盖)
    GRASP_INWARD_OFFSET = 0.0
    FILTER_MAX_GRASP_Z = 0.10         # 候选 vtx z 上限, 挡挂点/黏住/穿模

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        grasp_close_val: float = 0.0,
        lift_height: float = 0.4,
        success_coverage: float = 0.95,
        max_flings: int = 3,
        workspace_radius: float = 0.70,
        ws_limit_scale: float = 0.95,
        fling_cfg: Optional[Dict[str, Any]] = None,
        spread_cfg: Optional[Dict[str, Any]] = None,
        # lift/spread/fling 全程 gripper 间距 = cloth_corner_dist × lift_stretch_ratio
        # silk_35 corner_dist ≈ 0.32m. ratio 0.9 → half_w 0.144m, ratio 0.85 → 0.136m
        # 真机 lift 0.311m / silk corner_dist 0.32 ≈ 0.97 (偏紧, 留余地用 0.9 / 0.85)
        lift_stretch_ratio: float = 0.85,
        debug: bool = False,
        show_plot: bool = False,
        collector=None,
        cfg: Optional[DictConfig] = None,
        # grasp modifiers (stagger / slow approach)
        async_close: bool = False,
        async_close_delay_range=(0.1, 0.3),
        slow_approach: bool = True,
        slow_approach_via_distance_range=(0.05, 0.10),
        slow_approach_final_vel_scale: float = 0.08,
        grasp_ground_check: bool = True,
        # yaw_jitter: 每 stage 给 L/R 抓取 target yaw 加一个独立随机 offset,
        # 拓宽手腕 roll 训练分布, 打破 L_j6 被 j1 吃死的 IK 零空间.
        yaw_jitter: bool = False,
        yaw_jitter_range_deg: Tuple[float, float] = (-3.0, 3.0),
        # Filter params
        #   gripper_radius_xy 1.5cm: gripper 半宽, 圆柱 XY 半径
        #   geodesic_safe 2.5cm: canonical 1-ring 覆盖范围 (occlusion 用)
        #   occlusion_z: 上方 2mm ~ 10cm 范围
        #   canonical_spread_thresh: multi-layer 判据 (近邻顶点 canonical XY spread)
        filter_gripper_radius: float = 0.015,
        filter_occlusion_z_range: Tuple[float, float] = (0.002, 0.10),
        filter_geodesic_safe_radius: float = 0.025,
        filter_on_surface_dist: float = 0.03,
        filter_canonical_spread_thresh: float = 0.10,
        degenerate_pair_min_dist: float = 0.15,
        retry_nearby_radius: float = 0.05,
        # far_outward_comp 总开关 (调试用). False 时直接旁路, yaml/默认参数值都不动.
        far_outward_comp_enabled: bool = True,
    ):
        super().__init__(env, planner_manager, collector, cfg)

        tp = {}
        if cfg is not None:
            tp = dict(cfg.get("task_params", {}))

        self.grasp_val = tp.get("grasp_close_val", grasp_close_val)
        self.lift_h = tp.get("lift_height", lift_height)
        self.success_coverage = tp.get("success_coverage", success_coverage)
        self.max_flings = tp.get("max_flings", max_flings)
        self.workspace_radius = tp.get("workspace_radius", workspace_radius)
        self.ws_limit_scale = tp.get("ws_limit_scale", ws_limit_scale)
        self.debug = tp.get("debug", debug)
        self.show_plot = tp.get("show_plot", show_plot)
        # lift/spread/fling Y 间距比例 (Boss tune 0.85 vs 0.9)
        self.lift_stretch_ratio = float(tp.get("lift_stretch_ratio", lift_stretch_ratio))

        # Filter params: 4 条 filter — reachable / occlusion / multi_layer / on_surface
        flt = tp.get("filter", {}) or {}
        self.filter_gripper_radius = float(
            flt.get("gripper_radius_xy", filter_gripper_radius))
        occ_z = flt.get("occlusion_z_range", list(filter_occlusion_z_range))
        self.filter_occ_z_min = float(occ_z[0])
        self.filter_occ_z_max = float(occ_z[1])
        self.filter_geodesic_safe = float(
            flt.get("geodesic_safe_radius", filter_geodesic_safe_radius))
        self.filter_on_surface_dist = float(
            flt.get("on_surface_dist", filter_on_surface_dist))
        self.filter_canonical_spread_thresh = float(
            flt.get("canonical_spread_thresh", filter_canonical_spread_thresh))

        self.degenerate_pair_min_dist = float(
            tp.get("degenerate_pair_min_dist", degenerate_pair_min_dist))
        self.retry_nearby_radius = float(
            tp.get("retry_nearby_radius", retry_nearby_radius))

        # fling cfg: yaml override > default
        fling_from_yaml = tp.get("fling", None)
        if fling_from_yaml:
            merged = dict(DEFAULT_FLING_CFG)
            merged.update(fling_from_yaml)
            self.fling_cfg = merged
        else:
            self.fling_cfg = fling_cfg if fling_cfg is not None else dict(DEFAULT_FLING_CFG)

        # spread cfg (抓远端时的预备动作), yaml override > default
        spread_from_yaml = tp.get("spread", None)
        if spread_from_yaml:
            merged_sp = dict(DEFAULT_SPREAD_CFG)
            merged_sp.update(spread_from_yaml)
            self.spread_cfg = merged_sp
        else:
            self.spread_cfg = spread_cfg if spread_cfg is not None else dict(DEFAULT_SPREAD_CFG)

        if self.debug:
            os.makedirs(FLING_DEBUG_DIR, exist_ok=True)

        # grasp modifiers
        grasp_cfg = tp.get("grasp", {}) or {}
        self.async_close = bool(grasp_cfg.get("async_close", async_close))
        delay_range = grasp_cfg.get("async_close_delay_range", list(async_close_delay_range))
        self.async_close_delay_range = [float(delay_range[0]), float(delay_range[1])]
        self.slow_approach = bool(grasp_cfg.get("slow_approach", slow_approach))
        # via_distance = 沿 grasp -TCP_z 后退多少米 (沿夹爪 approach 轴反向, 与 orientation 同步).
        # 每 ep 在 [lo, hi] 随机, lo==hi 退化成固定值. 替代原 3 个 world-xyz range.
        via_dist_range = grasp_cfg.get("slow_approach_via_distance_range",
                                        list(slow_approach_via_distance_range))
        self.via_distance_range = [float(via_dist_range[0]), float(via_dist_range[1])]
        self.slow_approach_final_vel_scale = float(
            grasp_cfg.get("slow_approach_final_vel_scale", slow_approach_final_vel_scale))
        self.grasp_ground_check = bool(
            grasp_cfg.get("ground_check", grasp_ground_check))

        # yaw_jitter: 每 stage 采一次 L/R 独立 target yaw offset.
        # 目的: 打破 L_j6 被 j1 吃死的 IK 零空间, 拓宽手腕 roll 训练分布.
        # 理论仿真 ±3° 能把 L_j6 approach std 从 3.87° → 7.14° (接近真机 5.35°).
        self.yaw_jitter_enabled = bool(grasp_cfg.get("yaw_jitter", yaw_jitter))
        yaw_range = grasp_cfg.get("yaw_jitter_range_deg", list(yaw_jitter_range_deg))
        self.yaw_jitter_range_rad = [
            float(np.deg2rad(yaw_range[0])), float(np.deg2rad(yaw_range[1]))
        ]
        self.yaw_jitter_L = 0.0
        self.yaw_jitter_R = 0.0
        if self.yaw_jitter_enabled:
            logger.info(
                f"[FlattenSilk] yaw_jitter ENABLED, "
                f"range_deg={yaw_range} (L/R independent per stage)"
            )

        # instance override of class-level grasp offsets
        if "inward_offset" in grasp_cfg:
            self.GRASP_INWARD_OFFSET = float(grasp_cfg["inward_offset"])
        if "z_offset" in grasp_cfg:
            self.GRASP_Z_OFFSET = float(grasp_cfg["z_offset"])
        if "z_offset_retry_step" in grasp_cfg:
            self.GRASP_Z_OFFSET_STEP = float(grasp_cfg["z_offset_retry_step"])

        # far-point outward compensation (reach-based)
        # 目的: 远点因 tilt 让 fabric 在 pad 的 fingertip 后方 ~11mm (θ=30°) 被夹.
        # 方向 = cloth 质心 → 抓点 (cloth-outward), 比 ûₐ 更贴 "抓真实边缘" 的意图.
        # comp = clamp(coeff · max(0, reach - r_thr), 0, max_m)
        far_cfg = grasp_cfg.get("far_outward_comp", {}) or {}
        self.far_comp_r_thr = float(far_cfg.get("r_thr", 0.45))
        self.far_comp_coeff = float(far_cfg.get("coeff", 0.12))
        self.far_comp_max_m = float(far_cfg.get("max_m", 0.020))
        self.far_outward_comp_enabled = bool(
            grasp_cfg.get("far_outward_comp_enabled", far_outward_comp_enabled))

        # mesh + canonical data
        obj_path = self.env.cloth_obj_path
        self._load_mesh_data(obj_path)

        self._episode_id_for_dump = 0
        self._fling_stage = 1

        # grasp 模式 — _select_grasp_pair / retry 路径写入, _do_stage 用来分支 (fling/spread)
        self._last_grasp_mode: str = "near"  # "near" → fling | "far" → spread

        logger.info(
            f"[FlattenSilk] debug={self.debug} ws_radius={self.workspace_radius} "
            f"ws_limit={self.ws_limit():.3f} max_flings={self.max_flings} "
            f"grasp_z_offset={self.GRASP_Z_OFFSET} "
            f"grasp_inward_offset={self.GRASP_INWARD_OFFSET} "
            f"far_outward_comp(enabled={self.far_outward_comp_enabled},"
            f"r_thr={self.far_comp_r_thr},"
            f"coeff={self.far_comp_coeff},max={self.far_comp_max_m}) "
            f"filter(r={self.filter_gripper_radius}, "
            f"occ_z=[{self.filter_occ_z_min},{self.filter_occ_z_max}], "
            f"geodesic={self.filter_geodesic_safe}, "
            f"on_surf={self.filter_on_surface_dist}, "
            f"canonical_spread={self.filter_canonical_spread_thresh}) "
            f"spread(lift=({self.spread_cfg['lift_x']},{self.spread_cfg['lift_z']}), "
            f"retract=({self.spread_cfg['retract_x']},{self.spread_cfg['retract_z']}), "
            f"down=({self.spread_cfg['down_x']},{self.spread_cfg['down_z']})) "
            f"lift_stretch_ratio={self.lift_stretch_ratio} "
            f"corner_dist={self.cloth_corner_dist:.3f}"
        )

    # ------------------------------------------------------------------
    # Setup / mesh
    # ------------------------------------------------------------------
    def _load_mesh_data(self, obj_path: str):
        """加载 canonical mesh, 预计算 geodesic_safe_map (2.5cm canonical 邻居)."""
        mesh = trimesh.load(obj_path, process=False, force="mesh")
        verts = np.asarray(mesh.vertices)
        self._canonical_verts = verts
        self._n_verts = len(verts)

        # Corner keypoint indices from the asset_meta (single source), not hardcoded.
        kp = self.env.cloth_keypoints
        self.corner_indices = {n: kp[f"corner_{n}"] for n in ("LU", "RU", "RD", "LD")}
        self.all_corner_set = set(self.corner_indices.values())

        # 4 个 corner keypoint 之间邻边平均 canonical 距离 (silk_35 ≈ 0.32m).
        # 用作 lift/spread/fling 全程 gripper 间距的基准 (× lift_stretch_ratio).
        # 通用: 任何 cloth 只要定义了 4 keypoints 都能算 (tshirt 也适用).
        edges = []
        for n1, n2 in [("LU", "RU"), ("LD", "RD"), ("LU", "LD"), ("RU", "RD")]:
            p1 = self._canonical_verts[self.corner_indices[n1]]
            p2 = self._canonical_verts[self.corner_indices[n2]]
            edges.append(float(np.linalg.norm(p1[:2] - p2[:2])))
        self.cloth_corner_dist = float(np.mean(edges))

        kdtree = cKDTree(verts)
        self._geodesic_safe_map = kdtree.query_ball_point(
            verts, r=self.filter_geodesic_safe
        )
        avg_safe = np.mean([len(n) for n in self._geodesic_safe_map])
        logger.debug(
            f"[FlattenSilk] mesh: {self._n_verts} verts, "
            f"avg geodesic_safe(r={self.filter_geodesic_safe}m) neighbors: {avg_safe:.1f}"
        )

    def ws_limit(self) -> float:
        return self.workspace_radius * self.ws_limit_scale

    # ------------------------------------------------------------------
    # Filters (4 条: reachable / on_surface / occlusion / multi-layer)
    # ------------------------------------------------------------------
    def _cylinder_residual(
        self, idx: int, cloth_pos: np.ndarray,
        z_min: float, z_max: float,
    ) -> int:
        """指定 idx 周围圆柱 (XY 半径 filter_gripper_radius, Z ∈ [z_min, z_max]) 内
        过滤 geodesic 拓扑邻居后的残差 vertex 数.
        """
        target = cloth_pos[idx]
        diff = cloth_pos - target
        dist_xy = np.linalg.norm(diff[:, :2], axis=1)
        dist_z = diff[:, 2]
        mask = (
            (dist_xy < self.filter_gripper_radius)
            & (dist_z > z_min) & (dist_z < z_max)
        )
        potential = np.where(mask)[0]
        if len(potential) == 0:
            return 0
        safe = set(self._geodesic_safe_map[idx])
        residual = [int(v) for v in potential if int(v) not in safe]
        return len(residual)

    def _check_occlusion_above(self, idx: int, cloth_pos: np.ndarray) -> bool:
        """True = 被上方非拓扑邻居 cloth 遮挡 (抓取通道不通)."""
        n = self._cylinder_residual(
            idx, cloth_pos,
            z_min=self.filter_occ_z_min, z_max=self.filter_occ_z_max,
        )
        return n > 0

    def _check_multi_layer(self, idx: int, cloth_pos: np.ndarray) -> bool:
        """True = idx 周围同一 XY 位置有 cloth 不同区域 (多层堆叠).

        判据: idx 周围 XY filter_gripper_radius 内所有 cloth vertex 的 canonical 坐标,
        max-min spread (XY) > canonical_spread_thresh → 多层.

        为什么不用 z 差: silk 软薄 (2mm), 折叠落地后两层贴平, 世界 z 差 < 2mm 漏检.
        canonical spread 不依赖 z, 靠"空间近邻但 mesh 拓扑远"判断, 更鲁棒.
        """
        p = cloth_pos[idx]
        diff = cloth_pos - p
        dxy = np.linalg.norm(diff[:, :2], axis=1)
        nearby = np.where(dxy < self.filter_gripper_radius)[0]
        if len(nearby) < 2:
            return False
        canon = self._canonical_verts[nearby]
        spread = float(max(
            canon[:, 0].max() - canon[:, 0].min(),
            canon[:, 1].max() - canon[:, 1].min(),
        ))
        return spread > self.filter_canonical_spread_thresh

    def _check_on_surface(self, idx: int, cloth_pos: np.ndarray,
                           hull_pts_xy: Optional[np.ndarray]) -> bool:
        """True = idx 在 cloth 外表面附近 (XY 距最近凸包顶点 < filter_on_surface_dist).

        canonical corner 预内缩 15-20mm, flat 状态下离凸包 ~15-20mm → 通过.
        canonical 翻入 cloth 内部 (折叠)，离凸包 > 3cm → 失败.
        hull 顶点自己在凸包上，距离 = 0 → 必过.
        """
        if hull_pts_xy is None or len(hull_pts_xy) == 0:
            return False
        p = cloth_pos[idx, :2]
        dists = np.linalg.norm(hull_pts_xy - p, axis=1)
        return float(dists.min()) < self.filter_on_surface_dist

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def _build_candidate_set(
        self, cloth_pos: np.ndarray, base_l: np.ndarray, base_r: np.ndarray
    ) -> Tuple[set, Dict[str, Any]]:
        """候选集 = (hull ∪ canonical), 过 4 条 filter:
          1. reachable (min dist to either base < ws_limit)
          2. on_surface (XY 距最近凸包顶点 < filter_on_surface_dist)
          3. no occlusion above
          4. no multi-layer below

        Returns (candidate_set, debug_info).
        """
        info: Dict[str, Any] = {
            "hull_vertices": [],
            "ejected_reason": {},  # idx -> reason string
        }

        try:
            hull = ConvexHull(cloth_pos[:, :2])
            hull_idx = np.asarray(hull.vertices, dtype=int)
            hull_pts_xy = cloth_pos[hull_idx, :2]
            info["hull_vertices"] = hull_idx.tolist()
        except Exception as e:
            logger.warning(f"[candidate] ConvexHull failed: {e}")
            hull_idx = np.array([], dtype=int)
            hull_pts_xy = None
            info["hull_vertices"] = []

        # 候选基础集 = hull vertex ∪ canonical corner (canonical 预内缩, 不在 hull 里)
        candidate_base = set(hull_idx.tolist()) | self.all_corner_set

        ws_limit = self.ws_limit()
        candidate = set()
        for v in candidate_base:
            v_int = int(v)
            pos = cloth_pos[v_int]

            # 0. on-ground: 挡住被夹爪挂起/黏住/穿模的 vtx (retry 场景)
            if pos[2] > self.FILTER_MAX_GRASP_Z:
                info["ejected_reason"][v_int] = f"airborne(z={pos[2]:.3f})"
                continue

            # 1. reachable: 至少一只臂够得到
            dl = float(np.linalg.norm(pos - base_l))
            dr = float(np.linalg.norm(pos - base_r))
            if min(dl, dr) >= ws_limit:
                info["ejected_reason"][v_int] = (
                    f"unreachable(dl={dl:.3f},dr={dr:.3f})"
                )
                continue

            # 2. on_surface: 距最近凸包顶点 < 3cm
            #    canonical 翻入 cloth 内部会失败; hull 自身必过 (dist = 0)
            if not self._check_on_surface(v_int, cloth_pos, hull_pts_xy):
                info["ejected_reason"][v_int] = "off_surface"
                continue

            # 3. occlusion above
            if self._check_occlusion_above(v_int, cloth_pos):
                info["ejected_reason"][v_int] = "occluded_above"
                continue

            # 4. multi-layer below
            if self._check_multi_layer(v_int, cloth_pos):
                info["ejected_reason"][v_int] = "multi_layer_below"
                continue

            candidate.add(v_int)

        info["candidate_count"] = len(candidate)
        return candidate, info

    # ------------------------------------------------------------------
    # Pair selection: normal (canonical corner) + degenerate (hull)
    # ------------------------------------------------------------------
    def _pair_reachable(
        self, idx_a: int, idx_b: int, cloth_pos: np.ndarray,
        base_l: np.ndarray, base_r: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """按 world-y 分配左右臂后返回 (dl, dr), 都 reachable 返回 tuple, 否则 None."""
        pa, pb = cloth_pos[idx_a], cloth_pos[idx_b]
        if pa[1] >= pb[1]:
            dl = float(np.linalg.norm(pa - base_l))
            dr = float(np.linalg.norm(pb - base_r))
        else:
            dl = float(np.linalg.norm(pb - base_l))
            dr = float(np.linalg.norm(pa - base_r))
        ws_limit = self.ws_limit()
        if dl > ws_limit or dr > ws_limit:
            return None
        return dl, dr

    def _assign_left_right(
        self, idx_a: int, idx_b: int, cloth_pos: np.ndarray
    ) -> Tuple[int, int]:
        """按 world-y 分左右: y 大 → 左."""
        if cloth_pos[idx_a][1] >= cloth_pos[idx_b][1]:
            return idx_a, idx_b
        return idx_b, idx_a

    def _select_grasp_pair(
        self, cloth_pos: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, str]], Dict[str, Any]]:
        """选点: canonical 邻接 pair 写死, 优先近端 fling, 远端 spread 兜底.

        near = (LD, RD) → fling      far = (LU, RU) → spread
        silk_35 ±10° rotation 鲁棒. 大旋转/翻面/大折叠不鲁棒 (当前任务不需要).
        """
        base_l = np.array(self.env.left_base[:3])
        base_r = np.array(self.env.right_base[:3])
        candidate, cand_info = self._build_candidate_set(cloth_pos, base_l, base_r)

        near_pair = (self.corner_indices["LD"], self.corner_indices["RD"])
        far_pair = (self.corner_indices["LU"], self.corner_indices["RU"])

        debug_info: Dict[str, Any] = {
            "candidate_info": cand_info,
            "mode": "none",
            "near_pair": [int(near_pair[0]), int(near_pair[1])],
            "far_pair": [int(far_pair[0]), int(far_pair[1])],
        }

        for (ia, ib), mode in [(near_pair, "near"), (far_pair, "far")]:
            if ia not in candidate or ib not in candidate:
                logger.info(f"[grasp-select] {mode} pair filter rejected (L={ia} R={ib})")
                continue
            if self._pair_reachable(ia, ib, cloth_pos, base_l, base_r) is None:
                logger.info(f"[grasp-select] {mode} pair unreachable (L={ia} R={ib})")
                continue
            l_idx, r_idx = self._assign_left_right(ia, ib, cloth_pos)
            debug_info["mode"] = mode
            debug_info["picked"] = {"l_idx": int(l_idx), "r_idx": int(r_idx), "mode": mode}
            logger.info(f"[grasp-select] mode={mode} L={l_idx} R={r_idx}")
            return (l_idx, r_idx, mode), debug_info

        debug_info["mode"] = "failed"
        logger.warning("[grasp-select] both near and far pair unavailable")
        return None, debug_info

    # ------------------------------------------------------------------
    # Retry partner
    # ------------------------------------------------------------------
    def _find_retry_partner(
        self, cloth_pos: np.ndarray, keep_idx: int, failed_old_idx: int,
        candidate: set,
    ) -> Optional[int]:
        """在 candidate 里找 keep_idx 的替代 partner.

        过滤:
          - 候选必须在 keep_idx 的 world-y 对侧
          - 距 keep_idx > degenerate_pair_min_dist (fling 撑开)
          - 距 failed_old_idx < retry_nearby_radius (retry 不跳远)
        排序: argmax(midpoint.x)
        """
        base_l = np.array(self.env.left_base[:3])
        base_r = np.array(self.env.right_base[:3])
        p_keep = cloth_pos[keep_idx]
        p_fail = cloth_pos[failed_old_idx]

        best_score = -np.inf
        best_idx = None
        for v in candidate:
            if v == keep_idx or v == failed_old_idx:
                continue
            pv = cloth_pos[v]
            # Y-span: keep 与 v 必须 world y 异号 (一臂一边, 避免挂帘子)
            if p_keep[1] * pv[1] >= 0:
                continue
            # 距 keep > degenerate_pair_min_dist (fling 撑开)
            if np.linalg.norm(pv - p_keep) < self.degenerate_pair_min_dist:
                continue
            # 距 fail < retry_nearby_radius (retry 不跳远)
            if np.linalg.norm(pv - p_fail) > self.retry_nearby_radius:
                continue
            # 两端 reachable
            reach = self._pair_reachable(keep_idx, v, cloth_pos, base_l, base_r)
            if reach is None:
                continue
            mid_x = float((p_keep[0] + pv[0]) / 2)
            if mid_x > best_score:
                best_score = mid_x
                best_idx = int(v)
        return best_idx

    # ------------------------------------------------------------------
    # Grasp pose construction
    # ------------------------------------------------------------------
    def _compute_far_outward_comp(
        self, target_xy: np.ndarray, base_xy: np.ndarray,
        cloth_centroid_xy: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """远点 outward 补偿 (cloth 质心 → 抓点方向).

        Returns (dxy, comp_magnitude). dxy = comp·outward_dir, 应加到 target[:2].
        reach = 抓点到臂 base 的 XY 距离. reach < r_thr 时不补偿.
        """
        if not self.far_outward_comp_enabled:
            return np.zeros(2), 0.0
        reach = float(np.linalg.norm(target_xy - base_xy))
        comp = self.far_comp_coeff * max(0.0, reach - self.far_comp_r_thr)
        comp = min(comp, self.far_comp_max_m)
        if comp <= 1e-6:
            return np.zeros(2), 0.0
        outward = target_xy - cloth_centroid_xy
        on = float(np.linalg.norm(outward))
        if on < 1e-6:
            return np.zeros(2), 0.0
        outward /= on
        return outward * comp, comp

    def _make_grasp_pose(
        self, target_pos: np.ndarray, arm_side: str,
        z_offset: Optional[float] = None,
        inward_normal: Optional[np.ndarray] = None,
        cloth_centroid_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pos = target_pos.copy()
        if inward_normal is not None and self.GRASP_INWARD_OFFSET > 0:
            pos[:2] += inward_normal[:2] * self.GRASP_INWARD_OFFSET
        base = self.env.left_base if arm_side == "left" else self.env.right_base
        if cloth_centroid_xy is not None:
            dxy, comp = self._compute_far_outward_comp(
                pos[:2], np.array(base[:2]), cloth_centroid_xy)
            if comp > 0:
                reach = float(np.linalg.norm(pos[:2] - np.array(base[:2])))
                logger.info(
                    f"[far_outward_comp] {arm_side} reach={reach:.3f} "
                    f"comp={comp*1000:.1f}mm dxy=({dxy[0]*1000:+.1f},{dxy[1]*1000:+.1f})"
                )
                pos[:2] += dxy
        if z_offset is None:
            z_offset = self.GRASP_Z_OFFSET
        pos[2] = max(pos[2] + z_offset, 0.000)
        yaw_jit = self.yaw_jitter_L if arm_side == "left" else self.yaw_jitter_R
        quat = self.skills.calculate_base_to_target_horizontal_quat(base, pos, yaw_jitter=yaw_jit)
        return np.concatenate([pos, quat])

    def _build_grasp_result(
        self, l_idx: int, r_idx: int, cloth_pos: np.ndarray,
        z_offset: Optional[float],
    ):
        target_l = cloth_pos[l_idx].copy()
        target_r = cloth_pos[r_idx].copy()

        centroid_xy = cloth_pos[:, :2].mean(axis=0)

        # Canonical corner: 直接抓 (index 已预内缩 15-20mm)
        # 非 canonical (退化/retry 可能选到的 hull 点): 沿 hull 内法向偏 15mm
        if l_idx in self.all_corner_set and r_idx in self.all_corner_set:
            pose_l = self._make_grasp_pose(
                target_l, "left", z_offset=z_offset,
                cloth_centroid_xy=centroid_xy)
            pose_r = self._make_grasp_pose(
                target_r, "right", z_offset=z_offset,
                cloth_centroid_xy=centroid_xy)
        else:
            try:
                hull = ConvexHull(cloth_pos[:, :2])
                normals = self._compute_hull_normals(cloth_pos, hull.vertices)
            except Exception:
                normals = {}
            pose_l = self._make_grasp_pose(
                target_l, "left", z_offset=z_offset,
                inward_normal=normals.get(l_idx),
                cloth_centroid_xy=centroid_xy)
            pose_r = self._make_grasp_pose(
                target_r, "right", z_offset=z_offset,
                inward_normal=normals.get(r_idx),
                cloth_centroid_xy=centroid_xy)

        logger.info(
            f"[GraspTargets] L={l_idx} {pose_l[:3]}, R={r_idx} {pose_r[:3]}"
        )
        return pose_l, pose_r, l_idx, r_idx

    @staticmethod
    def _compute_hull_normals(cloth_pos, hull_vertices) -> Dict[int, np.ndarray]:
        hull_pts = cloth_pos[hull_vertices, :2]
        n = len(hull_vertices)
        normals: Dict[int, np.ndarray] = {}
        for i in range(n):
            e1 = hull_pts[i] - hull_pts[(i - 1) % n]
            e2 = hull_pts[(i + 1) % n] - hull_pts[i]
            n1 = np.array([-e1[1], e1[0]])
            n2 = np.array([-e2[1], e2[0]])
            l1, l2 = np.linalg.norm(n1), np.linalg.norm(n2)
            if l1 > 1e-8:
                n1 /= l1
            if l2 > 1e-8:
                n2 /= l2
            avg = (n1 + n2) / 2
            al = np.linalg.norm(avg)
            if al > 1e-8:
                avg /= al
            normals[int(hull_vertices[i])] = avg
        return normals

    # ------------------------------------------------------------------
    # Calculate grasp targets (called by robust_bimanual_grasp)
    # ------------------------------------------------------------------
    def _calculate_grasp_targets(
        self,
        obs: ObservationMessage,
        specific_indices: Optional[Tuple[int, int]] = None,
        z_offset: Optional[float] = None,
        l_ok: bool = False,
        r_ok: bool = False,
    ):
        cloth_pos = obs.pcd_cloth_truth

        if specific_indices is not None:
            base_l = np.array(self.env.left_base[:3])
            base_r = np.array(self.env.right_base[:3])
            candidate, _ = self._build_candidate_set(cloth_pos, base_l, base_r)

            old_l, old_r = specific_indices
            if l_ok and not r_ok:
                new_r = self._find_retry_partner(
                    cloth_pos, keep_idx=old_l, failed_old_idx=old_r,
                    candidate=candidate)
                if new_r is not None:
                    logger.info(
                        f"[GraspTargets] Retry (L ok): keep L={old_l}, nearby R={new_r}"
                    )
                    # retry partner 在原 corner 附近, mode 沿用上次 (near/far)
                    return self._build_grasp_result(old_l, new_r, cloth_pos, z_offset)
            elif r_ok and not l_ok:
                new_l = self._find_retry_partner(
                    cloth_pos, keep_idx=old_r, failed_old_idx=old_l,
                    candidate=candidate)
                if new_l is not None:
                    logger.info(
                        f"[GraspTargets] Retry (R ok): keep R={old_r}, nearby L={new_l}"
                    )
                    return self._build_grasp_result(new_l, old_r, cloth_pos, z_offset)
            logger.info(
                f"[GraspTargets] Retry (L_ok={l_ok}, R_ok={r_ok}): full reselect"
            )

        # 全量选点
        result, debug_info = self._select_grasp_pair(cloth_pos)
        if self.debug:
            self._visualize_grasp(cloth_pos, result, debug_info)
        if result is None:
            raise ValueError("No valid grasp pair found")
        l_idx, r_idx, mode = result
        # mode ∈ {"near","far"} → _do_fling_stage 直接据此分支 (fling/spread)
        self._last_grasp_mode = mode
        return self._build_grasp_result(l_idx, r_idx, cloth_pos, z_offset)

    # ------------------------------------------------------------------
    # Fling execution
    # ------------------------------------------------------------------
    def _compute_fling_waypoints(self, start_l, start_r):
        cfg = self.fling_cfg
        def _set_xz(pose, x, z):
            out = pose.copy(); out[0] = x; out[2] = z
            return out
        pb_l = _set_xz(start_l, cfg["pullback_x"], cfg["pullback_z"])
        pb_r = _set_xz(start_r, cfg["pullback_x"], cfg["pullback_z"])
        st_l = _set_xz(start_l, cfg["swing_top_x"], cfg["swing_top_z"])
        st_r = _set_xz(start_r, cfg["swing_top_x"], cfg["swing_top_z"])
        rt_l = _set_xz(start_l, cfg["retract_x"], cfg["retract_z"])
        rt_r = _set_xz(start_r, cfg["retract_x"], cfg["retract_z"])
        rl_l = _set_xz(start_l, cfg["release_x"], cfg["release_z"])
        rl_r = _set_xz(start_r, cfg["release_x"], cfg["release_z"])
        return pb_l, pb_r, [st_l, rt_l, rl_l], [st_r, rt_r, rl_r]

    def _execute_fling(self, start_l, start_r):
        """2026-04-25 单次 plan + per-segment velocity scaling.

        关键机制: toppra JointVelocityConstraintVarying 让 vlim 随 path param s 分段变化.
        - 前两段 scale=1.0 全速 fling (v_peak ~1.07 m/s, 接近真机 1.13)
        - 最后段 scale=0.2 限速 → toppra 自然慢减速, 无停顿 (v@retract ≈ 0.46 非零)
        - st→release 1.36s (真机 1.68s 的 81%, vs 原 0.86s 提升 58%)
        """
        cfg = self.fling_cfg
        pb_l, pb_r, seq_l, seq_r = self._compute_fling_waypoints(start_l, start_r)

        # 1) pullback (单段)
        p1 = self.skills.move_dual_arm_to(
            left_target_pose=pb_l, right_target_pose=pb_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom", velocity_scale=cfg["pullback_vel_scale"],
        )
        if not p1.success:
            logger.error("[fling] pullback failed"); return
        yield ExecutionStep(plan=p1, settle_time=0.0,
                            description=f"fling{self._fling_stage}_pullback")

        # 2) swing: 单次 plan 过 3 个 waypoint, per-segment vel scaling
        swing_vel_scales = cfg.get("swing_vel_scales", [1.0, 1.0, 0.2])
        p2 = self.skills.move_dual_arm_to(
            left_target_pose=seq_l, right_target_pose=seq_r,
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0],
                             [1, 1, 0.5, 0, 0, 0],   # retract z 轻软 (允许 IK 微调)
                             [1, 1, 1, 0, 0, 0]],
            interpolation_mode=["joint", "joint", "cartesian"],
            steps_per_segment=3,
            ee_link="tcp_link_bottom",
            velocity_scale=swing_vel_scales,
            acceleration_scale=cfg["swing_accel_scale"],
        )
        if not p2.success:
            logger.error("[fling] swing failed"); return
        yield ExecutionStep(plan=p2, settle_time=0.5,
                            description=f"fling{self._fling_stage}_swing")

    # ------------------------------------------------------------------
    # spread (抓远端时的预备动作)
    # ------------------------------------------------------------------
    def _execute_spread(self, half_w: float):
        """抓远端 stage: lift (单独) → spread sweep (waypoints) → release."""
        cfg = self.spread_cfg
        forward_q = np.array([0.707106781, 0.0, 0.707106781, 0.0])

        def _wp(x, z):
            return (
                np.concatenate([[x, half_w, z], forward_q]),
                np.concatenate([[x, -half_w, z], forward_q]),
            )

        # 1. lift step (单独 plan): 远端原地升 z (X 跟抓取接近, 不扯)
        lift_l, lift_r = _wp(cfg["lift_x"], cfg["lift_z"])
        p1 = self.skills.move_dual_arm_to(
            left_target_pose=lift_l, right_target_pose=lift_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=cfg["lift_vel_scale"],
        )
        if not p1.success:
            logger.error("[spread] lift failed"); return
        yield ExecutionStep(plan=p1, settle_time=cfg["lift_settle"],
                            description=f"spread{self._fling_stage}_lift")

        # 2. spread sweep (waypoints, 跟 fling _execute_fling swing 同形式)
        retract_l, retract_r = _wp(cfg["retract_x"], cfg["retract_z"])
        down_l,    down_r    = _wp(cfg["down_x"],    cfg["down_z"])
        p2 = self.skills.move_dual_arm_to(
            left_target_pose=[retract_l, down_l],
            right_target_pose=[retract_r, down_r],
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0]] * 2,
            interpolation_mode=["joint", "cartesian"],
            steps_per_segment=3,
            ee_link="tcp_link_bottom",
            velocity_scale=cfg["sweep_vel_scale"],
            acceleration_scale=cfg["sweep_accel_scale"],
        )
        if not p2.success:
            logger.error("[spread] sweep failed"); return
        yield ExecutionStep(plan=p2, settle_time=cfg["sweep_settle"],
                            description=f"spread{self._fling_stage}_sweep")

        # 3. release (spread 没有甩飞动量, cloth 不会自动脱手)
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(
                1.0, 1.0, duration=cfg["release_duration"]),
            settle_time=cfg["release_settle"],
            description=f"spread{self._fling_stage}_release",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _move_to_init_pose(self):
        # 回家用干净 base, 不带 episode 初始化噪声 (避免噪声贯穿整个 ep)
        q_left = getattr(self.env, 'home_q_left_clean',
                         getattr(self.env, 'home_q_left', np.zeros(6)))
        q_right = getattr(self.env, 'home_q_right_clean',
                          getattr(self.env, 'home_q_right', np.zeros(6)))
        return self.skills.move_dual_arm_to(
            left_target_pose=q_left, right_target_pose=q_right,
            goal_type="joint", left_gripper=1.0, right_gripper=1.0,
        )

    def _release_to_center(self, stage_name: str, reason: str,
                           l_holding: bool = False, r_holding: bool = False):
        logger.info(
            f"[{stage_name}] {reason}, L_holding={l_holding} R_holding={r_holding}"
        )
        l_ok = l_holding
        r_ok = r_holding

        if not l_ok and not r_ok:
            yield ExecutionStep(
                plan=self.skills.gripper_bimanual_action(1.0, 1.0, duration=0.3),
                settle_time=0.5, description=f"{stage_name}_release_both")
            yield ExecutionStep(plan=self._move_to_init_pose(),
                                settle_time=0.5, description=f"{stage_name}_home")
            return

        center_l = [0.45, 0.05, 0.10] if l_ok else None
        center_r = [0.45, -0.05, 0.10] if r_ok else None
        # 回家用干净 base, 不带 episode 初始化噪声
        q_home_left = getattr(self.env, 'home_q_left_clean',
                              getattr(self.env, 'home_q_left', np.zeros(6)))
        q_home_right = getattr(self.env, 'home_q_right_clean',
                               getattr(self.env, 'home_q_right', np.zeros(6)))

        if l_ok and r_ok:
            plan = self.skills.move_dual_arm_to(
                left_target_pose=center_l, right_target_pose=center_r,
                pose_constraint=[1, 1, 1, 0, 0, 0], velocity_scale=0.3)
        elif l_ok:
            plan = self.skills.move_dual_arm_to(
                left_target_pose=center_l, right_target_pose=q_home_right,
                left_goal_type="pose", right_goal_type="joint",
                pose_constraint=[1, 1, 1, 0, 0, 0], right_gripper=1.0,
                velocity_scale=0.3)
        else:
            plan = self.skills.move_dual_arm_to(
                left_target_pose=q_home_left, right_target_pose=center_r,
                left_goal_type="joint", right_goal_type="pose",
                pose_constraint=[1, 1, 1, 0, 0, 0], left_gripper=1.0,
                velocity_scale=0.3)
        if plan.success:
            yield ExecutionStep(plan=plan, settle_time=0.3,
                                description=f"{stage_name}_to_center")
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(1.0, 1.0, duration=0.3),
            settle_time=1.5, description=f"{stage_name}_release")
        yield ExecutionStep(plan=self._move_to_init_pose(),
                            settle_time=0.5, description=f"{stage_name}_home")

    def _log_coverage(self, label: str) -> float:
        curr = self.env.get_cloth_position()
        if curr is None:
            return 0.0
        area = get_covered_area_from_particles(np.array(curr), 0.005)
        cov = min(area / self.init_covered_area, 1.0) if self.init_covered_area > 0 else 0.0
        logger.info(f"[FlattenSilk] {label} coverage: {cov:.4f}")
        self.last_coverage = cov
        return cov

    def check_success(self) -> bool:
        if self._success_checker is not None:
            threshold = self._success_checker.cfg.get("threshold", self.success_coverage)
        else:
            threshold = self.success_coverage
        cov = self._log_coverage("check_success")
        return cov >= threshold

    # ------------------------------------------------------------------
    # Init reachability verification
    # ------------------------------------------------------------------
    def _verify_init_reachable(self, cloth_pos: np.ndarray):
        """确保初始化后至少有一条 Y-SPAN adjacent canonical edge 可达.

        Y-SPAN: 两端 world y 异号. 只考虑 y-span edge, 和主选点规则一致.
        失败 raise InitReachableError → 上层 skip episode.
        """
        base_l = np.array(self.env.left_base[:3])
        base_r = np.array(self.env.right_base[:3])
        ws_limit = self.ws_limit()

        best_mid_x = -np.inf
        best_edge = None
        best_reach = None
        for name_a, name_b in ADJACENT_EDGES:
            ia, ib = self.corner_indices[name_a], self.corner_indices[name_b]
            pa, pb = cloth_pos[ia], cloth_pos[ib]
            # 只关心 y-span (和 _select_grasp_pair 的规则一致)
            if pa[1] * pb[1] >= 0:
                continue
            mid_x = float((pa[0] + pb[0]) / 2)
            reach = self._pair_reachable(ia, ib, cloth_pos, base_l, base_r)
            if reach is not None and mid_x > best_mid_x:
                best_mid_x = mid_x
                best_edge = (name_a, name_b)
                best_reach = reach

        if best_reach is None:
            msg = (f"[verify_init] no reachable y-span canonical edge under "
                   f"ws_limit={ws_limit:.3f}")
            logger.error(msg)
            raise InitReachableError(msg)

        dl, dr = best_reach
        logger.info(
            f"[verify_init] far y-span edge {best_edge} mid.x={best_mid_x:.3f} "
            f"dl={dl:.3f} dr={dr:.3f} ws_limit={ws_limit:.3f} OK"
        )

    # ------------------------------------------------------------------
    # Visualization (debug=True 时)
    # ------------------------------------------------------------------
    def _visualize_grasp(self, cloth_pos, result, debug_info):
        try:
            import matplotlib
            matplotlib.use("TkAgg" if self.show_plot else "Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        stage = self._fling_stage
        mode = debug_info.get("mode", "none")
        cand_info = debug_info.get("candidate_info", {})
        hull_vertices = cand_info.get("hull_vertices", [])
        ejected = cand_info.get("ejected_reason", {})
        candidate_count = cand_info.get("candidate_count", 0)

        fig, ax = plt.subplots(figsize=(10, 8))

        # 所有 cloth 点 (浅灰)
        step = max(1, len(cloth_pos) // 2000)
        ax.scatter(cloth_pos[::step, 0], cloth_pos[::step, 1],
                   c="lightgray", s=0.5, alpha=0.2, zorder=1)

        # 凸包多边形
        if hull_vertices:
            hull_xy = cloth_pos[hull_vertices, :2]
            ctr = hull_xy.mean(axis=0)
            order = np.argsort(np.arctan2(hull_xy[:, 1] - ctr[1], hull_xy[:, 0] - ctr[0]))
            poly = plt.Polygon(hull_xy[order], fill=False, edgecolor="dodgerblue",
                               linewidth=1.5, zorder=3)
            ax.add_patch(poly)

        # candidate_base 所有点: 绿=candidate, 红=multi_layer 踢, 灰=unreachable 踢
        cand_base = set(hull_vertices) | self.all_corner_set
        for v in cand_base:
            p = cloth_pos[v]
            reason = ejected.get(int(v))
            if reason is None:
                color = "limegreen"
            elif "multi_layer" in reason or "occluded" in reason:
                color = "red"
            elif "off_surface" in reason:
                color = "orange"
            else:
                color = "gray"
            size = 40 if v in self.all_corner_set else 18
            ax.scatter([p[0]], [p[1]], c=color, s=size, zorder=5,
                       edgecolors="black", linewidths=0.3)

        # 4 canonical corner 标注
        for name, idx in self.corner_indices.items():
            p = cloth_pos[idx]
            ok = idx not in ejected
            ax.scatter([p[0]], [p[1]], c="gold" if ok else "black",
                       s=120, marker="*", zorder=8,
                       edgecolors="darkorange", linewidths=0.6)
            ax.annotate(name, (p[0], p[1]), fontsize=8, fontweight="bold",
                        color="black", ha="center",
                        xytext=(0, 9), textcoords="offset points")

        # 选中 pair
        if result is not None:
            l_idx, r_idx, _mode = result
            for idx, arm in [(l_idx, "LEFT"), (r_idx, "RIGHT")]:
                p = cloth_pos[idx]
                ax.scatter([p[0]], [p[1]], c="magenta", s=260, marker="o",
                           edgecolors="black", linewidths=2.0, zorder=20)
                ax.annotate(f"{arm}\nz={p[2]:.3f}", (p[0], p[1]),
                            fontsize=8, fontweight="bold", ha="center", va="top",
                            xytext=(0, -12), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                      ec="black", alpha=0.85))
            pl, pr = cloth_pos[l_idx], cloth_pos[r_idx]
            ax.plot([pl[0], pr[0]], [pl[1], pr[1]], "m-",
                    linewidth=2.5, zorder=15)
            mid = (pl + pr) / 2
            ax.annotate(f"mid.x={mid[0]:.3f}\ndist={np.linalg.norm(pl-pr):.3f}",
                        (mid[0], mid[1]), fontsize=8, ha="center",
                        xytext=(0, 6), textcoords="offset points")

        # Base + workspace
        for bx, by, arm in [(0, 0.3, "base_L"), (0, -0.3, "base_R")]:
            ax.scatter([bx], [by], c="black", s=80, marker="^", zorder=10)
            ax.annotate(arm, (bx, by), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.add_patch(plt.Circle((bx, by), self.ws_limit(), fill=False,
                                    ls="--", color="gray", alpha=0.4))

        title = f"FlattenSilk Fling{stage}  mode={mode}  hull={len(hull_vertices)}  candidate={candidate_count}"
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        plt.tight_layout()

        save_path = os.path.join(
            FLING_DEBUG_DIR,
            f"flatten_grasp_ep{self._episode_id_for_dump:04d}_fling{stage}.png",
        )
        fig.savefig(save_path, dpi=150)
        logger.info(f"[grasp-vis] saved {save_path}")
        if self.show_plot:
            plt.show(block=False)
            plt.pause(0.3)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Do fling stage
    # ------------------------------------------------------------------
    def _do_fling_stage(self, stage_name: str):
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        close_delay = 0.0
        if self.async_close:
            lo, hi = self.async_close_delay_range
            close_delay = float(np.random.uniform(lo, hi))
            logger.info(f"[{stage_name}] async_close delay={close_delay:.3f}s")

        via_distance = 0.0
        if self.slow_approach:
            via_distance = float(np.random.uniform(*self.via_distance_range))
            logger.info(
                f"[{stage_name}] slow_approach via_distance={via_distance:.3f}m "
                f"(沿 grasp -TCP_z 后退)"
            )

        # yaw_jitter: L/R 独立采样, 每 stage 一次 (retry 沿用同一个值)
        if self.yaw_jitter_enabled:
            lo, hi = self.yaw_jitter_range_rad
            self.yaw_jitter_L = float(np.random.uniform(lo, hi))
            self.yaw_jitter_R = float(np.random.uniform(lo, hi))
            logger.info(
                f"[{stage_name}] yaw_jitter L={np.rad2deg(self.yaw_jitter_L):+.2f}° "
                f"R={np.rad2deg(self.yaw_jitter_R):+.2f}°"
            )
        else:
            self.yaw_jitter_L = 0.0
            self.yaw_jitter_R = 0.0

        success = yield from self.skills.robust_bimanual_grasp(
            get_targets_func=self._calculate_grasp_targets,
            max_retries=3,
            close_gripper_value=self.grasp_val,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            pose_constraint=[1, 1, 1, 1, 0, 1],
            pose_constraint_frame="local",
            test_lift_height=0.15,
            check_threshold_height=0.08,
            check_threshold_dist=0.02,    # 夹持态实测 min_dist 0.013-0.016 (TCP/EEF 帧差,
                                          # to-optimize §7), 0.015 仍边界误判 → 2cm
            ee_link="tcp_link_bottom",
            grasp_settle_time=0.6,
            velocity_scale=0.15,
            async_close=self.async_close,
            async_close_delay=close_delay,
            slow_approach=self.slow_approach,
            slow_approach_via_distance=via_distance,
            slow_approach_final_vel_scale=self.slow_approach_final_vel_scale,
            ground_check=self.grasp_ground_check,
        )
        if not success:
            logger.warning(
                f"[{stage_name}] Robust bimanual grasp failed, proceeding to lift check"
            )

        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        actual_l, actual_r = self.skills.get_actual_grasp_indices(obs=obs)
        width = float(np.linalg.norm(
            self._canonical_verts[actual_l] - self._canonical_verts[actual_r]))
        logger.info(
            f"[{stage_name}] actual grasp: L_vtx={actual_l} R_vtx={actual_r}, "
            f"mesh_dist={width:.3f}m"
        )

        if width < 0.1:
            width = 0.32
        # half_w = cloth_corner_dist × lift_stretch_ratio / 2, round 到 3 位小数 (mm 精度)
        # canonical corner pair: 用 fixed_half_w (Boss tune ratio 0.85/0.9)
        # retry hull partner: 用 mesh_dist/2 兜底防扯, cap [0.10, fixed_half_w]
        fixed_half_w = round(self.cloth_corner_dist * self.lift_stretch_ratio / 2, 3)
        is_canonical_pair = (actual_l in self.all_corner_set) and (actual_r in self.all_corner_set)
        if is_canonical_pair:
            lift_half_w = fixed_half_w
        else:
            lift_half_w = round(max(min(width / 2, fixed_half_w), 0.10), 3)
        logger.info(
            f"[{stage_name}] mode={self._last_grasp_mode} canonical={is_canonical_pair} "
            f"mesh_dist={width:.3f} corner_dist={self.cloth_corner_dist:.3f} "
            f"ratio={self.lift_stretch_ratio} → lift_half_w={lift_half_w:.3f}"
        )
        half_w = lift_half_w

        # mode 由 _select_grasp_pair 直接决定 ("near"=fling, "far"=spread)
        # 远端 spread: 跳过 lift 拉回 (X=0.2), 直接进 spread sweep (3 waypoint), 不扯布料
        if self._last_grasp_mode == "far":
            l_ok, r_ok = self.skills.check_bimanual_grasp(threshold_dist=0.04)
            if not (l_ok and r_ok):
                yield from self._release_to_center(
                    stage_name, "Grasp check failed before spread",
                    l_holding=l_ok, r_holding=r_ok)
                return False
            obs = self.env.get_observation(include_visuals=False)
            self.skills.sync_state(obs)

            yield from self._execute_spread(half_w)
            yield ExecutionStep(plan=self._move_to_init_pose(),
                                settle_time=0.5, description=f"{stage_name}_home")
            self._log_coverage(stage_name)
            return True

        # mode == "near" → fling: 保留 lift step (lift→fling start→pullback→swing→release)
        # 轨迹与 deformable_bench 原版逐字一致 (单段斜拉, 默认 ee_link, vel 0.5).
        # MuJoCo Style3D 下此轨迹能 hold 住; newton 下 pinch 在转运中脱手 —— 是
        # solver 级抓取保持差异, 不是轨迹问题, 修在物理层 (to-optimize §11), 不动此处.
        target_left_lift = [0.2, lift_half_w, self.lift_h]
        target_right_lift = [0.2, -lift_half_w, self.lift_h]
        plan_lift = self.skills.move_dual_arm_to(
            left_target_pose=target_left_lift,
            right_target_pose=target_right_lift,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            velocity_scale=0.5,
        )
        yield ExecutionStep(plan=plan_lift, settle_time=0.5,
                            description=f"{stage_name}_lift")

        l_ok, r_ok = self.skills.check_bimanual_grasp(threshold_dist=0.04)
        if not (l_ok and r_ok):
            yield from self._release_to_center(
                stage_name, "Grasp check failed after lift",
                l_holding=l_ok, r_holding=r_ok)
            return False

        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        start_x = float(self.fling_cfg["release_x"])
        forward_q = np.array([0.707106781, 0.0, 0.707106781, 0.0])
        start_l = np.concatenate([[start_x, half_w, self.lift_h], forward_q])
        start_r = np.concatenate([[start_x, -half_w, self.lift_h], forward_q])

        plan_to_start = self.skills.move_dual_arm_to(
            left_target_pose=start_l, right_target_pose=start_r,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom", velocity_scale=0.5,
        )
        if plan_to_start.success:
            yield ExecutionStep(plan=plan_to_start, settle_time=0.3,
                                description=f"{stage_name}_pre_fling")
        else:
            logger.error(f"[{stage_name}] Failed to move to fling start")
            return False

        l_ok, r_ok = self.skills.check_bimanual_grasp(threshold_dist=0.04)
        if not (l_ok and r_ok):
            yield from self._release_to_center(
                stage_name, "Cloth dropped before fling",
                l_holding=l_ok, r_holding=r_ok)
            return False

        yield from self._execute_fling(start_l, start_r)

        yield ExecutionStep(plan=self._move_to_init_pose(),
                            settle_time=0.5, description=f"{stage_name}_home")
        self._log_coverage(stage_name)
        return True

    # ------------------------------------------------------------------
    # Plan episode
    # ------------------------------------------------------------------
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        logger.info(f"[FlattenSilk] Starting episode {episode_id}")
        self._episode_id_for_dump = episode_id

        self.init_covered_area = get_covered_area_from_particles(
            self._canonical_verts, 0.005)

        cov = self._log_coverage("init")
        if cov >= self.success_coverage:
            logger.error(
                f"[FlattenSilk] init already flat (cov={cov:.4f}); "
                f"random_fold may not have executed. Fail episode."
            )
            raise ValueError(f"init_already_flat(cov={cov:.4f})")

        # 初始化可达性验证
        obs = self.env.get_observation(include_visuals=False)
        try:
            self._verify_init_reachable(obs.pcd_cloth_truth)
        except InitReachableError as e:
            logger.error(f"[FlattenSilk] init reachability failed: {e}")
            raise

        for fling_i in range(1, self.max_flings + 1):
            self._fling_stage = fling_i
            with self.recording(f"fling{fling_i}"):
                success = yield from self._do_fling_stage(f"fling{fling_i}")
                if not success:
                    logger.warning(
                        f"[FlattenSilk] Fling{fling_i} failed, continuing"
                    )
                    continue

            cov = self._log_coverage(f"after_fling{fling_i}")
            if cov >= self.success_coverage:
                logger.success(
                    f"[FlattenSilk] Flat after fling{fling_i} ({cov:.4f})"
                )
                return

        self._log_coverage("final")
