"""Unified success checker — shared by task.check_success() and eval scripts.

Evaluates task success from YAML-defined criteria. Supports atomic checks
(max_z, mean_z, mean_z_below, coverage, flatness, bimanual_hold) and composite checks
(all, any) that can be nested arbitrarily.
"""

from typing import Callable

import numpy as np
from loguru import logger


class SuccessChecker:
    """Evaluate task success from YAML-defined criteria."""

    CLOTH_PARTICLE_RADIUS = 0.005  # same as used across the codebase

    def __init__(
        self,
        check_cfg: dict,
        env,
        custom_check: Callable[[], bool] | None = None,
    ):
        self.cfg = check_cfg
        self.env = env
        self.custom_check = custom_check
        self._init_coverage_area: float | None = None

        # Prefer an environment-provided flat reference area. The open Newton env
        # exposes the pre-disturbance flat particles through initial_cloth_states
        # instead, so derive the same reference from that snapshot when necessary.
        if self._needs_coverage(check_cfg):
            self._init_coverage_area = float(
                getattr(env, "init_covered_area", 0.0) or 0.0)
            if self._init_coverage_area < 1e-9:
                initial_states = getattr(env, "initial_cloth_states", None)
                if initial_states is not None and len(initial_states) > 0:
                    from deformablebench.bench.common.metric_utils import (
                        get_covered_area_from_particles,
                    )
                    self._init_coverage_area = float(
                        get_covered_area_from_particles(
                            np.asarray(initial_states[0]),
                            self.CLOTH_PARTICLE_RADIUS,
                        )
                    )
            logger.info(
                f"[SuccessChecker] Reference coverage area (flat mesh): "
                f"{self._init_coverage_area:.6f}"
            )

    def _needs_coverage(self, cfg: dict) -> bool:
        """Recursively check if any sub-check uses coverage."""
        t = cfg.get("type", "")
        if t == "coverage":
            return True
        if t in ("all", "any"):
            return any(self._needs_coverage(c) for c in cfg.get("checks", []))
        return False

    def check(self) -> bool:
        """Evaluate the success criteria tree."""
        cloth_pos = self.env.get_cloth_position()
        if cloth_pos is None:
            return False
        return self._eval(self.cfg, cloth_pos)

    def _eval(self, cfg: dict, cloth_pos: np.ndarray) -> bool:
        t = cfg["type"]

        if t == "none":
            return False

        if t == "custom":
            if self.custom_check is None:
                raise ValueError(
                    "success_check type 'custom' requires a task check_success callback"
                )
            return bool(self.custom_check())

        if t == "max_z":
            val = float(np.max(cloth_pos[:, 2]))
            return val > cfg["threshold"]

        if t == "mean_z":
            val = float(np.mean(cloth_pos[:, 2]))
            return val > cfg["threshold"]

        if t == "mean_z_below":
            val = float(np.mean(cloth_pos[:, 2]))
            return val < cfg["threshold"]

        if t == "flatness":
            val = float(np.std(cloth_pos[:, 2]))
            return val < cfg["threshold"]

        if t == "coverage":
            if self._init_coverage_area is None or self._init_coverage_area < 1e-9:
                return False
            from deformablebench.bench.common.metric_utils import get_covered_area_from_particles
            cur_area = get_covered_area_from_particles(
                cloth_pos, self.CLOTH_PARTICLE_RADIUS
            )
            ratio = cur_area / self._init_coverage_area
            return ratio > cfg["threshold"]

        if t == "bimanual_hold":
            return self._check_bimanual_hold(cfg, cloth_pos)

        if t == "all":
            return all(self._eval(c, cloth_pos) for c in cfg["checks"])

        if t == "any":
            return any(self._eval(c, cloth_pos) for c in cfg["checks"])

        raise ValueError(f"Unknown success check type: {t}")

    @staticmethod
    def _arm_grasp_ok(cloth_pos: np.ndarray, ee_pos: np.ndarray,
                      max_dist: float, min_height: float):
        """Single-arm grasp+lift test: NEAREST cloth point is close to the EE
        AND lifted above the table. Mirrors ClosedLoopSkills.check_single_arm_grasp
        so eval and data-collection use the same criterion.

        Returns (ok, min_dist, nearest_z).
        """
        dists = np.linalg.norm(cloth_pos - ee_pos, axis=1)
        i = int(np.argmin(dists))
        min_dist = float(dists[i])
        nearest_z = float(cloth_pos[i, 2])
        return (min_dist < max_dist and nearest_z > min_height), min_dist, nearest_z

    def _check_bimanual_hold(self, cfg: dict, cloth_pos: np.ndarray) -> bool:
        """Both arms actually grasped AND lifted the cloth.

        For each arm the NEAREST cloth point must be within ``grasp_dist`` of the
        EE (grasped) AND above ``lift_height`` (held off the table). Both arms
        must pass. This mirrors ClosedLoopSkills.check_bimanual_grasp (the
        collection-time verifier) — one shared criterion for eval + collection.

        Replaces the old count-within-radius proximity test, which falsely
        passed whenever an arm merely hovered over the (dense) flat cloth:
        a 5 cm sphere around an EE catches hundreds of vertices even with the
        gripper open, so ``min_particles=3`` was trivially satisfied.

        Back-compat: legacy ``ee_radius`` is accepted as the distance threshold.
        """
        max_dist = float(cfg.get("grasp_dist", cfg.get("ee_radius", 0.04)))
        min_height = float(cfg.get("lift_height", 0.05))

        left_ee = self.env.left_eef_pose[:3]
        right_ee = self.env.right_eef_pose[:3]
        l_ok, l_d, l_z = self._arm_grasp_ok(cloth_pos, left_ee, max_dist, min_height)
        r_ok, r_d, r_z = self._arm_grasp_ok(cloth_pos, right_ee, max_dist, min_height)

        logger.debug(
            f"[bimanual_hold] L(dist={l_d:.3f},z={l_z:.3f})->{l_ok} "
            f"R(dist={r_d:.3f},z={r_z:.3f})->{r_ok} "
            f"(need dist<{max_dist}, z>{min_height})"
        )
        return l_ok and r_ok

    def describe(self) -> str:
        """Human-readable description of the check tree."""
        return self._describe(self.cfg)

    def _describe(self, cfg: dict) -> str:
        t = cfg["type"]
        if t in ("max_z", "mean_z"):
            return f"{t} > {cfg['threshold']}"
        if t == "mean_z_below":
            return f"mean_z < {cfg['threshold']}"
        if t == "flatness":
            return f"std(z) < {cfg['threshold']}"
        if t == "coverage":
            return f"coverage > {cfg['threshold']}"
        if t == "bimanual_hold":
            return (
                f"bimanual_grasp(dist<{cfg.get('grasp_dist', cfg.get('ee_radius', 0.04))}, "
                f"z>{cfg.get('lift_height', 0.05)})"
            )
        if t == "none":
            return "disabled (no early stop)"
        if t == "custom":
            return "custom(task.check_success)"
        if t in ("all", "any"):
            sep = " AND " if t == "all" else " OR "
            parts = [self._describe(c) for c in cfg["checks"]]
            return f"({sep.join(parts)})"
        return f"unknown({t})"
