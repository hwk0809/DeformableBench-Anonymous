"""Side-fold T-shirt variant with a right-arm nearest-hem drag grasp.

Only the edge-drag point selection differs from :class:`TshirtFoldV2Task`:
the left arm keeps the legacy selector, while the right arm selects the
currently nearest particle from the garment's low-Y hem band.
"""

from typing import Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from fabricbench.bench.robot.planning.planner_manager import PlannerManager
from fabricbench.bench.task.task_side_fold_tshirt_v2 import TshirtFoldV2Task


class TshirtFoldRightNearestHemTask(TshirtFoldV2Task):
    """Keep the left drag grasp unchanged and move the right grasp to the hem."""

    # For this task, do not test or retry the edge grasp.  Check once after the
    # complete drag, immediately before opening the grippers, and abort the
    # episode if either gripper no longer has cloth nearby.
    DRAG_EDGE_PRE_RELEASE_ABORT = True

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
            planner_manager=planner_manager,
            weld_grasp=weld_grasp,
            collector=collector,
            cfg=cfg,
        )
        task_params = dict(cfg.get("task_params", {}) or {}) if cfg is not None else {}
        self.right_drag_hem_band_frac = float(
            task_params.get("right_drag_hem_band_frac", 0.05)
        )
        if not 0.0 < self.right_drag_hem_band_frac <= 1.0:
            raise ValueError(
                "right_drag_hem_band_frac must be in (0, 1], got "
                f"{self.right_drag_hem_band_frac}"
            )
        logger.info(
            "[TshirtFoldRightNearestHem] right drag selector: "
            f"lowest-Y {100.0 * self.right_drag_hem_band_frac:.1f}% band, "
            "nearest particle to right base; left selector unchanged"
        )

    def _select_drag_edge_indices(
        self,
        cp: np.ndarray,
        lbase_xy: np.ndarray,
        rbase_xy: np.ndarray,
    ) -> Tuple[int, int]:
        # Preserve the original left-arm selector exactly.  The legacy right
        # result is intentionally discarded.
        l_idx, _ = super()._select_drag_edge_indices(cp, lbase_xy, rbase_xy)

        # Match the task's existing forward-fold definition of the hem: the
        # particles in the lowest fraction of the current world-Y extent.  From
        # that dynamic band, use the *nearest* point to the right arm (do not
        # skip the closest tip as the legacy general-edge selector does).
        y_min = float(cp[:, 1].min())
        y_max = float(cp[:, 1].max())
        y_thresh = y_min + self.right_drag_hem_band_frac * max(y_max - y_min, 1e-4)
        hem_idxs = np.where(cp[:, 1] <= y_thresh)[0]
        if len(hem_idxs) == 0:
            # Numerical fallback; the minimum-Y particle always exists for a
            # non-empty cloud, but keep the selector total and explicit.
            hem_idxs = np.array([int(np.argmin(cp[:, 1]))], dtype=np.int64)

        dists = np.linalg.norm(cp[hem_idxs, :2] - rbase_xy, axis=1)
        r_idx = int(hem_idxs[int(np.argmin(dists))])
        logger.info(
            "[TshirtFoldRightNearestHem] right hem selected: "
            f"candidates={len(hem_idxs)}, y_range=[{y_min:.4f}, {y_thresh:.4f}], "
            f"right_idx={r_idx}, point={cp[r_idx].tolist()}, "
            f"base_xy_dist={float(np.min(dists)):.4f}m"
        )
        return l_idx, r_idx
