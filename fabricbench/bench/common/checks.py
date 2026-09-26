"""Reusable success/score check functions — the shared "common" half.

Each task's evaluator (its ``check_success`` / ``get_score`` methods, living in
the task module) composes these. Keep them pure and stateless; per-episode
temporal aggregation is done by the small helpers at the bottom, fed a per-poll
boolean series by the task.

This mirrors how RoboTwin (`_base_task` helpers) / DexGarmentLab (`Utils_Project`
judges) share reusable checks while each task owns its own success logic.
"""
from __future__ import annotations

import numpy as np


# --- instantaneous physical checks -----------------------------------------
def nearest_particle(cloth: np.ndarray, ee_pos: np.ndarray) -> tuple[float, float]:
    """(distance to nearest cloth particle, that particle's height z)."""
    d = np.linalg.norm(cloth - ee_pos[:3], axis=1)
    i = int(d.argmin())
    return float(d[i]), float(cloth[i, 2])


def arm_hold(cloth: np.ndarray, ee_pos: np.ndarray,
             grasp_dist: float = 0.05, lift_height: float = 0.10) -> bool:
    """One gripper is holding the cloth up: nearest cloth particle within
    ``grasp_dist`` of the EE AND above ``lift_height``."""
    d, z = nearest_particle(cloth, ee_pos)
    return d < grasp_dist and z > lift_height


def mean_z_ok(cloth: np.ndarray, threshold: float = 0.10) -> bool:
    """The whole cloth is lifted: mean particle height above ``threshold``."""
    return float(cloth[:, 2].mean()) > threshold


# --- temporal helpers over a per-poll boolean series -----------------------
def ever(series) -> bool:
    """Held at ANY poll (e.g. grabbed even if later slipped)."""
    return any(series)


def end_held(series) -> bool:
    """Held stably at the END: true in the last two polls (~1s)."""
    return len(series) >= 2 and bool(series[-1]) and bool(series[-2])


def sustained(series) -> bool:
    """Held for two consecutive polls (~1s) anywhere — the success-confirm
    criterion, robust to a one-frame flicker."""
    return any(a and b for a, b in zip(series, series[1:]))
