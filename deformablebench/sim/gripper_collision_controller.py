"""Runtime gripper/cloth collision toggle for Newton Style3D scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from loguru import logger

import newton


@dataclass(frozen=True)
class GripperCollisionTrickConfig:
    """State-machine thresholds for the gripper collision release trick."""

    body_names: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "left": ("link7", "link8"),
            "right": ("link7_arm2", "link8_arm2"),
        }
    )
    closed_threshold: float = 0.02
    opening_threshold: float = 0.03
    fully_open_threshold: float = 0.5
    reenable_cooldown_steps: int = 12
    stable_motion_epsilon: float = 1.0e-3
    opening_epsilon: float = 1.0e-4
    closing_epsilon: float = 1.0e-4
    closing_reenable_delta: float = 0.02
    closing_reenable_steps: int = 2
    closed_confirm_steps: int = 1


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1] if label else label


class NewtonGripperCollisionController:
    """Toggle gripper shapes out of cloth contact during release.

    The old MuJoCo/Style3D controller disabled a Style3D rigid body's collision
    by setting ``group=0, mask=0``. Newton exposes ``shape_collision_group`` but
    cloth/rigid soft contacts are gated by ``ShapeFlags.COLLIDE_PARTICLES``, so
    disabling an arm does both:

    - set the gripper shapes' collision group to 0;
    - clear ``COLLIDE_PARTICLES`` for those shapes.

    Restoring an arm writes back the exact original group and flags.
    """

    def __init__(
        self,
        model,
        *,
        config: GripperCollisionTrickConfig | None = None,
        initial_left_gripper: float = 1.0,
        initial_right_gripper: float = 1.0,
    ) -> None:
        self.model = model
        self.config = config or GripperCollisionTrickConfig()

        self.gripper_shape_ids = self._build_gripper_shape_ids()
        self._original_flags = self._snapshot_array("shape_flags")
        self._original_groups = self._snapshot_array("shape_collision_group")

        self._collision_enabled = {"left": True, "right": True}
        self._prev_gripper = {
            "left": float(initial_left_gripper),
            "right": float(initial_right_gripper),
        }
        self._armed = {"left": False, "right": False}
        self._closed_steps = {"left": 0, "right": 0}
        self._disabled_steps = {"left": 0, "right": 0}
        self._pending_open_reenable = {"left": False, "right": False}
        self._disabled_peak_gripper = {"left": 0.0, "right": 0.0}
        self._reverse_close_steps = {"left": 0, "right": 0}

        logger.info("NewtonGripperCollisionController initialized:")
        logger.info(f"  Left gripper shape ids:  {self.gripper_shape_ids.get('left', [])}")
        logger.info(f"  Right gripper shape ids: {self.gripper_shape_ids.get('right', [])}")
        logger.info(
            "  Thresholds: closed={}, opening={}, fully_open={}, cooldown={}".format(
                self.config.closed_threshold,
                self.config.opening_threshold,
                self.config.fully_open_threshold,
                self.config.reenable_cooldown_steps,
            )
        )
        logger.info(
            "  Reverse close restore: delta>={}, steps>={}".format(
                self.config.closing_reenable_delta,
                self.config.closing_reenable_steps,
            )
        )

    def _snapshot_array(self, attr: str) -> np.ndarray | None:
        arr = getattr(self.model, attr, None)
        return arr.numpy().copy() if arr is not None else None

    def _build_gripper_shape_ids(self) -> dict[str, list[int]]:
        result = {"left": [], "right": []}
        if getattr(self.model, "shape_body", None) is None:
            return result

        body_name_to_id = {
            _basename(label): i for i, label in enumerate(getattr(self.model, "body_label", []))
        }
        shape_body = self.model.shape_body.numpy()

        for arm, names in self.config.body_names.items():
            body_ids = {body_name_to_id[name] for name in names if name in body_name_to_id}
            if not body_ids:
                logger.warning(f"[CollisionCtrl][{arm}] no gripper bodies found for {tuple(names)}")
                continue
            result[arm] = [
                shape_id for shape_id, body_id in enumerate(shape_body)
                if int(body_id) in body_ids
            ]
            if not result[arm]:
                logger.warning(f"[CollisionCtrl][{arm}] no shapes found for bodies {tuple(names)}")

        return result

    def _reset_disabled_state(self, arm: str) -> None:
        self._disabled_steps[arm] = 0
        self._pending_open_reenable[arm] = False
        self._disabled_peak_gripper[arm] = self._prev_gripper[arm]
        self._reverse_close_steps[arm] = 0

    def _set_collision_for_arm(self, arm: str, enabled: bool) -> None:
        shape_ids = self.gripper_shape_ids.get(arm, [])
        if not shape_ids:
            self._collision_enabled[arm] = bool(enabled)
            return

        if self._original_flags is not None and getattr(self.model, "shape_flags", None) is not None:
            flags = self.model.shape_flags.numpy()
            particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
            for shape_id in shape_ids:
                if enabled:
                    flags[shape_id] = self._original_flags[shape_id]
                else:
                    flags[shape_id] = int(flags[shape_id]) & ~particle_bit
            self.model.shape_flags.assign(flags)

        if (
            self._original_groups is not None
            and getattr(self.model, "shape_collision_group", None) is not None
        ):
            groups = self.model.shape_collision_group.numpy()
            for shape_id in shape_ids:
                groups[shape_id] = self._original_groups[shape_id] if enabled else 0
            self.model.shape_collision_group.assign(groups)

        self._collision_enabled[arm] = bool(enabled)

    def _is_safe_to_reenable_on_open(self, arm: str, delta: float) -> bool:
        if self._disabled_steps[arm] < int(self.config.reenable_cooldown_steps):
            return False
        return abs(float(delta)) <= float(self.config.stable_motion_epsilon)

    def _is_reverse_close_reenable(self, arm: str, g_val: float, delta: float) -> bool:
        if delta < -float(self.config.closing_epsilon):
            self._reverse_close_steps[arm] += 1
        elif delta > float(self.config.opening_epsilon):
            self._reverse_close_steps[arm] = 0

        self._disabled_peak_gripper[arm] = max(self._disabled_peak_gripper[arm], g_val)
        closing_amount = self._disabled_peak_gripper[arm] - g_val
        return (
            self._reverse_close_steps[arm] >= int(self.config.closing_reenable_steps)
            and closing_amount >= float(self.config.closing_reenable_delta)
        )

    def update(self, left_gripper_val: float, right_gripper_val: float) -> None:
        """Update both arms from normalized gripper values.

        ``g=0`` is closed, ``g=1`` is fully open.
        """

        values = {"left": float(left_gripper_val), "right": float(right_gripper_val)}
        for arm, g_val in values.items():
            prev = self._prev_gripper[arm]
            enabled = self._collision_enabled[arm]
            armed = self._armed[arm]
            delta = g_val - prev

            if g_val <= float(self.config.closed_threshold):
                self._closed_steps[arm] += 1
                if self._closed_steps[arm] >= int(self.config.closed_confirm_steps):
                    self._armed[arm] = True
                if enabled:
                    self._reset_disabled_state(arm)
            else:
                self._closed_steps[arm] = 0

            armed = self._armed[arm]
            if (
                armed
                and enabled
                and g_val > float(self.config.opening_threshold)
                and delta > float(self.config.opening_epsilon)
            ):
                logger.info(
                    f"[CollisionCtrl][{arm}] OPENING g={g_val:.3f} "
                    f"-> DISABLE gripper/cloth collision"
                )
                self._set_collision_for_arm(arm, False)
                self._disabled_steps[arm] = 0
                self._pending_open_reenable[arm] = False
                self._disabled_peak_gripper[arm] = g_val
                self._reverse_close_steps[arm] = 0

            elif not enabled:
                self._disabled_steps[arm] += 1

                if g_val >= float(self.config.fully_open_threshold):
                    self._pending_open_reenable[arm] = True

                if self._is_reverse_close_reenable(arm, g_val, delta):
                    logger.info(
                        f"[CollisionCtrl][{arm}] CLOSING g={g_val:.3f}, "
                        f"from_peak={self._disabled_peak_gripper[arm] - g_val:.3f} "
                        "-> ENABLE gripper/cloth collision"
                    )
                    self._set_collision_for_arm(arm, True)
                    self._armed[arm] = False
                    self._reset_disabled_state(arm)
                elif (
                    self._pending_open_reenable[arm]
                    and self._is_safe_to_reenable_on_open(arm, delta)
                ):
                    logger.info(
                        f"[CollisionCtrl][{arm}] SAFE RE-ENABLE g={g_val:.3f}, "
                        f"disabled_steps={self._disabled_steps[arm]}"
                    )
                    self._set_collision_for_arm(arm, True)
                    self._armed[arm] = False
                    self._reset_disabled_state(arm)

            self._prev_gripper[arm] = g_val

    def set_collision_enabled(self, arm: str, enabled: bool) -> None:
        """Manually toggle one arm, keeping the state machine coherent."""

        if arm not in self._collision_enabled:
            raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")
        self._set_collision_for_arm(arm, bool(enabled))
        if enabled:
            self._armed[arm] = False
            self._reset_disabled_state(arm)
        else:
            self._disabled_peak_gripper[arm] = self._prev_gripper[arm]
            self._reverse_close_steps[arm] = 0
        logger.info(f"[CollisionCtrl][{arm}] Manual set: {'ENABLED' if enabled else 'DISABLED'}")

    def is_collision_enabled(self, arm: str) -> bool:
        return self._collision_enabled.get(arm, True)
