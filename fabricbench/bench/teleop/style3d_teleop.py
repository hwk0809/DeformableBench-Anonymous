"""Keyboard and joystick teleoperation for ``DeformableEnv`` benchmark scenes.

This is intentionally solver-agnostic at the control layer: it talks only to
``DeformableEnv`` (joint state, gripper state, and ``step(q_l, q_r, g_l, g_r)``).
Today that means the open Newton ``SolverStyle3D`` backend; if another backend
is enabled behind ``make_cloth_solver()``, this controller should not need to
change.

Example:
    uv run python -m fabricbench.bench.teleop.style3d_teleop \
        --task grasp_silk_bimanual --viewer gl --input keyboard

Newton IK backend:
    uv run python -m fabricbench.bench.teleop.style3d_teleop \
        --task grasp_silk_bimanual --viewer gl --ik-backend newton
"""

from __future__ import annotations

from fabricbench._env import apply_contact_backend_env, ensure_clean_ld_path

ensure_clean_ld_path()  # before importing pinocchio
apply_contact_backend_env()  # contact backend wp.constants must be set before importing Newton

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

try:
    import pygame  # noqa: E402
except ImportError:  # pragma: no cover - keyboard mode deliberately needs no pygame
    pygame = None

from fabricbench.bench.data.collect import state_14d  # noqa: E402
from fabricbench.bench.data.schema import MOTOR_NAMES  # noqa: E402
from fabricbench.bench.robot.kinematics.base_kinematics import BaseKinematics  # noqa: E402
from fabricbench.bench.task.randomization import Randomizer  # noqa: E402
from fabricbench.bench.task.spec import TaskSpec  # noqa: E402
from fabricbench.sim import DeformableEnv  # noqa: E402


GROUND_CHECK_FRAMES = (
    "tcp_link",
    "tcp_link_bottom",
    "tcp_link_top",
    "tcp_link_center",
    "tcp_link_bottom_left",
    "tcp_link_bottom_right",
    "tcp_link_top_left",
    "tcp_link_top_right",
    "link7",
    "link8",
)

# ViewerGL reserves WASD/QE for camera motion, Space/Period for pause and
# single-step, and H/F/Escape for UI/camera actions.  Keep every robot binding
# outside that set so one key press can never move both the robot and camera.
NEWTON_VIEWER_RESERVED_KEYS = frozenset(
    {
        "w",
        "a",
        "s",
        "d",
        "q",
        "e",
        "space",
        "period",
        "h",
        "f",
        "escape",
        "up",
        "down",
        "left",
        "right",
    }
)
KEYBOARD_AXIS_PAIRS = (
    ("i", "k"),  # joint 1 or X
    ("j", "l"),  # joint 2 or Y
    ("u", "o"),  # joint 3 or Z
    ("t", "g"),  # joint 4 or roll
    ("y", "n"),  # joint 5 or pitch
    ("r", "v"),  # joint 6 or yaw
)
KEYBOARD_EVENT_MAP = {
    "c": "back",   # switch active arm
    "m": "start",  # joint/pose mode
    "1": "lb",     # slower
    "2": "rb",     # faster
    "3": "x",      # collision toggle
    "b": "y",      # home
}
KEYBOARD_GRIPPER_KEYS = {"z", "x"}  # close, open
KEYBOARD_CONTROL_KEYS = (
    frozenset(key for pair in KEYBOARD_AXIS_PAIRS for key in pair)
    | frozenset(KEYBOARD_EVENT_MAP)
    | frozenset(KEYBOARD_GRIPPER_KEYS)
)


@dataclass
class Button:
    last_state: bool = False

    def update(self, current_state: bool) -> bool:
        triggered = bool(current_state) and not self.last_state
        self.last_state = bool(current_state)
        return triggered


class TeleopTrajectoryRecorder:
    """Collect commanded and realized 14-D robot trajectories in memory."""

    FORMAT_VERSION = 1

    def __init__(self, path: str | Path, metadata: dict, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".npz":
            self.path = self.path.with_suffix(".npz")
        self.overwrite = bool(overwrite)
        self.metadata = dict(metadata)
        self.metadata["format_version"] = self.FORMAT_VERSION
        self.metadata["created_at"] = datetime.now().astimezone().isoformat()
        self._started_at: float | None = None
        self._wall_times: list[float] = []
        self._sim_times: list[float] = []
        self._actions: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._active_arms: list[str] = []
        self._control_modes: list[str] = []

    @property
    def num_samples(self) -> int:
        return len(self._actions)

    def record(self, controller: "Style3DTeleopController") -> None:
        env = controller.env
        now = time.perf_counter()
        if self._started_at is None:
            self._started_at = now
        action = np.array(
            [
                *np.asarray(controller.left_joints, dtype=float)[:6],
                float(controller.left_gripper),
                *np.asarray(controller.right_joints, dtype=float)[:6],
                float(controller.right_gripper),
            ],
            dtype=np.float32,
        )
        get_sim_time = getattr(env, "get_current_sim_time", None)
        sim_time = float(get_sim_time()) if get_sim_time is not None else float(env.sim_time)
        self._wall_times.append(now - self._started_at)
        self._sim_times.append(sim_time)
        self._actions.append(action)
        self._states.append(state_14d(env))
        self._active_arms.append(controller.active_arm)
        self._control_modes.append(controller.control_mode)

    def save(self) -> Path:
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"trajectory output already exists: {self.path} (pass --record-overwrite)")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        actions = (
            np.stack(self._actions).astype(np.float32, copy=False)
            if self._actions
            else np.empty((0, len(MOTOR_NAMES)), dtype=np.float32)
        )
        states = (
            np.stack(self._states).astype(np.float32, copy=False)
            if self._states
            else np.empty((0, len(MOTOR_NAMES)), dtype=np.float32)
        )
        payload = {
            "time_s": np.asarray(self._wall_times, dtype=np.float64),
            "sim_time_s": np.asarray(self._sim_times, dtype=np.float64),
            "action": actions,
            "observation_state": states,
            "active_arm": np.asarray(self._active_arms),
            "teleop_mode": np.asarray(self._control_modes),
            "motor_names": np.asarray(MOTOR_NAMES),
            "metadata_json": np.asarray(
                json.dumps({**self.metadata, "num_samples": self.num_samples}, sort_keys=True)
            ),
        }

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self.path.stem}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                np.savez_compressed(temp_file, **payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        logger.success(f"Saved {self.num_samples} teleop trajectory samples to {self.path}")
        return self.path


class NullGripperCollisionController:
    """Compatibility shim for the old MuJoCo teleop hotkeys.

    Current Newton/Style3D scenes use material/contact tuning rather than the old
    runtime gripper collision toggle. Keeping this no-op object lets the UI and
    status line stay useful without adding a second control path.
    """

    def __init__(self) -> None:
        self._enabled = {"left": True, "right": True}

    def update(self, _left_gripper: float, _right_gripper: float) -> None:
        return

    def is_collision_enabled(self, arm: str) -> bool:
        return self._enabled.get(arm, True)

    def set_collision_enabled(self, arm: str, enabled: bool) -> None:
        self._enabled[arm] = bool(enabled)
        logger.info(f"{arm} gripper collision toggle is a no-op for DeformableEnv")


def _existing_frames(kin: BaseKinematics) -> tuple[str, ...]:
    model = getattr(kin, "model", None)
    exist_frame = getattr(model, "existFrame", None)
    if exist_frame is None:
        return ()
    return tuple(name for name in GROUND_CHECK_FRAMES if exist_frame(name))


def _check_ground_penetration(
    kin: BaseKinematics,
    q_arm: np.ndarray,
    gripper_open: float,
    grip_limit: float,
    frame_names: tuple[str, ...],
    base_z: float = 0.0,
) -> bool:
    g_phys = float(gripper_open) * float(grip_limit)
    q_full = np.concatenate([np.asarray(q_arm, dtype=float)[:6], [g_phys, -g_phys]])
    for frame in frame_names:
        pose = kin.solve_fk(q_full, ee_link=frame)
        if pose[2] + base_z < 0.0:
            return True
    return False


class Style3DTeleopController:
    HOME_PRESETS = {
        "zero": None,
        "lift_silk": {
            "home_left": np.array([-0.1240, 0.7980, -1.1250, -0.2253, 0.9856, 0.0829]),
            "home_right": np.array([0.1240, 0.7980, -1.1250, 0.2253, 0.9856, -0.0829]),
            "gripper": 1.0,
        },
    }

    def __init__(
        self,
        env: DeformableEnv,
        left_ik: BaseKinematics,
        right_ik: BaseKinematics,
        *,
        control_freq: float,
        gripper_close_min_value: float,
        home_preset: str,
        ground_check: bool,
        input_device: str,
        keyboard_available: bool,
        trajectory_recorder: TeleopTrajectoryRecorder | None = None,
    ) -> None:
        if not env.has_robot:
            raise ValueError("Teleop requires a robot scene")

        self.env = env
        self.left_ik = left_ik
        self.right_ik = right_ik
        self.gripper_close_min_value = float(gripper_close_min_value)
        self.ground_check = bool(ground_check)
        self.left_ground_frames = _existing_frames(left_ik)
        self.right_ground_frames = _existing_frames(right_ik)
        if self.ground_check and not (self.left_ground_frames or self.right_ground_frames):
            logger.warning("Ground check is unavailable for this IK backend; continuing without frame ground checks.")

        self.joystick = None
        self._pygame_initialized = False
        self.input_device = self._setup_input_device(input_device, keyboard_available)
        self.trajectory_recorder = trajectory_recorder

        self._sync_state_from_env()
        self._apply_home_preset(home_preset)

        self.active_arm = "left"
        self.control_mode = "joint"
        self.deadzone = 0.1
        self.speed_factors = [0.25, 0.5, 1.0, 2.0, 3.0]
        self.speed_idx = 2
        self.step_joint = 0.04
        self.step_pos = 0.004
        self.step_rot = 2.0
        self.step_gripper = 0.1

        self.control_freq = float(control_freq)
        self.sim_dt = self.env.frame_dt
        self.substeps = max(1, int(round((1.0 / self.control_freq) / self.sim_dt)))

        self.buttons = {k: Button() for k in ("a", "b", "x", "y", "lb", "rb", "back", "start", "home")}
        self.keyboard_buttons = {key: Button() for key in KEYBOARD_EVENT_MAP}
        self.axis_map = {"lx": 0, "ly": 1, "rx": 3, "ry": 4, "lt": 2, "rt": 5}
        self.btn_map = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5, "back": 6, "start": 7, "home": 8}
        self.collision_ctrl = (
            env.get_gripper_collision_controller()
            if hasattr(env, "get_gripper_collision_controller")
            else NullGripperCollisionController()
        )
        self.print_instructions()

    def _setup_input_device(self, requested: str, keyboard_available: bool) -> str:
        if requested in ("auto", "joystick"):
            if pygame is None:
                if requested == "joystick":
                    raise SystemExit(
                        "pygame is required for joystick teleop. Install it with `uv sync --extra teleop`."
                    )
            else:
                pygame.init()
                pygame.joystick.init()
                self._pygame_initialized = True
                if pygame.joystick.get_count() > 0:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    logger.success(f"Joystick: {self.joystick.get_name()}")
                    return "joystick"
                if requested == "joystick":
                    raise SystemExit("No joystick detected. Connect one and restart teleop.")

        if keyboard_available:
            if self._pygame_initialized:
                pygame.quit()
                self._pygame_initialized = False
            logger.success("Keyboard input: focus the Newton ViewerGL window")
            return "keyboard"
        raise SystemExit("Keyboard teleop requires `--viewer gl`; no usable joystick was detected.")

    @property
    def current_joints(self) -> np.ndarray:
        return self.left_joints if self.active_arm == "left" else self.right_joints

    @current_joints.setter
    def current_joints(self, value: np.ndarray) -> None:
        if self.active_arm == "left":
            self.left_joints = value
        else:
            self.right_joints = value

    @property
    def current_ik(self) -> BaseKinematics:
        return self.left_ik if self.active_arm == "left" else self.right_ik

    @property
    def current_ground_frames(self) -> tuple[str, ...]:
        return self.left_ground_frames if self.active_arm == "left" else self.right_ground_frames

    @property
    def current_limits(self) -> np.ndarray:
        indices = self.env.idx_l_arm if self.active_arm == "left" else self.env.idx_r_arm
        lower = self.env.model.joint_limit_lower.numpy()[indices]
        upper = self.env.model.joint_limit_upper.numpy()[indices]
        return np.stack([lower, upper], axis=1)

    def _sync_state_from_env(self) -> None:
        self.left_joints = self.env.left_qpos[:6].copy()
        self.right_joints = self.env.right_qpos[:6].copy()
        self.left_gripper = float(self.env.left_gripper_val)
        self.right_gripper = float(self.env.right_gripper_val)
        self.last_left_joints = self.left_joints.copy()
        self.last_right_joints = self.right_joints.copy()
        self.last_left_gripper = self.left_gripper
        self.last_right_gripper = self.right_gripper

    def _save_home(self) -> None:
        self.home_left_joints = self.left_joints.copy()
        self.home_right_joints = self.right_joints.copy()
        self.home_left_gripper = self.left_gripper
        self.home_right_gripper = self.right_gripper

    def _apply_home_preset(self, preset_name: str) -> None:
        preset = self.HOME_PRESETS.get(preset_name)
        if preset is None:
            self._save_home()
            logger.info("Home preset: zero/current env state")
            return

        self.home_left_joints = preset["home_left"].copy()
        self.home_right_joints = preset["home_right"].copy()
        self.home_left_gripper = float(preset["gripper"])
        self.home_right_gripper = float(preset["gripper"])
        self.left_joints = self.home_left_joints.copy()
        self.right_joints = self.home_right_joints.copy()
        self.left_gripper = self.home_left_gripper
        self.right_gripper = self.home_right_gripper
        self.last_left_joints = self.left_joints.copy()
        self.last_right_joints = self.right_joints.copy()
        self.last_left_gripper = self.left_gripper
        self.last_right_gripper = self.right_gripper
        logger.info(f"Home preset: {preset_name}")

    def _get_axis(self, name: str) -> float:
        if self.joystick is None:
            return 0.0
        value = float(self.joystick.get_axis(self.axis_map[name]))
        if abs(value) < self.deadzone:
            return 0.0
        if name in ("lt", "rt"):
            return (value + 1.0) / 2.0
        return value

    def _keyboard_is_captured(self) -> bool:
        gui = getattr(self.env.viewer, "gui", None)
        should_ignore = getattr(gui, "should_ignore_keyboard_input", None)
        return bool(should_ignore()) if should_ignore is not None else False

    def _key_down(self, key: str) -> bool:
        if self.input_device != "keyboard" or self._keyboard_is_captured():
            return False
        return bool(self.env.viewer.is_key_down(key))

    def _keyboard_axes(self) -> np.ndarray:
        """Return six signed axes; each pair is documented as negative/positive."""
        return np.asarray(
            [float(self._key_down(pos)) - float(self._key_down(neg)) for neg, pos in KEYBOARD_AXIS_PAIRS],
            dtype=float,
        )

    def _input_events(self) -> dict[str, bool]:
        if self.input_device == "joystick":
            pygame.event.pump()
            return {
                name: True
                for name, btn in self.buttons.items()
                if btn.update(bool(self.joystick.get_button(self.btn_map[name])))
            }
        return {
            action: True
            for key, action in KEYBOARD_EVENT_MAP.items()
            if self.keyboard_buttons[key].update(self._key_down(key))
        }

    def update(self) -> bool:
        self._handle_buttons(self._input_events())

        speed = self.speed_factors[self.speed_idx]
        if self.control_mode == "joint":
            self._update_joint_mode(speed)
        else:
            self._update_pose_mode(speed)
        self._update_gripper(speed)
        self._step_physics_substeps()
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record(self)
        return True

    def _step_physics_substeps(self) -> None:
        start_left = self.last_left_joints
        start_right = self.last_right_joints
        start_g_left = self.last_left_gripper
        start_g_right = self.last_right_gripper
        target_left = self.left_joints
        target_right = self.right_joints
        target_g_left = self.left_gripper
        target_g_right = self.right_gripper

        for i in range(1, self.substeps + 1):
            alpha = i / self.substeps
            self.env.step(
                start_left * (1.0 - alpha) + target_left * alpha,
                start_right * (1.0 - alpha) + target_right * alpha,
                start_g_left * (1.0 - alpha) + target_g_left * alpha,
                start_g_right * (1.0 - alpha) + target_g_right * alpha,
            )

        self.collision_ctrl.update(float(self.env.left_gripper_val), float(self.env.right_gripper_val))
        self.last_left_joints = target_left.copy()
        self.last_right_joints = target_right.copy()
        self.last_left_gripper = target_g_left
        self.last_right_gripper = target_g_right

    def _handle_buttons(self, events: dict[str, bool]) -> None:
        if "back" in events:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            logger.info(f"Switched to {self.active_arm.upper()} arm")
        if "start" in events:
            self.control_mode = "pose" if self.control_mode == "joint" else "joint"
            logger.info(f"Switched to {self.control_mode.upper()} mode")
        if "lb" in events:
            self.speed_idx = max(0, self.speed_idx - 1)
            logger.info(f"Speed: {self.speed_factors[self.speed_idx]}x")
        if "rb" in events:
            self.speed_idx = min(len(self.speed_factors) - 1, self.speed_idx + 1)
            logger.info(f"Speed: {self.speed_factors[self.speed_idx]}x")
        if "x" in events:
            enabled = self.collision_ctrl.is_collision_enabled(self.active_arm)
            self.collision_ctrl.set_collision_enabled(self.active_arm, not enabled)
        if "y" in events:
            logger.info("Returning arms to saved home pose")
            self.left_joints = self.home_left_joints.copy()
            self.right_joints = self.home_right_joints.copy()
            self.left_gripper = self.home_left_gripper
            self.right_gripper = self.home_right_gripper

    def _update_joint_mode(self, speed: float) -> None:
        if self.input_device == "keyboard":
            delta = self._keyboard_axes() * self.step_joint * speed
            new_joints = self.current_joints + delta
            limits = self.current_limits
            new_joints = np.clip(new_joints, limits[:, 0], limits[:, 1])
            if self._is_safe(new_joints):
                self.current_joints = new_joints
            return

        lx = self._get_axis("lx")
        ly = self._get_axis("ly")
        rx = self._get_axis("rx")
        ry = self._get_axis("ry")
        hat = self.joystick.get_hat(0)

        delta = np.zeros(6)
        delta[0] = -lx * self.step_joint * speed
        delta[1] = -ly * self.step_joint * speed
        delta[2] = ry * self.step_joint * speed
        delta[5] = rx * self.step_joint * speed
        delta[3] = hat[0] * self.step_joint * speed
        delta[4] = -hat[1] * self.step_joint * speed

        new_joints = self.current_joints + delta
        limits = self.current_limits
        new_joints = np.clip(new_joints, limits[:, 0], limits[:, 1])
        if self._is_safe(new_joints):
            self.current_joints = new_joints

    def _update_pose_mode(self, speed: float) -> None:
        if self.input_device == "keyboard":
            axes = self._keyboard_axes()
            if not np.any(axes):
                return
            d_world = axes[:3] * self.step_pos * speed
            r_local_euler = axes[3:] * self.step_rot * speed
        else:
            lx = self._get_axis("lx")
            ly = self._get_axis("ly")
            rx = self._get_axis("rx")
            ry = self._get_axis("ry")
            hat = self.joystick.get_hat(0)
            if abs(lx) + abs(ly) + abs(rx) + abs(ry) + abs(hat[0]) + abs(hat[1]) < 1.0e-3:
                return
            d_world = np.array([-ly, -lx, -ry]) * self.step_pos * speed
            r_local_euler = np.array([hat[0], -hat[1], -rx]) * self.step_rot * speed

        current_pose = self.current_ik.solve_fk(self.current_joints)
        curr_pos = current_pose[:3]
        curr_quat_wxyz = current_pose[3:]
        r_curr = R.from_quat([curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]])

        r_new = r_curr * R.from_euler("xyz", r_local_euler, degrees=True)
        q_xyzw = r_new.as_quat()
        target_pose = np.concatenate([curr_pos + d_world, [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]])

        ik_result = self.current_ik.solve_ik(
            target_pose=target_pose,
            seed_joints=self.current_joints,
            threshold=1.0e-3,
            strict_seed=True,
        )
        if ik_result is not None:
            candidate = np.asarray(ik_result, dtype=float)[:6]
            limits = self.current_limits
            candidate = np.clip(candidate, limits[:, 0], limits[:, 1])
            if self._is_safe(candidate):
                self.current_joints = candidate

    def _is_safe(self, joints: np.ndarray) -> bool:
        if not self.ground_check:
            return True
        if not self.current_ground_frames:
            return True
        gripper = self.left_gripper if self.active_arm == "left" else self.right_gripper
        return not _check_ground_penetration(
            self.current_ik,
            joints,
            gripper,
            self.env.robot.gripper_limit,
            self.current_ground_frames,
        )

    def _update_gripper(self, speed: float) -> None:
        if self.input_device == "keyboard":
            delta = (float(self._key_down("x")) - float(self._key_down("z"))) * self.step_gripper * speed
        else:
            lt = self._get_axis("lt")
            rt = self._get_axis("rt")
            delta = 0.0
            if lt > 0.0:
                delta = -lt * self.step_gripper * speed
            if rt > 0.0:
                delta = rt * self.step_gripper * speed

        if self.active_arm == "left":
            self.left_gripper = float(np.clip(self.left_gripper + delta, self.gripper_close_min_value, 1.0))
        else:
            self.right_gripper = float(np.clip(self.right_gripper + delta, self.gripper_close_min_value, 1.0))

    def print_instructions(self) -> None:
        print("\n" + "=" * 50)
        print("STYLE3D TELEOP CONTROL GUIDE")
        print("=" * 50)
        print(
            f"Input: {self.input_device.upper()} | control: {self.active_arm.upper()} arm "
            f"| mode: {self.control_mode.upper()}"
        )
        if self.input_device == "keyboard":
            print("Focus ViewerGL. Robot keys intentionally avoid Newton's WASD/QE camera bindings.")
            print("[C] switch arm   [M] joint/pose mode   [B] home")
            print("[1/2] slower/faster   [3] collision toggle placeholder")
            print("Joint: I/K J1-/+, J/L J2-/+, U/O J3-/+, T/G J4-/+, Y/N J5-/+, R/V J6-/+")
            print("Pose:  I/K X-/+,  J/L Y-/+,  U/O Z-/+,  T/G roll-/+, Y/N pitch-/+, R/V yaw-/+")
            print("Gripper: Z close, X open")
        else:
            print("[BACK] switch arm   [START] joint/pose mode   [Y] home")
            print("[LB/RB] speed       [X] collision toggle placeholder")
            print("Joint: left stick J1/J2, right stick J3/J6, d-pad J4/J5")
            print("Pose: left stick XY, right stick Z/yaw, d-pad roll/pitch")
            print("Gripper: LT close, RT open")
        print(f"frame dt: {self.sim_dt * 1000.0:.2f} ms | control: {self.control_freq:.1f} Hz")
        print(f"env steps per control tick: {self.substeps}")
        print("=" * 50 + "\n")


def _make_viewer(kind: str):
    import newton.viewer as V

    if kind == "gl":
        return V.ViewerGL()
    return V.ViewerNull()


def _build_env(args: argparse.Namespace) -> tuple[TaskSpec, DeformableEnv]:
    spec = TaskSpec.from_name(args.task)
    spec.control_mode = args.control
    randomizer = None if args.no_rand else Randomizer(spec.randomization, episode_id=args.episode)
    viewer = _make_viewer(args.viewer)
    env = DeformableEnv(viewer=viewer, cfg=spec.scene_config(args.device, randomizer))

    if randomizer is not None:
        info = randomizer.apply_robot_init(env)
        if info:
            logger.info(f"robot_init: {info}")
    for _ in range(args.settle_frames if args.settle_frames is not None else spec.settle_frames):
        env.step()
        env.render()
    if randomizer is not None and args.apply_random_fold:
        info = randomizer.apply_cloth_fold(env)
        if info:
            logger.info(f"random_fold: {info}")
    return spec, env


def _build_kinematics(
    args: argparse.Namespace,
    spec: TaskSpec,
    env: DeformableEnv,
) -> tuple[BaseKinematics, BaseKinematics]:
    if args.ik_backend == "newton":
        from fabricbench.bench.robot.kinematics.newton_kinematics import NewtonKinematics

        return (
            NewtonKinematics.from_deformable_env(env, "left", iterations=args.ik_iters),
            NewtonKinematics.from_deformable_env(env, "right", iterations=args.ik_iters),
        )

    try:
        from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
    except ImportError as exc:  # pragma: no cover - exercised only in lean installs
        raise SystemExit(
            "Pinocchio/planning dependencies are required for the default teleop IK backend. "
            "Install them with `uv sync --extra planning --extra teleop`, or use `--ik-backend newton`."
        ) from exc

    robot = spec.robot_config()
    return (
        PinocchioKinematics(robot.urdf, base_link=robot.base_link, ee_link=robot.ee_link),
        PinocchioKinematics(robot.urdf, base_link=robot.base_link, ee_link=robot.ee_link),
    )


def _resolve_record_path(raw_path: str | None, task: str) -> Path | None:
    if raw_path is None:
        return None
    if raw_path == "__auto__":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Path("data/teleop") / f"{task}_{stamp}.npz"
    path = Path(raw_path).expanduser()
    return path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="grasp_silk_bimanual")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument(
        "--input",
        default="auto",
        choices=["auto", "keyboard", "joystick"],
        help="auto uses the first joystick when present, otherwise ViewerGL keyboard input",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--control", default="kinematic", choices=["kinematic", "pd"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--no-rand", action="store_true")
    ap.add_argument("--settle-frames", type=int, default=None)
    ap.add_argument("--apply-random-fold", action="store_true")
    ap.add_argument("--control-freq", type=float, default=50.0)
    ap.add_argument("--home-preset", default="zero", choices=sorted(Style3DTeleopController.HOME_PRESETS))
    ap.add_argument("--gripper-close-min", type=float, default=0.0)
    ap.add_argument("--ground-check", action="store_true")
    ap.add_argument("--ik-backend", default="pinocchio", choices=["pinocchio", "newton"])
    ap.add_argument("--ik-iters", type=int, default=24, help="Newton IK iterations when --ik-backend=newton")
    ap.add_argument(
        "--record",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help="record target and actual 14-D trajectories; default path: data/teleop/<task>_<timestamp>.npz",
    )
    ap.add_argument("--record-overwrite", action="store_true", help="allow replacing an explicit record path")
    args = ap.parse_args()

    record_path = _resolve_record_path(args.record, args.task)
    if record_path is not None and record_path.exists() and not args.record_overwrite:
        ap.error(f"trajectory output already exists: {record_path} (pass --record-overwrite)")

    spec, env = _build_env(args)
    left_ik, right_ik = _build_kinematics(args, spec, env)
    recorder = None
    if record_path is not None:
        recorder = TeleopTrajectoryRecorder(
            record_path,
            {
                "task": spec.name,
                "actuation_control": spec.control_mode,
                "requested_input": args.input,
                "episode": args.episode,
                "randomization_enabled": not args.no_rand,
                "random_fold_applied": args.apply_random_fold,
                "control_freq": args.control_freq,
                "home_preset": args.home_preset,
                "ik_backend": args.ik_backend,
                "sample_semantics": "action[i] is the target applied before observation_state[i] is sampled",
            },
            overwrite=args.record_overwrite,
        )
    controller = Style3DTeleopController(
        env,
        left_ik,
        right_ik,
        control_freq=args.control_freq,
        gripper_close_min_value=args.gripper_close_min,
        home_preset=args.home_preset,
        ground_check=args.ground_check,
        input_device=args.input,
        keyboard_available=args.viewer == "gl",
        trajectory_recorder=recorder,
    )
    if recorder is not None:
        recorder.metadata["input"] = controller.input_device
        logger.info(f"Recording teleop trajectory to {recorder.path}")

    logger.success(
        f"Teleop ready: task={spec.name} solver=style3d control={spec.control_mode} "
        f"ik={args.ik_backend} input={controller.input_device}"
    )
    try:
        while getattr(env.viewer, "is_running", lambda: True)():
            tick_start = time.perf_counter()
            controller.update()
            env.render()
            col_l = "ON" if controller.collision_ctrl.is_collision_enabled("left") else "OFF"
            col_r = "ON" if controller.collision_ctrl.is_collision_enabled("right") else "OFF"
            record_status = f" | REC:{recorder.num_samples}" if recorder is not None else ""
            sys.stdout.write(
                f"\r[{controller.active_arm.upper()} {controller.control_mode.upper()}] "
                f"Grip L:{controller.left_gripper:.2f} R:{controller.right_gripper:.2f} | "
                f"Col L:{col_l} R:{col_r}{record_status}   "
            )
            sys.stdout.flush()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - tick_start)))
    except KeyboardInterrupt:
        print()
    finally:
        try:
            if recorder is not None:
                recorder.save()
        finally:
            if controller._pygame_initialized:
                pygame.quit()
            if env.viewer is not None and hasattr(env.viewer, "close"):
                env.viewer.close()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
