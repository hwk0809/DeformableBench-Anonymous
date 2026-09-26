"""
Real Robot Policy Deployment (Multi-Process)

Ported from ``deformable_bench/scripts/eval_lerobot_real.py``. That script is
field-validated on the dual-Piper rig, so everything here — observation build,
unit conversions, joint limits, gripper clipping, ground-penetration FK+IK
correction, reset interpolation, the 200Hz SDK sender thread — is kept
byte-identical to the original. The ONLY thing that differs is the policy seam
in the Inference process, so that all six benchmark policies can be deployed
from one entry point instead of one copy-pasted script per policy.

Architecture:
    Perception Process  →  Inference Process  →  Execution Process
    (lerobot Robot+Cam)    (see --policy_type)    (piper_sdk direct)
                    ↑                ↑                    ↑
                    └────── Keyboard Process ──────────────┘

Policy backends (--policy_type):
  - ``lerobot`` (default): in-process LeRobot checkpoint — ACT, Diffusion
    Policy, Pi0, Pi0.5/pi05_evo, SmolVLA. Algorithm is auto-detected from the
    checkpoint. This is the original, verified code path, unchanged, and is the
    ONLY path supporting --chunked_inference / --rtc_enabled.
  - ``gr00t`` / ``lingbot`` / ``fastwam``: out-of-process model servers reached
    through the shared ``bench/policy/`` adapters (the same ones sim eval uses).
    They own their action chunking internally, so the chunked/RTC flags do not
    apply and are rejected rather than silently ignored.

Inference modes (``lerobot`` backend only):
  - Default (per-step): each 30Hz obs triggers one predict_action call.
  - --chunked_inference: predict_action_chunk once per chunk, subsequent obs
    pop pre-computed actions from a ChunkedActionBuffer. Async gaps during
    the ~200ms chunk inference are bridged by the 200Hz execution loop
    re-using the last queued action — no stall.
    Optional --rtc_enabled adds Real-Time Chunking guidance to align the
    new chunk's prefix with the old chunk's leftover.

Usage:
    conda run -n lerobot_v042 python -m fabricbench.bench.real_eval \
        --robot.type=dual_piper \
        --robot.cameras="{ \
            static_cam:     {type: intelrealsense, serial_number_or_name: '<SN1>', fps: 30, width: 1280, height: 720}, \
            left_hand_cam:  {type: intelrealsense, serial_number_or_name: '<SN2>', fps: 30, width: 1280, height: 720}, \
            right_hand_cam: {type: intelrealsense, serial_number_or_name: '<SN3>', fps: 30, width: 1280, height: 720}}" \
        --policy.path=path/to/checkpoint \
        --task="fold the cloth" \
        --manual_exposure=26 \
        --manual_gain=64 \
        --chunked_inference --aggregate_fn weighted_average --rtc_enabled

See docs/real-eval-setup.md for per-policy launch commands.
"""

import os

# The robot box has no outbound internet. Without this, loading a pi05_evo
# checkpoint tries to fetch its paligemma tokenizer from huggingface.co and dies
# on DNS resolution — the weights are all local, only the hub lookup isn't.
# setdefault, so an explicit HF_HUB_OFFLINE=0 still wins. Same line eval_sim.py
# carries for the same reason.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ROS's boost on LD_LIBRARY_PATH shadows pinocchio's bundled boost, and the
# Execution process imports PinocchioKinematics for the ground clamp. Must run
# before any other import in this process (same as eval_sim.py).
from fabricbench._env import ensure_clean_ld_path

ensure_clean_ld_path()

# === lerobot imports ===
# NOTE: Import paths may be lerobot.common.xxx (older) or lerobot.xxx (newer).
# We try the current convention first, fall back to the older one.
try:
    from lerobot.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
    from lerobot.robots import dual_piper  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.pretrained import PreTrainedPolicy
except ImportError:
    from lerobot.common.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
    from lerobot.common.robots import dual_piper  # noqa: F401
    from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.control_utils import predict_action

# === piper sdk ===
from piper_sdk import *

import numpy as np
import torch
import time
import os
import signal
import threading
import multiprocessing
import logging
from contextlib import nullcontext
from copy import copy
from queue import Empty, Full, Queue

from fabricbench.bench.real.chunked_action_buffer import (
    ChunkedActionBuffer,
    postprocess_chunk,
    setup_rtc_on_policy,
)
from dataclasses import dataclass, field
from threading import BrokenBarrierError
from pynput import keyboard

logging.basicConfig(level=logging.INFO, format='[%(levelname)s|%(processName)s] %(message)s')

# ===================================================================
#                        Constants & Unit Conversion
# ===================================================================
# --- Unit chain documentation ---
#
# [SDK hardware]          [DualPiper driver]              [eval script]              [training data (sim)]
# joints: 0.001 degrees → / ARM_FACTOR → radians          radians (直接使用)          radians
# gripper: 0.001 mm      → / GRIPPER_SDK_MAX → [0,1]      [0,1] (qpos / grip_limit)
#
# SDK joint:  int32, 单位 0.001°,   范围 e.g. [-150000, 150000] for j1
# SDK gripper: int32, 单位 0.001mm, 范围 [0, 70000] (0~70mm)
#
# DualPiper driver conversion:
#   joint_rad = sdk_value / ARM_FACTOR    (ARM_FACTOR = 1000 * 180 / pi ≈ 57295.78)
#   gripper_norm = sdk_value / GRIPPER_SDK_MAX  (70mm = 70000 SDK units)
#
# Sim training data:
#   joint: radians (MuJoCo qpos)
#   gripper: qpos / grip_limit → [0, 1]  (grip_limit 通常为 0.035m)
#
# Execution (反向转换):
#   joint: radians × ARM_FACTOR → sdk units (0.001°)
#   gripper: [0,1] × 70 × 1000 → sdk units (0.001mm)

ARM_FACTOR = 57295.779513  # 1000 * 180 / pi, converts rad ↔ SDK 0.001° units
GRIPPER_SDK_MAX = 70.0 * 1000           # 真实 gripper 最大开合: 70mm = 70000 SDK units

# MIT 控制增益 (仅在 cfg.mit_control=True 时使用). 直接在脚本里编辑切换.
# 当前启用: 默认 (保守版)
MIT_KP = [5.5, 5.5, 5.5, 10.0, 10.0, 10.0]
MIT_KD = [0.35, 0.35, 0.35, 0.8, 0.8, 0.8]
# 与 piper_ctrl_single_node.py 一致的版本 (备用):
# MIT_KP = [10.0, 10.0, 10.0, 15.0, 15.0, 10.0]
# MIT_KD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def _send_arm_joints(piper, joints_rad, mit_control):
    """发送 6 关节命令: MIT 模式用 JointMitCtrl, 否则用位置模式 JointCtrl."""
    if mit_control:
        piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)  # MIT mode
        for i in range(6):
            piper.JointMitCtrl(i + 1, joints_rad[i], 0.0, MIT_KP[i], MIT_KD[i], 0.0)
    else:
        sdk_joints = [round(joints_rad[i] * ARM_FACTOR) for i in range(6)]
        piper.JointCtrl(*sdk_joints)


# Home poses are NOT duplicated here. Upstream carried a hand-copied `lift_silk`
# table ("from randomization.py _ROBOT_INIT_PRESETS, without noise"); this port
# imports that table directly instead (see _preset_home_14d) so the real rig
# cannot silently drift from the pose collection and sim eval start every episode
# at. The values were verified identical at port time.


def _preset_home_14d(name: str) -> list[float]:
    """Look up a start pose in the authoritative preset table and flatten to 14-d.

    Source of truth is ``task/randomization.py:_ROBOT_INIT_PRESETS`` — the same
    table ``apply_robot_init`` uses to place the arms at the start of every
    collection and sim-eval episode. Imported rather than copied so the real rig
    cannot drift from what the policies were trained on. Noise is deliberately
    NOT applied here: sim samples within ``rand_range`` per episode, the real rig
    starts at the un-noised centre of that distribution.
    """
    from fabricbench.bench.task.randomization import _ROBOT_INIT_PRESETS

    if name not in _ROBOT_INIT_PRESETS:
        raise ValueError(
            f"Unknown home preset {name!r}. Available: {', '.join(_ROBOT_INIT_PRESETS)}"
        )
    p = _ROBOT_INIT_PRESETS[name]
    g = float(p.get("gripper_open", 1.0))
    return [*(float(v) for v in p["home_left"]), g,
            *(float(v) for v in p["home_right"]), g]


def _task_yaml(task_name: str | None) -> dict:
    """Load ``config/task/<name>.yaml`` (empty dict when absent)."""
    if not task_name:
        return {}
    import yaml as _yaml

    path = os.path.join(_TASK_CONFIG_DIR, f"{task_name}.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return _yaml.safe_load(f) or {}


def resolve_perception_fps(cfg) -> int:
    """Observation rate: explicit --perception_fps, else the task's control_fps.

    Collection and sim eval both record/command at ``sim.control_fps`` (25 Hz
    across this benchmark), so the policy's action chunks are timed against
    observations that far apart. Running the real loop faster executes each
    chunk proportionally early.
    """
    if cfg.perception_fps is not None:
        logging.info("Perception rate: %d Hz (explicit --perception_fps)", cfg.perception_fps)
        return int(cfg.perception_fps)
    name = getattr(cfg, "task_name", None)
    fps = int(((_task_yaml(name).get("sim") or {}).get("control_fps") or 25))
    logging.info("Perception rate: %d Hz (from %s control_fps)", fps, f"{name}.yaml" if name else "default")
    return fps


def _task_start_pose(task_name: str | None) -> str | None:
    """Read ``robot_init.start_pose`` out of ``config/task/<name>.yaml``."""
    if not task_name:
        return None
    import yaml as _yaml

    path = os.path.join(_TASK_CONFIG_DIR, f"{task_name}.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = _yaml.safe_load(f) or {}
    return ((raw.get("randomization") or {}).get("robot_init") or {}).get("start_pose")


def _resolve_home_position(cfg):
    """Resolve the startup home pose.

    Precedence: explicit --home_preset > the task config's robot_init.start_pose
    > explicit --home_position > all-zeros.

    The task-config route matters: policies are trained from the task's init
    pose, so starting the real arms anywhere else (in particular the all-zeros
    default, which is the arm fully extended) puts the very first observation
    out of distribution.
    """
    if getattr(cfg, 'home_preset', None):
        home = _preset_home_14d(cfg.home_preset)
        logging.info("Home pose: preset %r (explicit --home_preset): %s",
                     cfg.home_preset, [f"{v:.4f}" for v in home])
        return home

    start_pose = _task_start_pose(getattr(cfg, 'task_name', None))
    if start_pose:
        home = _preset_home_14d(start_pose)
        logging.info("Home pose: preset %r (from %s.yaml robot_init.start_pose): %s",
                     start_pose, cfg.task_name, [f"{v:.4f}" for v in home])
        return home

    home = list(cfg.home_position)
    if not any(abs(v) > 1e-9 for v in home[:6] + home[7:13]):
        logging.warning(
            "Home pose is ALL ZEROS (arm fully extended) — no --home_preset and no "
            "task config to derive it from. Policies are trained from the task's "
            "init pose; the first observation will be out of distribution. "
            "Pass --task=<config name> or --home_preset=<name>."
        )
    else:
        logging.info("Home pose: explicit --home_position: %s", [f"{v:.4f}" for v in home])
    return home


def _read_current_position(piper_left, piper_right):
    """Read current joint positions (radians) + gripper (normalized) from CAN bus."""
    left_js = piper_left.GetArmJointMsgs().joint_state
    left_grip = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
    right_js = piper_right.GetArmJointMsgs().joint_state
    right_grip = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
    return [
        left_js.joint_1 / ARM_FACTOR,
        left_js.joint_2 / ARM_FACTOR,
        left_js.joint_3 / ARM_FACTOR,
        left_js.joint_4 / ARM_FACTOR,
        left_js.joint_5 / ARM_FACTOR,
        left_js.joint_6 / ARM_FACTOR,
        np.clip(left_grip / GRIPPER_SDK_MAX, 0.0, 1.0),  # [0,1] normalized
        right_js.joint_1 / ARM_FACTOR,
        right_js.joint_2 / ARM_FACTOR,
        right_js.joint_3 / ARM_FACTOR,
        right_js.joint_4 / ARM_FACTOR,
        right_js.joint_5 / ARM_FACTOR,
        right_js.joint_6 / ARM_FACTOR,
        np.clip(right_grip / GRIPPER_SDK_MAX, 0.0, 1.0),  # [0,1] normalized
    ]


def _slow_move_to_home(piper_left, piper_right, home_position, duration=3.0, control_freq=200, mit_control=False):
    """Smoothly interpolate from current position to home over `duration` seconds.

    Uses ease-in-out cubic: 3t^2 - 2t^3.
    Sends joint commands via SDK at `control_freq` Hz.
    """
    start_position = _read_current_position(piper_left, piper_right)
    logging.info(
        "Slow move to home: %.1fs, from current=%s",
        duration,
        [f"{v:.3f}" for v in start_position],
    )
    logging.info("  target home=%s", [f"{v:.3f}" for v in home_position])

    period = 1.0 / control_freq
    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= duration:
            break

        progress = elapsed / duration
        progress = 3 * progress**2 - 2 * progress**3  # ease-in-out

        position = [
            start_position[i] + progress * (home_position[i] - start_position[i])
            for i in range(len(home_position))
        ]

        # Send to left arm
        left_gripper = round(float(np.clip(abs(position[6]), 0.0, 1.0)) * 70 * 1000)
        _send_arm_joints(piper_left, position[0:6], mit_control)
        piper_left.GripperCtrl(left_gripper, 1000, 0x01, 0)

        # Send to right arm
        right_gripper = round(float(np.clip(abs(position[13]), 0.0, 1.0)) * 70 * 1000)
        _send_arm_joints(piper_right, position[7:13], mit_control)
        piper_right.GripperCtrl(right_gripper, 1000, 0x01, 0)

        time.sleep(period)

    # Send final home position
    left_gripper = round(float(np.clip(abs(home_position[6]), 0.0, 1.0)) * 70 * 1000)
    _send_arm_joints(piper_left, home_position[0:6], mit_control)
    piper_left.GripperCtrl(left_gripper, 1000, 0x01, 0)

    right_gripper = round(float(np.clip(abs(home_position[13]), 0.0, 1.0)) * 70 * 1000)
    _send_arm_joints(piper_right, home_position[7:13], mit_control)
    piper_right.GripperCtrl(right_gripper, 1000, 0x01, 0)

    logging.info("Slow move to home completed.")

JOINT_LIMITS = [
    (-2.687, 2.687),   # joint1
    (0.0, 3.403),      # joint2
    (-3.0541012, 0.0),  # joint3
    (-1.5499, 1.5499),  # joint4
    (-1.22, 1.22),      # joint5
    (-1.7452, 1.7452),  # joint6
]


# ===================================================================
#                        Config
# ===================================================================
@dataclass
class InferenceConfig:
    robot: RobotConfig | None = None
    policy: PreTrainedConfig | None = None
    ckpt_path: str | None = None

    # --- Task / language instruction ---
    # ``task`` is a task-config NAME (config/task/<name>.yaml), matching
    # eval_sim's --task. The instruction actually fed to the policy is resolved
    # from that yaml's ``task_description`` — the same field eval_sim reads
    # (eval_sim.py:205) and the same string the training datasets carry in
    # meta/tasks.parquet. ``task_instruction`` overrides it verbatim.
    #
    # NOTE this differs from the upstream deformable_bench script, where --task
    # WAS the literal instruction. Hand-typing it is how the fold pi05 eval
    # silently scored 13% (wrong instruction, no error) — resolving from yaml
    # makes that class of mistake impossible.
    task: str | None = None
    task_instruction: str | None = None
    # Set by main() from the original --task before cfg.task is rewritten to the
    # instruction; lets the Execution process derive the home pose from the same
    # task config (robot_init.start_pose).
    task_name: str | None = None

    # --- Policy backend selection (the ONLY addition to the original script) ---
    # "lerobot" = in-process LeRobot checkpoint (ACT / DP / Pi0.5 / SmolVLA),
    #   the original verified path; requires --policy.path.
    # "gr00t" / "lingbot" / "fastwam" = out-of-process servers reached via the
    #   shared bench/policy/ adapters; --policy.path is not used (each backend
    #   locates its checkpoint through its own flags, see policy_args).
    policy_type: str = "lerobot"
    # Verbatim passthrough of backend flags, parsed by the adapter's own
    # add_arguments() — same contract as benchmark.py's extra_args in sim eval,
    # so no adapter flag has to be re-declared here.
    #   e.g. --policy_args="--gr00t_port 5555 --gr00t_exec_horizon 16"
    policy_args: str = ""
    # Auto-spawn the backend's model server (reuses eval_sim's launchers) when
    # the corresponding --*_server_python is present in policy_args. Set False
    # to attach to a server you started yourself in another terminal.
    launch_server: bool = True

    # Camera exposure control
    manual_exposure: int | None = None  # None=auto, value=manual exposure (us)
    manual_gain: int | None = None      # None=auto, value=manual gain
    auto_white_balance_when_manual_exposure: bool = True

    # Camera visualization
    show_cameras: bool = True
    camera_window_name: str = "Real Cameras"
    camera_viz_height: int = 360

    # Execution
    control_freq: int = 200
    # None => derive from the task config's control_fps (25 Hz for every task in
    # this benchmark — collection and sim eval both record/command at that rate,
    # so the policy's action chunks assume observations that far apart). Upstream
    # hardcoded 30, which runs chunks ~20% fast. An explicit value always wins.
    perception_fps: int | None = None
    reset_duration: float = 3.0
    can_left: str = "can_left"   # CAN interface name for left arm
    can_right: str = "can_right"  # CAN interface name for right arm

    # MIT 控制 (默认关闭, 走位置模式 JointCtrl). 启用后改用 JointMitCtrl,
    # kp/kd 在脚本顶部 MIT_KP / MIT_KD 编辑.
    mit_control: bool = False

    # Debug flags
    debug_print_control_cmd: bool = False     # 打印执行层发送的 SDK 指令
    debug_print_observation: bool = False     # 打印感知层原始观测值
    debug_print_inference: bool = False       # 打印推理层输入(归一化后)和输出(反归一化后)
    debug_print_first_n: int = 30             # 启动后前 N 次每次都打印
    debug_print_every_n: int = 100            # 之后每 N 次打印一次

    # Home position: [left_j1~j6, left_gripper, right_j1~j6, right_gripper]
    home_position: list[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   # left arm
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,    # right arm
    ])
    home_preset: str | None = None              # preset name (e.g. "lift_silk"), overrides home_position
    startup_move_duration: float = 3.0          # seconds to interpolate from current pos to home at startup

    # Safety protection
    clamp_ground: bool = False                # 启用地面保护 (FK+IK 修正)
    min_ground_z: float = 0.001               # 地面高度阈值 (世界坐标 z, 默认地面下方1cm)
    urdf_path: str = "assets/robot/piper/urdf/piper_with_horizon_gripper.urdf"
    ee_link: str = "tcp_link"
    left_base_z: float = 0.0                  # 左臂基座 z 偏移 (世界坐标)
    right_base_z: float = 0.0                 # 右臂基座 z 偏移 (世界坐标)
    clip_gripper: bool = False                # 启用夹爪最小值保护
    grasp_close_val: float = 0.015            # 夹爪最小值 [0,1] 范围

    # Chunked inference (predict_action_chunk + overlap aggregation; optional RTC).
    # Default OFF: the old per-step predict_action path. Turn on for real-robot
    # deployment to (1) amortize ~200ms GPU inference across ~7 obs frames,
    # (2) optionally apply RTC guidance to keep the new chunk's prefix aligned
    # with the previous chunk's leftover, (3) mix stale/fresh predictions in
    # the overlap region via the chosen aggregate_fn. See module docstring.
    chunked_inference: bool = False
    aggregate_fn: str = "latest_only"         # latest_only / weighted_average / average / conservative / temporal_ensemble
    refill_threshold: float = 0.5             # trigger new chunk when queue_size/n_action_steps <= this
    temporal_ensemble_coeff: float = 0.01     # only used when aggregate_fn=temporal_ensemble
    rtc_enabled: bool = False                 # requires --chunked_inference and policy with rtc_config (Pi0.5)
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            config_json = os.path.join(policy_path, "config.json")
            if not os.path.exists(config_json):
                nested_policy_path = os.path.join(policy_path, "pretrained_model")
                nested_config_json = os.path.join(nested_policy_path, "config.json")
                if os.path.exists(nested_config_json):
                    logging.info(
                        "--policy.path does not contain config.json, auto-using nested pretrained_model: "
                        f"{nested_policy_path}"
                    )
                    policy_path = nested_policy_path

            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.ckpt_path is None and self.policy is not None:
            self.ckpt_path = getattr(self.policy, 'pretrained_path', None)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ===================================================================
#                    Helper Functions
# ===================================================================
def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


def enable_arm(piper, name: str = ""):
    """Enable arm with timeout (compatible with Piper V2)."""
    timeout = 8.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # V2 best practice from SDK demos: poll EnablePiper until enabled.
            if hasattr(piper, "EnablePiper"):
                is_enabled = bool(piper.EnablePiper())
            else:
                piper.EnableArm(7)
                is_enabled = True

            # If status API exists, require all motors enabled.
            if hasattr(piper, "GetArmEnableStatus"):
                status = piper.GetArmEnableStatus()
                is_enabled = is_enabled or (len(status) >= 6 and all(status))

            if is_enabled:
                # Ensure gripper servo is enabled.
                piper.GripperCtrl(70 * 1000, 1000, 0x01, 0)  # enable servo with open position
                logging.info(f"Arm {name} enabled.")
                return
        except Exception as e:
            logging.warning(f"Enable arm {name} retrying: {e}")

        logging.info(f"Enabling arm {name}...")
        time.sleep(0.1)

    raise RuntimeError(f"Arm {name} enable timeout!")


def load_policy(cfg: InferenceConfig, device: str = "cuda"):
    """Load policy with preprocessor/postprocessor (handles normalization).

    Returns (policy, config, preprocessor, postprocessor).
    """
    ckpt_path = cfg.ckpt_path
    policy_type = cfg.policy.type
    logging.info(f"Loading policy: type={policy_type}, path={ckpt_path}")

    policy_config = PreTrainedConfig.from_pretrained(ckpt_path)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(ckpt_path, config=policy_config)

    policy.eval()
    policy.to(device)
    policy.reset()

    # Load preprocessor / postprocessor (handles normalization / denormalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, pretrained_path=ckpt_path
    )

    logging.info(f"Policy loaded on {device} with pre/post processors.")

    # --- 打印归一化统计量, 便于验证 ---
    _log_normalization_stats(ckpt_path)

    return policy, policy_config, preprocessor, postprocessor


def _log_normalization_stats(ckpt_path: str):
    """打印 checkpoint 中保存的归一化统计量 (mean/std or min/max), 便于排查 sim2real 不匹配."""
    import json as _json
    for label, proc_name in [("PREPROCESSOR", "policy_preprocessor"),
                              ("POSTPROCESSOR", "policy_postprocessor")]:
        json_path = os.path.join(ckpt_path, f"{proc_name}.json")
        if not os.path.exists(json_path):
            logging.warning(f"  [{label}] {json_path} NOT FOUND — legacy checkpoint?")
            continue
        try:
            with open(json_path) as f:
                proc_cfg = _json.load(f)
            # 遍历 steps, 找到含 stats 的 normalizer/unnormalizer
            steps = proc_cfg.get("steps", [])
            for step in steps:
                step_type = step.get("type", "")
                stats = step.get("stats")
                if stats and ("normaliz" in step_type.lower() or "unnormaliz" in step_type.lower()):
                    logging.info(f"  [{label}] step={step_type}")
                    for feat_key, feat_stats in stats.items():
                        if isinstance(feat_stats, dict):
                            parts = []
                            for stat_name, stat_val in feat_stats.items():
                                if isinstance(stat_val, list):
                                    vals = [round(v, 4) for v in stat_val[:4]]
                                    suffix = f"...({len(stat_val)})" if len(stat_val) > 4 else ""
                                    parts.append(f"{stat_name}={vals}{suffix}")
                                else:
                                    parts.append(f"{stat_name}={stat_val}")
                            logging.info(f"    {feat_key}: {' | '.join(parts)}")
        except Exception as e:
            logging.warning(f"  [{label}] Failed to parse {json_path}: {e}")


def set_manual_exposure(robot, cfg: InferenceConfig):
    """Set manual exposure on all RealSense cameras after connection."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        logging.warning("pyrealsense2 not available, cannot set camera controls.")
        return

    for cam_key, cam in robot.cameras.items():
        try:
            device = cam.rs_profile.get_device()
            # Find the color sensor
            for sensor in device.query_sensors():
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    if cfg.manual_exposure is None:
                        logging.info(f"[{cam_key}] Auto exposure mode (manual_exposure not set)")
                    else:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.exposure, cfg.manual_exposure)
                        logging.info(f"[{cam_key}] Manual exposure={cfg.manual_exposure}us")
                        if cfg.manual_gain is not None:
                            sensor.set_option(rs.option.gain, cfg.manual_gain)
                            logging.info(f"[{cam_key}] Manual gain={cfg.manual_gain}")

                        if cfg.auto_white_balance_when_manual_exposure:
                            try:
                                sensor.set_option(rs.option.enable_auto_white_balance, 1)
                                logging.info(f"[{cam_key}] Auto white balance enabled")
                            except Exception as wb_err:
                                logging.warning(f"[{cam_key}] Failed to enable auto white balance: {wb_err}")
                    break
        except Exception as e:
            logging.warning(f"[{cam_key}] Failed to set exposure: {e}")


# Expected key order from DualPiper driver (Python dict insertion order):
# [left_joint_1..6.pos, left_gripper.pos, right_joint_1..6.pos, right_gripper.pos]
# → observation.state indices: [0:6]=left_joints, [6]=left_gripper, [7:13]=right_joints, [13]=right_gripper
STATE_LABELS = [
    "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip",
    "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip",
]


def build_raw_observation(observation: dict, debug: bool = False) -> dict[str, np.ndarray]:
    """Convert raw robot observation to numpy dict (lerobot format, before normalization).

    DualPiper driver 返回的 observation:
        - 'left_joint_1.pos' ~ 'right_gripper.pos': float (joints 弧度, gripper [0, 1])
        - camera keys: ndarray (H, W, 3) uint8 RGB

    转换后:
        - 'observation.state': (14,) float32 — joints 弧度, gripper [0, 1]
        - 'observation.images.<cam_key>': (H, W, 3) uint8
    """
    obs_frame = {}
    state_keys = []
    state_values_raw = []
    state_values = []

    for key, value in observation.items():
        if key.endswith('.pos'):
            raw_val = float(value)
            val = np.float32(value)
            if 'gripper' in key:
                val = np.clip(val, 0.0, 1.0)
            state_keys.append(key)
            state_values_raw.append(raw_val)
            state_values.append(float(val))
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            obs_frame[f'observation.images.{key}'] = value

    obs_frame['observation.state'] = np.array(state_values, dtype=np.float32)

    if debug:
        # 打印每个 motor key → raw driver value → normalized value
        parts = []
        for k, raw, rescaled in zip(state_keys, state_values_raw, state_values):
            short = k.replace('.pos', '').replace('left_', 'L_').replace('right_', 'R_').replace('joint_', 'j')
            if 'gripper' in k:
                parts.append(f"{short}: driver={raw:.4f} → rescaled={rescaled:.4f}")
            else:
                parts.append(f"{short}: {rescaled:.4f} rad ({np.degrees(rescaled):.1f}°)")
        logging.info("[OBS RAW] state(14d): %s", " | ".join(parts))
        for k, v in obs_frame.items():
            if 'image' in k:
                logging.info("[OBS RAW] %s: shape=%s dtype=%s range=[%d, %d]",
                             k, v.shape, v.dtype, v.min(), v.max())

    return obs_frame


# ===================================================================
#              Ground Penetration Protection (FK+IK)
# ===================================================================
# NOTE: 当前只检查固定 tcp frame，未考虑夹爪开合状态。
# 夹爪张开时 link7/link8 实际位置比固定 tcp_link_bottom/top 更靠外，
# 极端倾斜+全开时最多差 ~2cm (0.035m * sin(45°))。
# 真机有物理桌面兜底，暂不改。如需精确检查，需传入 gripper_val 做 FK。
_GROUND_CHECK_FRAMES = ["tcp_link", "tcp_link_bottom", "tcp_link_top", "tcp_link_center"]


def _correct_ground_penetration(kin, q_cmd, current_q, base_z: float = 0.0, min_ground_z: float = -0.01):
    """Check FK for gripper frames. If any z < min_ground_z (world frame), clamp via IK.

    Args:
        kin: PinocchioKinematics instance (single arm)
        q_cmd: Policy's target joint angles (6,)
        current_q: Current joint angles (6,) for fallback
        base_z: Z offset of the arm base in world frame
        min_ground_z: Minimum allowed z in world frame (default -0.01 = 1cm below floor)

    Returns:
        (corrected_q, info_str_or_None)
    """
    min_z_world = float("inf")
    min_frame = ""
    for frame in _GROUND_CHECK_FRAMES:
        pose = kin.solve_fk(q_cmd, ee_link=frame)
        z_world = pose[2] + base_z
        if z_world < min_z_world:
            min_z_world = z_world
            min_frame = frame

    if min_z_world >= min_ground_z:
        return q_cmd, None

    target_pose = kin.solve_fk(q_cmd, ee_link=min_frame).copy()
    target_pose[2] = min_ground_z - base_z  # world z=min_ground_z → base-frame z

    q_corrected = kin.solve_ik(
        target_pose,
        seed_joints=q_cmd,
        ee_link=min_frame,
        strict_seed=True,
        ground_check=False,
    )

    if q_corrected is not None:
        return q_corrected, f"{min_frame} z_world={min_z_world:.4f} → IK corrected"
    else:
        return current_q.copy(), f"{min_frame} z_world={min_z_world:.4f} → IK failed, frozen"


# ===================================================================
#              Perception Process
# ===================================================================
def perception_process_main(
    perception_queue: multiprocessing.Queue,
    cfg: InferenceConfig,
    stop_event: multiprocessing.Event,
    barrier: multiprocessing.Barrier,
):
    cv2 = None
    cv2_gui_enabled = False
    if cfg.show_cameras:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            cv2_gui_enabled = True
        except Exception as e:
            logging.warning(f"OpenCV GUI unavailable, camera visualization disabled: {e}")

    def _build_camera_panel(obs_frame: dict):
        if cv2 is None:
            return None
        camera_items = [
            (k, v) for k, v in obs_frame.items()
            if k.startswith("observation.images.") and isinstance(v, np.ndarray) and v.ndim == 3
        ]
        if not camera_items:
            return None

        camera_items.sort(key=lambda x: x[0])
        vis_images = []
        target_h = max(80, int(cfg.camera_viz_height))

        for cam_key, img_rgb in camera_items:
            if img_rgb is None or img_rgb.size == 0:
                continue

            # obs_frame images are raw numpy (H, W, 3) uint8 RGB.
            # OpenCV expects BGR for correct display colors.
            img = img_rgb[:, :, ::-1].copy()
            img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img

            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                continue
            target_w = max(80, int(w * target_h / h))
            vis = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                vis,
                cam_key.replace("observation.images.", ""),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            vis_images.append(vis)

        if not vis_images:
            return None

        return np.concatenate(vis_images, axis=1)

    try:
        logging.info("Initializing robot and cameras...")
        robot = make_robot_from_config(cfg.robot)
        if not robot.is_connected:
            robot.connect()

        # Set manual exposure after camera connection
        set_manual_exposure(robot, cfg)

        logging.info("Perception READY. Waiting for start signal.")
        barrier.wait()  # Phase 1: report ready
        barrier.wait()  # Phase 2: wait for go signal

    except Exception as e:
        logging.error("Perception init failed:", exc_info=True)
        barrier.abort()
        return

    logging.info("STARTING perception loop.")
    target_period = 1.0 / cfg.perception_fps
    obs_count = 0

    while not stop_event.is_set():
        loop_start = time.perf_counter()

        observation = robot.get_observation()
        obs_count += 1
        should_debug = cfg.debug_print_observation and (
            obs_count <= cfg.debug_print_first_n
            or (cfg.debug_print_every_n > 0 and obs_count % cfg.debug_print_every_n == 0)
        )
        obs_frame = build_raw_observation(observation, debug=should_debug)

        # Optional camera visualization
        if cv2_gui_enabled:
            try:
                panel = _build_camera_panel(obs_frame)
                if panel is not None:
                    cv2.imshow(cfg.camera_window_name, panel)
                    cv2.waitKey(1)
            except Exception as e:
                logging.warning(f"Camera visualization disabled due to runtime error: {e}")
                cv2_gui_enabled = False

        # Put into queue (overwrite old data)
        while True:
            try:
                perception_queue.get_nowait()
            except Empty:
                break
        try:
            perception_queue.put_nowait(obs_frame)
        except Full:
            pass

        elapsed = time.perf_counter() - loop_start
        sleep_time = target_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    try:
        if cv2_gui_enabled and cv2 is not None:
            cv2.destroyWindow(cfg.camera_window_name)
            cv2.destroyAllWindows()
        robot.disconnect()
    except Exception:
        pass
    logging.info("Perception process shut down.")


# ===================================================================
#              Task instruction resolution
# ===================================================================
_TASK_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "task")


def _yaml_task_description(name: str) -> str | None:
    """Read ``task_description`` out of ``config/task/<name>.yaml``.

    Deliberately a plain yaml read rather than ``TaskSpec.from_name``: TaskSpec
    imports the physics stack (warp/newton), which the real-robot env has no
    reason to carry. Task configs are flat with no inheritance, so reading the
    single key is equivalent to what TaskSpec would return.
    """
    import yaml as _yaml

    path = os.path.join(_TASK_CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        avail = sorted(f[:-5] for f in os.listdir(_TASK_CONFIG_DIR) if f.endswith(".yaml"))
        raise ValueError(
            f"task config not found: {path}\nAvailable --task names: {', '.join(avail)}"
        )
    with open(path) as f:
        raw = _yaml.safe_load(f) or {}
    return raw.get("task_description")


def resolve_task_instruction(cfg: "InferenceConfig") -> str:
    """Return the language instruction to feed the policy.

    --task_instruction wins when given (and warns if it disagrees with the yaml);
    otherwise the instruction comes from ``config/task/<--task>.yaml``.
    """
    if cfg.task_instruction:
        if cfg.task:
            from_yaml = _yaml_task_description(cfg.task)
            if from_yaml and from_yaml != cfg.task_instruction:
                logging.warning(
                    "--task_instruction does NOT match %s.yaml task_description — "
                    "using the override, but a VLA trained on the yaml string will "
                    "see a different task:\n  given: %r\n  yaml : %r",
                    cfg.task, cfg.task_instruction, from_yaml,
                )
        return cfg.task_instruction

    if not cfg.task:
        return ""

    desc = _yaml_task_description(cfg.task)
    if not desc:
        raise ValueError(
            f"config/task/{cfg.task}.yaml has no `task_description`. "
            f"Pass --task_instruction=... explicitly."
        )
    return desc


# ===================================================================
#         Policy Adapter Backends (gr00t / lingbot / fastwam)
# ===================================================================
# Everything in this section is NEW relative to the original script. It is the
# single seam that lets the out-of-process benchmark policies run on hardware.
# The verified LeRobot path below is untouched by it.

_ADAPTER_BACKENDS = ("gr00t", "lingbot", "fastwam")


def build_adapter_args(cfg: InferenceConfig):
    """Parse ``--policy_args`` with the chosen adapter's own ``add_arguments``.

    Verbatim passthrough, same contract as ``benchmark.py``'s ``extra_args`` in
    sim eval, so backend flags never need re-declaring here. ``checkpoint`` /
    ``device`` / ``task`` are seeded from the top-level config because the
    shared server launchers and the lingbot prompt read them off this namespace.
    """
    import argparse
    import shlex

    from fabricbench.bench.policy._base import add_policy_arguments

    p = argparse.ArgumentParser(prog=f"--policy_args[{cfg.policy_type}]", add_help=False)
    p.add_argument("--checkpoint", default=cfg.ckpt_path)
    p.add_argument("--device", default="cuda")
    add_policy_arguments(p, cfg.policy_type)

    raw_args = cfg.policy_args or ""
    ns, unknown = p.parse_known_args(shlex.split(raw_args))
    if unknown:
        raise ValueError(
            f"unrecognized --policy_args for --policy_type={cfg.policy_type}: {unknown}. "
            f"Valid flags are whatever policy/{cfg.policy_type} declares in add_arguments()."
        )
    # cfg.task holds the RESOLVED instruction by the time this runs (main()
    # rewrites it). lingbot's build() already reads task/task_description.
    ns.task = cfg.task or ""
    ns.task_description = cfg.task or ""

    # gr00t and fastwam take their prompt from a backend-specific flag whose
    # argparse default is a hardcoded silk string — if left alone, a fold or bag
    # run would silently send "dual-arm grasp silk and lift" to the VLA. Push the
    # resolved instruction into those flags unless the caller set them by hand.
    for flag in (f"{cfg.policy_type}_task",):
        if not hasattr(ns, flag):
            continue
        if f"--{flag}" in raw_args:
            logging.info("policy_args sets --%s explicitly: %r", flag, getattr(ns, flag))
        elif cfg.task:
            logging.info("propagating resolved instruction into --%s: %r", flag, cfg.task)
            setattr(ns, flag, cfg.task)
    return ns


def build_adapter_policy(cfg: InferenceConfig):
    """Construct the shared bench/policy adapter for an out-of-process backend."""
    from fabricbench.bench.policy._base import load_policy as load_adapter

    adapter = load_adapter(cfg.policy_type, build_adapter_args(cfg))
    logging.info(
        "Policy adapter ready: %s (needs_images=%s, needs_language=%s, needs_point_cloud=%s)",
        cfg.policy_type,
        getattr(adapter, "needs_images", True),
        getattr(adapter, "needs_language", False),
        getattr(adapter, "needs_point_cloud", False),
    )
    return adapter


def adapter_inference_loop(
    adapter,
    perception_queue: multiprocessing.Queue,
    action_queue: multiprocessing.Queue,
    control_queue: multiprocessing.Queue,
    infer_counter,
    execution_ready,
    cfg: InferenceConfig,
    stop_event: multiprocessing.Event,
):
    """Inference loop for adapter backends.

    Mirrors the baseline (per-step) LeRobot loop below — same control commands,
    same debug prints, same joint-limit warnings, same overwrite-latest action
    queue. It differs only in the predict call: these backends normalize and
    chunk internally and expose a single ``predict(obs, task) -> (14,)``, so
    there is no preprocessor/postprocessor/ChunkedActionBuffer here.
    """
    logging.info("STARTING inference loop (%s adapter).", cfg.policy_type)
    inference_paused = False
    was_ready = False
    infer_count = 0
    task_desc = cfg.task or ""

    while not stop_event.is_set():
        try:
            cmd = control_queue.get_nowait()
            if cmd == 'reset_home':
                pass  # Inference continues
            elif cmd == 'reset_pause_resume':
                inference_paused = not inference_paused
                status = "PAUSED" if inference_paused else "RESUMED"
                logging.info(f"Inference {status}")
                if not inference_paused:
                    adapter.reset()
            elif cmd == 'quit':
                stop_event.set()
                break
        except Empty:
            pass

        if inference_paused:
            time.sleep(0.1)
            continue

        # Handoff gate. While the execution loop interpolates to home (startup, or
        # an 'r'/SPACE reset) it does NOT drain action_queue -- but the policy
        # replays its action chunk open-loop, popping one step per observation.
        # Left free-running the chunk advances in TIME while the arm stays parked,
        # so the first action execution finally applies is N steps into the
        # trajectory. Applied as a single absolute joint target with no rate
        # limit, that is the startup/resume lurch (worse the longer the chunk:
        # act n_action_steps=100 = 4s, pi05 50 = 2s, dp 8 = 0.32s). Fix: produce
        # nothing until execution is really consuming, and start the chunk there.
        if not execution_ready.is_set():
            was_ready = False
            time.sleep(0.005)
            continue
        if not was_ready:
            was_ready = True
            adapter.reset()
            while True:                    # drop frames captured mid-motion
                try:
                    perception_queue.get_nowait()
                except Empty:
                    break
            logging.info("[HANDOFF] inference armed: policy reset, chunk starts now")

        try:
            obs_dict = perception_queue.get(timeout=0.1)
            if obs_dict is None:
                continue

            infer_count += 1
            with infer_counter.get_lock():
                infer_counter.value += 1
            should_debug = cfg.debug_print_inference and (
                infer_count <= cfg.debug_print_first_n
                or (cfg.debug_print_every_n > 0 and infer_count % cfg.debug_print_every_n == 0)
            )

            if should_debug:
                state_raw = obs_dict.get('observation.state')
                if state_raw is not None:
                    labels = STATE_LABELS if len(state_raw) == 14 else [f"s{i}" for i in range(len(state_raw))]
                    pairs = [f"{l}={v:.4f}" for l, v in zip(labels, state_raw)]
                    logging.info("[INF #%d] obs_raw state: %s", infer_count, " ".join(pairs))

            t_infer_start = time.perf_counter()
            numpy_action = np.asarray(
                adapter.predict(obs_dict, task_desc), dtype=np.float32
            ).reshape(-1)
            position = numpy_action.tolist()

            if should_debug:
                labels = STATE_LABELS if len(position) == 14 else [f"a{i}" for i in range(len(position))]
                parts = []
                for l, v in zip(labels, position):
                    if 'grip' in l:
                        parts.append(f"{l}={v:.4f}")
                    else:
                        parts.append(f"{l}={v:.4f}({np.degrees(v):.1f}°)")
                logging.info("[INF #%d] action_out(14d): %s", infer_count, " ".join(parts))
                logging.info("[INF #%d] inference latency: %.1fms",
                             infer_count, (time.perf_counter() - t_infer_start) * 1000)
                joints_l = position[:6]
                joints_r = position[7:13] if len(position) > 13 else []
                for i, (jl, jr) in enumerate(zip(joints_l, joints_r)):
                    lo, hi = JOINT_LIMITS[i]
                    if jl < lo or jl > hi:
                        logging.warning("[INF #%d] LEFT joint%d=%.4f OUT OF LIMITS [%.3f, %.3f]",
                                        infer_count, i+1, jl, lo, hi)
                    if jr < lo or jr > hi:
                        logging.warning("[INF #%d] RIGHT joint%d=%.4f OUT OF LIMITS [%.3f, %.3f]",
                                        infer_count, i+1, jr, lo, hi)

            # Put into action queue (overwrite old)
            while True:
                try:
                    action_queue.get_nowait()
                except Empty:
                    break
            try:
                action_queue.put_nowait(position)
            except Full:
                pass

        except Empty:
            pass

    logging.info("Inference process shut down.")


# ===================================================================
#              Inference Process
# ===================================================================
def inference_process_main(
    perception_queue: multiprocessing.Queue,
    action_queue: multiprocessing.Queue,
    control_queue: multiprocessing.Queue,
    infer_counter,
    execution_ready,
    cfg: InferenceConfig,
    stop_event: multiprocessing.Event,
    barrier: multiprocessing.Barrier,
):
    device = torch.device("cuda")
    action_buffer: ChunkedActionBuffer | None = None
    rtc_active = False

    # Out-of-process backends short-circuit here so that everything below stays
    # exactly as validated on the rig for the LeRobot path.
    if cfg.policy_type in _ADAPTER_BACKENDS:
        try:
            logging.info("Initializing policy adapter: %s", cfg.policy_type)
            adapter = build_adapter_policy(cfg)
            logging.info("Inference READY. Waiting for start signal.")
            barrier.wait()
            barrier.wait()
        except Exception:
            logging.error("Inference init failed:", exc_info=True)
            barrier.abort()
            return
        adapter_inference_loop(
            adapter, perception_queue, action_queue, control_queue,
            infer_counter, execution_ready, cfg, stop_event,
        )
        return

    try:
        logging.info("Initializing policy...")
        policy, policy_config, preprocessor, postprocessor = load_policy(cfg, device="cuda")

        # Optional chunked-inference pipeline: predict_action_chunk + overlap
        # aggregation + optional RTC guidance.
        if cfg.chunked_inference:
            if cfg.rtc_enabled:
                rtc_active = setup_rtc_on_policy(
                    policy,
                    execution_horizon=cfg.rtc_execution_horizon,
                    max_guidance_weight=cfg.rtc_max_guidance_weight,
                )
                if rtc_active:
                    logging.info(
                        "RTC enabled: execution_horizon=%d, max_guidance_weight=%.1f",
                        cfg.rtc_execution_horizon, cfg.rtc_max_guidance_weight,
                    )
                else:
                    logging.warning(
                        "--rtc_enabled but policy does not support RTC; continuing without RTC."
                    )
            _chunk_size = getattr(policy_config, "chunk_size", None) or policy_config.n_action_steps
            action_buffer = ChunkedActionBuffer(
                chunk_size=_chunk_size,
                n_action_steps=policy_config.n_action_steps,
                refill_threshold=cfg.refill_threshold,
                aggregate_fn=cfg.aggregate_fn,
                gripper_indices=(6, 13),
                ensemble_coeff=cfg.temporal_ensemble_coeff,
                fps=cfg.perception_fps,
                rtc_execution_horizon=cfg.rtc_execution_horizon,
            )
            logging.info(
                "Chunked inference ON: chunk_size=%d, n_action_steps=%d, "
                "aggregate_fn=%s, refill_threshold=%.2f, rtc_active=%s",
                _chunk_size, policy_config.n_action_steps,
                cfg.aggregate_fn, cfg.refill_threshold, rtc_active,
            )
        else:
            logging.info("Chunked inference OFF: per-step predict_action path (baseline).")

        logging.info("Inference READY. Waiting for start signal.")
        barrier.wait()
        barrier.wait()

    except Exception as e:
        logging.error("Inference init failed:", exc_info=True)
        barrier.abort()
        return

    # --- Log normalization config for verification ---
    norm_map = getattr(policy_config, 'normalization_mapping', {})
    logging.info("Policy normalization_mapping: %s", norm_map)
    logging.info("Policy input_features: %s",
                 {k: (v.type, v.shape) for k, v in policy_config.input_features.items()}
                 if hasattr(policy_config, 'input_features') else "N/A")
    logging.info("Policy output_features: %s",
                 {k: (v.type, v.shape) for k, v in policy_config.output_features.items()}
                 if hasattr(policy_config, 'output_features') else "N/A")

    logging.info("STARTING inference loop.")
    inference_paused = False
    was_ready = False
    use_amp = getattr(policy_config, 'use_amp', False)
    infer_count = 0
    chunk_count = 0  # how many times predict_action_chunk has fired (chunked mode only)

    while not stop_event.is_set():
        # Check control commands
        try:
            cmd = control_queue.get_nowait()
            if cmd == 'reset_home':
                pass  # Inference continues
            elif cmd == 'reset_pause_resume':
                inference_paused = not inference_paused
                status = "PAUSED" if inference_paused else "RESUMED"
                logging.info(f"Inference {status}")
                if not inference_paused:
                    policy.reset()
                    if action_buffer is not None:
                        action_buffer.reset()
            elif cmd == 'quit':
                stop_event.set()
                break
        except Empty:
            pass

        if inference_paused:
            time.sleep(0.1)
            continue

        # Handoff gate. While the execution loop interpolates to home (startup, or
        # an 'r'/SPACE reset) it does NOT drain action_queue -- but the policy
        # replays its action chunk open-loop, popping one step per observation.
        # Left free-running the chunk advances in TIME while the arm stays parked,
        # so the first action execution finally applies is N steps into the
        # trajectory. Applied as a single absolute joint target with no rate
        # limit, that is the startup/resume lurch (worse the longer the chunk:
        # act n_action_steps=100 = 4s, pi05 50 = 2s, dp 8 = 0.32s). Fix: produce
        # nothing until execution is really consuming, and start the chunk there.
        if not execution_ready.is_set():
            was_ready = False
            time.sleep(0.005)
            continue
        if not was_ready:
            was_ready = True
            policy.reset()
            if action_buffer is not None:
                action_buffer.reset()
            while True:                    # drop frames captured mid-motion
                try:
                    perception_queue.get_nowait()
                except Empty:
                    break
            logging.info("[HANDOFF] inference armed: policy reset, chunk starts now")

        try:
            obs_dict = perception_queue.get(timeout=0.1)
            if obs_dict is None:
                continue

            infer_count += 1
            with infer_counter.get_lock():
                infer_counter.value += 1
            should_debug = cfg.debug_print_inference and (
                infer_count <= cfg.debug_print_first_n
                or (cfg.debug_print_every_n > 0 and infer_count % cfg.debug_print_every_n == 0)
            )

            if should_debug:
                # --- 1. 打印 preprocessor 输入 (raw observation) ---
                state_raw = obs_dict.get('observation.state')
                if state_raw is not None:
                    labels = STATE_LABELS if len(state_raw) == 14 else [f"s{i}" for i in range(len(state_raw))]
                    pairs = [f"{l}={v:.4f}" for l, v in zip(labels, state_raw)]
                    logging.info("[INF #%d] obs_raw state: %s", infer_count, " ".join(pairs))

            t_infer_start = time.perf_counter()
            if action_buffer is not None:
                # Chunked path: do predict_action_chunk when buffer drains below
                # refill_threshold; otherwise just pop a cached action. This
                # amortizes ~200ms chunk inference across ~7 perception frames.
                if action_buffer.should_request_new_chunk():
                    start_ts = action_buffer.next_timestep_to_request()
                    rtc_kwargs = action_buffer.get_rtc_kwargs() if rtc_active else {}
                    obs_for_inf = copy(obs_dict)
                    # Use no_grad (not inference_mode) so RTC's internal
                    # torch.enable_grad() + autograd.grad can take effect.
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
                    ):
                        obs_for_inf = prepare_observation_for_inference(
                            obs_for_inf, device, cfg.task, None,
                        )
                        obs_for_inf = preprocessor(obs_for_inf)
                        t_chunk = time.perf_counter()
                        raw_chunk = policy.predict_action_chunk(obs_for_inf, **rtc_kwargs)
                        chunk_dt = time.perf_counter() - t_chunk
                        action_buffer.record_inference_time(chunk_dt)
                        post_chunk = postprocess_chunk(raw_chunk, postprocessor)
                    action_buffer.ingest(post_chunk, start_timestep=start_ts, raw_tensor=raw_chunk)
                    chunk_count += 1
                    if should_debug or chunk_count <= 3 or chunk_count % 20 == 0:
                        logging.info(
                            "[INF chunk #%d] predict_action_chunk dt=%.1fms queue=%d%s",
                            chunk_count, chunk_dt * 1000, action_buffer.queue_size,
                            " (RTC)" if rtc_kwargs else "",
                        )
                action_np, _ = action_buffer.pop_next()
                numpy_action = action_np
                position = numpy_action.tolist()
            else:
                # Baseline per-step path. predict_action:
                #   1. prepare_observation_for_inference: numpy→tensor, /255 HWC→CHW, 加 batch 维
                #   2. preprocessor: 归一化 (mean/std or min/max, 训练集统计量)
                #   3. policy.select_action: 在归一化空间推理
                #   4. postprocessor: 反归一化, 恢复到物理空间
                action = predict_action(
                    observation=obs_dict,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=use_amp,
                    task=cfg.task,
                )
                numpy_action = action.squeeze(0).cpu().numpy()
                position = numpy_action.tolist()

            if should_debug:
                # --- 2. 打印 postprocessor 输出 (denormalized action, 物理空间) ---
                labels = STATE_LABELS if len(position) == 14 else [f"a{i}" for i in range(len(position))]
                parts = []
                for l, v in zip(labels, position):
                    if 'grip' in l:
                        parts.append(f"{l}={v:.4f}")
                    else:
                        parts.append(f"{l}={v:.4f}({np.degrees(v):.1f}°)")
                logging.info("[INF #%d] action_out(14d): %s", infer_count, " ".join(parts))
                logging.info("[INF #%d] inference latency: %.1fms",
                             infer_count, (time.perf_counter() - t_infer_start) * 1000)
                # 检查 action 是否在合理范围内
                joints_l = position[:6]
                joints_r = position[7:13] if len(position) > 13 else []
                for i, (jl, jr) in enumerate(zip(joints_l, joints_r)):
                    lo, hi = JOINT_LIMITS[i]
                    if jl < lo or jl > hi:
                        logging.warning("[INF #%d] LEFT joint%d=%.4f OUT OF LIMITS [%.3f, %.3f]",
                                        infer_count, i+1, jl, lo, hi)
                    if jr < lo or jr > hi:
                        logging.warning("[INF #%d] RIGHT joint%d=%.4f OUT OF LIMITS [%.3f, %.3f]",
                                        infer_count, i+1, jr, lo, hi)

            # Put into action queue (overwrite old)
            while True:
                try:
                    action_queue.get_nowait()
                except Empty:
                    break
            try:
                action_queue.put_nowait(position)
            except Full:
                pass

        except Empty:
            pass

    logging.info("Inference process shut down.")


# ===================================================================
#              Execution Process
# ===================================================================
def execution_process_main(
    action_queue: multiprocessing.Queue,
    control_queue: multiprocessing.Queue,
    infer_counter,          # shared int: predictions the Inference process has made
    execution_ready,        # Event: set only while execution is consuming action_queue
    cfg: InferenceConfig,
    stop_event: multiprocessing.Event,
    barrier: multiprocessing.Barrier,
):
    _blind_at, _blind_t0 = 0, time.time()
    try:
        logging.info("Initializing piper_sdk arms...")

        piper_left = C_PiperInterface_V2(cfg.can_left)
        piper_left.ConnectPort()
        enable_arm(piper_left, "left")
        if not cfg.mit_control:
            piper_left.MotionCtrl_2(0x01, 0x01, 60, 0xAD)

        piper_right = C_PiperInterface_V2(cfg.can_right)
        piper_right.ConnectPort()
        enable_arm(piper_right, "right")
        if not cfg.mit_control:
            piper_right.MotionCtrl_2(0x01, 0x01, 60, 0xAD)

        if cfg.mit_control:
            logging.info("MIT control ENABLED — kp=%s, kd=%s", MIT_KP, MIT_KD)

        # Initialize kinematics for ground protection (if enabled)
        _kin = None
        if cfg.clamp_ground:
            from fabricbench.bench.robot.kinematics.pinocchio_kinematics import PinocchioKinematics
            from fabricbench.paths import REPO_ROOT
            _urdf = cfg.urdf_path
            if not os.path.isabs(_urdf):
                _urdf = os.path.join(str(REPO_ROOT), _urdf)
            _kin = PinocchioKinematics(_urdf, ee_link=cfg.ee_link)
            _kin.ground_check_enabled = False  # we handle ground check ourselves
            logging.info("Ground clamp enabled (FK+IK correction, base_z: L=%.3f, R=%.3f, min_ground_z=%.3f)",
                         cfg.left_base_z, cfg.right_base_z, cfg.min_ground_z)

        home_position = _resolve_home_position(cfg)

        logging.info("Execution READY. Waiting for start signal.")
        barrier.wait()
        barrier.wait()

        # Slow move from current position to home
        if cfg.startup_move_duration > 0:
            _blind_at, _blind_t0 = int(infer_counter.value), time.time()
            _slow_move_to_home(
                piper_left, piper_right, home_position,
                duration=cfg.startup_move_duration,
                control_freq=cfg.control_freq,
                mit_control=cfg.mit_control,
            )

    except Exception as e:
        logging.error("Execution init failed:", exc_info=True)
        barrier.abort()
        return

    while True:              # stale actions from before the blind window
        try:
            action_queue.get_nowait()
        except Empty:
            break
    execution_ready.set()   # arm is parked at home; inference may start its chunk

    logging.info("STARTING execution loop.")
    if cfg.debug_print_control_cmd:
        logging.info(
            "Control debug print enabled: first %d commands, then every %d commands.",
            cfg.debug_print_first_n,
            cfg.debug_print_every_n,
        )

    # SDK sender thread to isolate blocking SDK calls
    sdk_command_queue = Queue(maxsize=1)

    def sdk_sender_thread():
        sent_count = 0
        while True:
            position = sdk_command_queue.get()
            if position is None:
                break
            try:
                # Left arm (indices 0-5 joints, 6 gripper)
                left_norm_joints = [
                    clamp(position[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
                    for i in range(6)
                ]
                left_joints = [
                    round(left_norm_joints[i] * ARM_FACTOR)
                    for i in range(6)
                ]
                left_norm_gripper = float(np.clip(abs(position[6]), 0.0, 1.0))
                left_gripper = round(left_norm_gripper * 70 * 1000)  # normalized [0,1] -> SDK unit
                _send_arm_joints(piper_left, left_norm_joints, cfg.mit_control)
                piper_left.GripperCtrl(left_gripper, 1000, 0x01, 0)

                # Right arm (indices 7-12 joints, 13 gripper)
                right_norm_joints = [
                    clamp(position[7 + i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
                    for i in range(6)
                ]
                right_joints = [
                    round(right_norm_joints[i] * ARM_FACTOR)
                    for i in range(6)
                ]
                right_norm_gripper = float(np.clip(abs(position[13]), 0.0, 1.0))
                right_gripper = round(right_norm_gripper * 70 * 1000)  # normalized [0,1] -> SDK unit
                _send_arm_joints(piper_right, right_norm_joints, cfg.mit_control)
                piper_right.GripperCtrl(right_gripper, 1000, 0x01, 0)

                sent_count += 1
                if cfg.debug_print_control_cmd and (
                    sent_count <= max(0, int(cfg.debug_print_first_n))
                    or (int(cfg.debug_print_every_n) > 0 and sent_count % int(cfg.debug_print_every_n) == 0)
                ):
                    # 逐关节打印: rad → deg → sdk, 便于对照检查
                    def _fmt_joints(name, norm_joints, sdk_joints, grip_norm, grip_sdk):
                        lines = [f"  {name} joints (rad→°→sdk):"]
                        for ji, (rad, sdk) in enumerate(zip(norm_joints, sdk_joints)):
                            deg = np.degrees(rad)
                            lo, hi = JOINT_LIMITS[ji]
                            clamped = " [CLAMPED]" if rad <= lo or rad >= hi else ""
                            lines.append(f"    j{ji+1}: {rad:+.4f} rad = {deg:+.1f}° → sdk {sdk}{clamped}")
                        lines.append(f"  {name} gripper: {grip_norm:.4f} [0-1] → sdk {grip_sdk} (={grip_sdk/1000:.1f}mm)")
                        return "\n".join(lines)

                    logging.info(
                        "[CMD #%d] Execution → SDK conversion:\n%s\n%s",
                        sent_count,
                        _fmt_joints("LEFT", left_norm_joints, left_joints, left_norm_gripper, left_gripper),
                        _fmt_joints("RIGHT", right_norm_joints, right_joints, right_norm_gripper, right_gripper),
                    )

            except Exception as e:
                logging.error(f"SDK sender error: {e}")
            sdk_command_queue.task_done()

        logging.info("SDK sender thread shut down.")

    sender = threading.Thread(target=sdk_sender_thread, name="SDK_Sender", daemon=True)
    sender.start()

    # Control state
    target_period = 1.0 / cfg.control_freq
    last_action = list(home_position)
    current_action = list(home_position)
    is_resetting = False
    reset_start_time = 0.0
    execution_paused = False
    ground_clamp_count = 0
    gripper_clipped_warned = False
    handoff_pending, handoff_label = True, "startup"
    blind_at, blind_t0 = _blind_at, _blind_t0

    while not stop_event.is_set():
        loop_start = time.perf_counter()

        # Check control commands
        try:
            cmd = control_queue.get_nowait()
            if cmd == 'reset_home':
                is_resetting = True
                reset_start_time = time.time()
                current_action = list(last_action)
                logging.info("Starting reset to home position...")
                execution_ready.clear()   # hold inference for the interpolation
                blind_at, blind_t0 = int(infer_counter.value), time.time()
                handoff_label = "reset(r)"
            elif cmd == 'reset_pause_resume':
                is_resetting = True
                reset_start_time = time.time()
                current_action = list(last_action)
                execution_paused = not execution_paused
                status = "PAUSED" if execution_paused else "RESUMED"
                logging.info(f"Reset to home + execution {status}")
                execution_ready.clear()   # hold inference for the interpolation
                blind_at, blind_t0 = int(infer_counter.value), time.time()
                handoff_label = "resume(SPACE)"
            elif cmd == 'quit':
                stop_event.set()
                break
        except Empty:
            pass

        # Handle reset interpolation
        if is_resetting:
            elapsed_reset = time.time() - reset_start_time
            if elapsed_reset < cfg.reset_duration:
                progress = elapsed_reset / cfg.reset_duration
                progress = 3 * progress**2 - 2 * progress**3  # ease-in-out
                last_action = [
                    current_action[i] + progress * (home_position[i] - current_action[i])
                    for i in range(len(home_position))
                ]
            else:
                last_action = list(home_position)
                is_resetting = False
                logging.info("Reset to home position completed.")
                if not execution_paused:
                    while True:              # stale actions from before the blind window
                        try:
                            action_queue.get_nowait()
                        except Empty:
                            break
                    execution_ready.set()
                    handoff_pending = True
        else:
            if not execution_paused:
                try:
                    new_action = action_queue.get_nowait()
                    last_action = new_action
                    if handoff_pending:
                        handoff_pending = False
                        _n = int(infer_counter.value) - int(blind_at)
                        _d = [abs(float(a) - float(h))
                              for a, h in zip(last_action, home_position)]
                        _arm = _d[:6] + _d[7:13]
                        _j = int(np.argmax(_arm))
                        logging.warning(
                            "[HANDOFF %s] consuming after %.2fs blind; inference "
                            "free-ran %d predictions; first action vs home: max "
                            "joint delta %.4f rad (%.1f deg) at %s, grippers "
                            "dL=%.3f dR=%.3f",
                            handoff_label, time.time() - blind_t0, _n,
                            _arm[_j], np.degrees(_arm[_j]),
                            STATE_LABELS[_j if _j < 6 else _j + 1], _d[6], _d[13])
                        logging.warning("[HANDOFF %s] |action-home| L=%s R=%s",
                                        handoff_label,
                                        [f"{v:.3f}" for v in _d[:6]],
                                        [f"{v:.3f}" for v in _d[7:13]])
                except Empty:
                    pass

                # Gripper protection: clamp to minimum value
                if cfg.clip_gripper:
                    l_before, r_before = float(last_action[6]), float(last_action[13])
                    last_action[6] = max(last_action[6], cfg.grasp_close_val)    # left gripper
                    last_action[13] = max(last_action[13], cfg.grasp_close_val)  # right gripper
                    if not gripper_clipped_warned and (
                        last_action[6] != l_before or last_action[13] != r_before
                    ):
                        gripper_clipped_warned = True
                        logging.warning(
                            "[EXEC] Gripper clipped: L %.3f→%.3f, R %.3f→%.3f (close_val=%.3f)",
                            l_before, last_action[6], r_before, last_action[13], cfg.grasp_close_val,
                        )

                # Ground penetration protection: FK check → IK correction
                if _kin is not None:
                    q_l = np.array(last_action[:6])
                    q_r = np.array(last_action[7:13])
                    q_l_corrected, info_l = _correct_ground_penetration(
                        _kin, q_l, q_l.copy(), cfg.left_base_z, cfg.min_ground_z)
                    q_r_corrected, info_r = _correct_ground_penetration(
                        _kin, q_r, q_r.copy(), cfg.right_base_z, cfg.min_ground_z)
                    if info_l or info_r:
                        ground_clamp_count += 1
                        if ground_clamp_count <= 5:
                            parts = []
                            if info_l: parts.append(f"L:{info_l}")
                            if info_r: parts.append(f"R:{info_r}")
                            logging.warning("[EXEC] Ground clamp #%d: %s",
                                            ground_clamp_count, ", ".join(parts))
                        elif ground_clamp_count == 6:
                            logging.warning("[EXEC] Suppressing further ground clamp warnings...")
                    last_action[:6] = q_l_corrected.tolist()
                    last_action[7:13] = q_r_corrected.tolist()
            else:
                last_action = list(home_position)

        # Send to SDK thread

        while True:
            try:
                sdk_command_queue.get_nowait()
            except Empty:
                break
        try:
            sdk_command_queue.put_nowait(last_action)
        except Full:
            pass

        elapsed = time.perf_counter() - loop_start
        sleep_time = target_period - elapsed
        if sleep_time < 0:
            logging.warning(f"Execution loop overrun by {-sleep_time*1000:.2f}ms")
        else:
            time.sleep(sleep_time)

    # Shutdown
    sdk_command_queue.put(None)
    sender.join(timeout=2)
    logging.info("Execution process shut down.")


# ===================================================================
#              Keyboard Process
# ===================================================================
def keyboard_process_main(
    control_queues: list[multiprocessing.Queue],
    stop_event: multiprocessing.Event,
):
    logging.info("Keyboard listener started.")
    logging.info("Controls:")
    logging.info("  'r'     - Reset to home position (inference continues)")
    logging.info("  'SPACE' - Reset to home + PAUSE/RESUME inference & execution")
    logging.info("  'q'     - Quit program")

    def on_press(key):
        command = None
        try:
            if key.char == 'r':
                command = 'reset_home'
                logging.info("[Keyboard] 'r' pressed -> reset_home")
            elif key.char == 'q':
                command = 'quit'
                logging.info("[Keyboard] 'q' pressed -> quit")
        except AttributeError:
            if key == keyboard.Key.space:
                command = 'reset_pause_resume'
                logging.info("[Keyboard] SPACE pressed -> reset_pause_resume")

        if command:
            for q in control_queues:
                try:
                    q.put_nowait(command)
                except Full:
                    logging.warning(f"Control queue full, command '{command}' may be missed.")
            if command == 'quit':
                return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    stop_event.wait()
    listener.stop()
    logging.info("Keyboard listener shut down.")


# ===================================================================
#              Main
# ===================================================================
@parser.wrap()
def main(cfg: InferenceConfig):
    if cfg.policy_type not in ("lerobot",) + _ADAPTER_BACKENDS:
        raise ValueError(
            f"Unknown --policy_type={cfg.policy_type!r}. "
            f"Valid: lerobot (ACT/DP/Pi0.5/SmolVLA, in-process), {', '.join(_ADAPTER_BACKENDS)}."
        )
    if cfg.policy_type == "lerobot":
        if cfg.policy is None:
            raise ValueError("Policy path required. Use --policy.path=<path>")
    else:
        # No silent feature loss: these backends drive their own action chunking
        # inside the model server and expose no predict_action_chunk / RTC hook,
        # so the flags cannot be honoured and must not be quietly ignored.
        if cfg.chunked_inference or cfg.rtc_enabled:
            raise ValueError(
                f"--chunked_inference / --rtc_enabled are only supported by "
                f"--policy_type=lerobot (they need predict_action_chunk + the "
                f"pre/post processors). {cfg.policy_type} chunks internally in "
                f"its own server; drop these flags."
            )
    if cfg.robot is None:
        raise ValueError("Robot config required. Use --robot.type=dual_piper")

    # Resolve --task (a config name) into the instruction, then write it back
    # into cfg.task so every downstream consumer — predict_action(task=...), the
    # adapter namespace, adapter_inference_loop — sees the instruction with no
    # further changes. cfg is pickled to the child processes after this point.
    cfg.task_name = cfg.task
    task_name = cfg.task
    cfg.perception_fps = resolve_perception_fps(cfg)
    cfg.task = resolve_task_instruction(cfg)
    logging.info("Task config : %s", task_name or "(none)")
    logging.info("Instruction : %r%s", cfg.task,
                 "  [--task_instruction override]" if cfg.task_instruction else "  [from yaml]")
    if not cfg.task:
        logging.warning("No task instruction resolved — a VLA (pi05/groot/lingbot) "
                        "will receive an EMPTY instruction. Pass --task=<config name>.")

    # Auto-spawn the backend's model server (same launchers sim eval uses) and
    # tear it down in the finally below. Skipped when --launch_server=False or
    # when policy_args carries no --*_server_python (= attach to a running one).
    server_proc = None
    if cfg.policy_type in _ADAPTER_BACKENDS and cfg.launch_server:
        from loguru import logger as _logger

        from fabricbench.bench.policy.servers import (
            _launch_fastwam_server,
            _launch_gr00t_server,
            _launch_lingbot_server,
        )

        # Closing the terminal SIGHUPs us, and the default action is to die without
        # unwinding -- while the server, in its own session, never gets the signal
        # and outlives us holding the port. Turn both signals into an exception so
        # the finally below actually runs and takes the server with it.
        def _signal_to_exception(signum, _frame):
            raise KeyboardInterrupt(f"received signal {signum}")

        for _sig in (signal.SIGHUP, signal.SIGTERM):
            signal.signal(_sig, _signal_to_exception)

        _launchers = {
            "gr00t": _launch_gr00t_server,
            "lingbot": _launch_lingbot_server,
            "fastwam": _launch_fastwam_server,
        }
        server_proc = _launchers[cfg.policy_type](build_adapter_args(cfg), _logger)

    # Create IPC objects
    stop_event = multiprocessing.Event()
    perception_q = multiprocessing.Queue(maxsize=1)
    action_q = multiprocessing.Queue(maxsize=1)
    control_q_execution = multiprocessing.Queue(maxsize=10)
    control_q_inference = multiprocessing.Queue(maxsize=10)
    infer_counter = multiprocessing.Value("i", 0)
    # Set only while the execution loop is consuming action_queue, so the policy's
    # action chunk cannot advance while the arm is parked. See the handoff gate.
    execution_ready = multiprocessing.Event()
    all_control_queues = [control_q_execution, control_q_inference]

    # Barrier: 3 child processes + 1 main
    start_barrier = multiprocessing.Barrier(4)

    processes = [
        multiprocessing.Process(
            name="Perception",
            target=perception_process_main,
            args=(perception_q, cfg, stop_event, start_barrier),
        ),
        multiprocessing.Process(
            name="Inference",
            target=inference_process_main,
            args=(perception_q, action_q, control_q_inference, infer_counter,
                  execution_ready, cfg, stop_event, start_barrier),
        ),
        multiprocessing.Process(
            name="Execution",
            target=execution_process_main,
            args=(action_q, control_q_execution, infer_counter,
                  execution_ready, cfg, stop_event, start_barrier),
        ),
        multiprocessing.Process(
            name="Keyboard",
            target=keyboard_process_main,
            args=(all_control_queues, stop_event),
        ),
    ]

    logging.info("Starting all processes for initialization...")
    for p in processes:
        p.start()

    try:
        logging.info("Waiting for all subprocesses to be ready...")
        start_barrier.wait()  # Phase 1: all initialized

        logging.info("All processes initialized and ready.")
        logging.info("Controls:")
        logging.info("  'r'     - Reset to home position")
        logging.info("  'SPACE' - Reset + pause/resume")
        logging.info("  'q'     - Quit")
        input("Press ENTER to start the real-time control loop... ")

        start_barrier.wait()  # Phase 2: release all

        while any(p.is_alive() for p in processes):
            time.sleep(0.5)

    except (KeyboardInterrupt, BrokenPipeError, BrokenBarrierError):
        logging.warning("Shutdown signal received or a process crashed!")
        stop_event.set()

    finally:
        logging.info("Waiting for all processes to terminate...")
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                logging.warning(f"Process {p.name} did not terminate. Forcing shutdown.")
                p.terminate()
        if server_proc is not None:
            logging.info("Stopping %s policy server...", cfg.policy_type)
            from fabricbench.bench.policy.servers import _kill_server
            _kill_server(server_proc)
        logging.info("All processes shut down.")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
