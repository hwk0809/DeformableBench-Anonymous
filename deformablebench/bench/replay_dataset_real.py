"""Replay a recorded LeRobot episode on the dual Piper rig.

The source dataset is the output of ``deformablebench.bench.collect_data``:
``action`` is 14-D, ordered as left(6 joints + gripper) + right(6 + gripper).
Joints are radians; grippers are normalized [0, 1] with 1=open.

By default this script is a dry run. Pass ``--execute`` on the real robot
machine to send commands.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

from deformablebench.bench.task.randomization import _ROBOT_INIT_PRESETS
from deformablebench.paths import REPO_ROOT

ARM_FACTOR = 57295.779513
JOINT_LIMITS = [
    (-2.687, 2.687),
    (0.0, 3.403),
    (-3.0541012, 0.0),
    (-1.5499, 1.5499),
    (-1.22, 1.22),
    (-1.7452, 1.7452),
]


class KeyState:
    def __init__(self) -> None:
        self.home = False
        self.stop = False
        self.pause = False
        self._lock = threading.Lock()

    def consume_home(self) -> bool:
        with self._lock:
            value = self.home
            self.home = False
            return value

    def should_stop(self) -> bool:
        with self._lock:
            return bool(self.stop)

    def is_paused(self) -> bool:
        with self._lock:
            return bool(self.pause)


def _start_key_listener(key_state: KeyState, stop_event: threading.Event):
    if not sys.stdin.isatty():
        print("keyboard controls disabled: stdin is not a TTY")
        return None

    def listen() -> None:
        old = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not stop_event.is_set():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue
                ch = sys.stdin.read(1).lower()
                with key_state._lock:
                    if ch in ("h", " "):
                        key_state.home = True
                        print("\n[KEY] return home", flush=True)
                    elif ch in ("q", "s"):
                        key_state.stop = True
                        print("\n[KEY] stop in place", flush=True)
                    elif ch == "p":
                        key_state.pause = not key_state.pause
                        print(f"\n[KEY] {'PAUSED' if key_state.pause else 'RESUMED'}", flush=True)
        finally:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
            except Exception:
                pass

    thread = threading.Thread(target=listen, name="ReplayKeyListener", daemon=True)
    thread.start()
    return thread


def _load_actions(root: str | Path, repo_id: str, episode: int) -> tuple[np.ndarray, int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode], download_videos=False)
    actions = []
    for item in ds.hf_dataset:
        ep = int(np.asarray(item["episode_index"]).reshape(()))
        if ep == int(episode):
            actions.append(np.asarray(item["action"], dtype=np.float32).reshape(14))
    if not actions:
        raise RuntimeError(f"no actions found for episode {episode} in {root}")
    return np.stack(actions), int(ds.fps)


def _task_start_pose(task_name: str | None) -> str | None:
    if not task_name:
        return None
    import yaml

    path = REPO_ROOT / "src" / "deformablebench" / "bench" / "config" / "task" / f"{task_name}.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text()) or {}
    return ((raw.get("randomization") or {}).get("robot_init") or {}).get("start_pose")


def _preset_home_14d(name: str) -> list[float]:
    if name not in _ROBOT_INIT_PRESETS:
        raise ValueError(f"unknown home preset {name!r}; available={sorted(_ROBOT_INIT_PRESETS)}")
    p = _ROBOT_INIT_PRESETS[name]
    g = float(p.get("gripper_open", 1.0))
    return [*(float(v) for v in p["home_left"]), g, *(float(v) for v in p["home_right"]), g]


def _read_current_position(piper_left, piper_right) -> list[float]:
    left_js = piper_left.GetArmJointMsgs().joint_state
    right_js = piper_right.GetArmJointMsgs().joint_state
    left_grip = piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
    right_grip = piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
    return [
        left_js.joint_1 / ARM_FACTOR,
        left_js.joint_2 / ARM_FACTOR,
        left_js.joint_3 / ARM_FACTOR,
        left_js.joint_4 / ARM_FACTOR,
        left_js.joint_5 / ARM_FACTOR,
        left_js.joint_6 / ARM_FACTOR,
        np.clip(left_grip / (70.0 * 1000), 0.0, 1.0),
        right_js.joint_1 / ARM_FACTOR,
        right_js.joint_2 / ARM_FACTOR,
        right_js.joint_3 / ARM_FACTOR,
        right_js.joint_4 / ARM_FACTOR,
        right_js.joint_5 / ARM_FACTOR,
        right_js.joint_6 / ARM_FACTOR,
        np.clip(right_grip / (70.0 * 1000), 0.0, 1.0),
    ]


def _joint_norm(state_14d: np.ndarray) -> float:
    joints = np.concatenate([state_14d[:6], state_14d[7:13]])
    return float(np.linalg.norm(joints))


def _wait_for_current_position(
    piper_left,
    piper_right,
    timeout: float,
    min_joint_norm: float,
    allow_zero_state: bool,
) -> np.ndarray:
    """Wait for Piper SDK state feedback instead of treating stale zeros as real."""
    deadline = time.time() + max(0.0, float(timeout))
    last = None
    while True:
        state = np.asarray(_read_current_position(piper_left, piper_right), dtype=np.float64)
        last = state
        if np.all(np.isfinite(state)):
            norm = _joint_norm(state)
            if allow_zero_state or norm >= float(min_joint_norm):
                return state
        if time.time() >= deadline:
            break
        time.sleep(0.05)

    detail = "none" if last is None else np.array2string(last, precision=4, suppress_small=True)
    raise RuntimeError(
        "Piper joint feedback did not look valid before motion. "
        f"last_state={detail}. Refusing to command from a likely stale zero state. "
        "Check arm feedback/enable state, or pass --allow-zero-start-state only if "
        "the robot is truly at the all-zero joint pose."
    )


def _sanitize_action(action: np.ndarray, grasp_close_val: float) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).reshape(14).copy()
    for arm_offset in (0, 7):
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            out[arm_offset + i] = float(np.clip(out[arm_offset + i], lo, hi))
    out[6] = max(float(np.clip(abs(out[6]), 0.0, 1.0)), grasp_close_val)
    out[13] = max(float(np.clip(abs(out[13]), 0.0, 1.0)), grasp_close_val)
    return out


def _send_position(piper_left, piper_right, action: np.ndarray) -> None:
    left_joints = [round(float(action[i]) * ARM_FACTOR) for i in range(6)]
    right_joints = [round(float(action[7 + i]) * ARM_FACTOR) for i in range(6)]
    left_gripper = round(float(action[6]) * 70 * 1000)
    right_gripper = round(float(action[13]) * 70 * 1000)
    # Piper joint position mode needs to be refreshed while streaming.  Sending
    # it only once can leave gripper commands working while JointCtrl is ignored.
    piper_left.MotionCtrl_2(0x01, 0x01, 60, 0xAD)
    piper_left.JointCtrl(*left_joints)
    piper_left.GripperCtrl(left_gripper, 1000, 0x01, 0)
    piper_right.MotionCtrl_2(0x01, 0x01, 60, 0xAD)
    piper_right.JointCtrl(*right_joints)
    piper_right.GripperCtrl(right_gripper, 1000, 0x01, 0)


def _enable_arm(piper, name: str) -> None:
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            if hasattr(piper, "EnablePiper") and bool(piper.EnablePiper()):
                piper.GripperCtrl(70 * 1000, 1000, 0x01, 0)
                print(f"Arm {name} enabled.")
                return
            piper.EnableArm(7)
            piper.GripperCtrl(70 * 1000, 1000, 0x01, 0)
            print(f"Arm {name} enabled.")
            return
        except Exception as exc:
            print(f"Enable arm {name} retrying: {exc}")
            time.sleep(0.1)
    raise RuntimeError(f"Arm {name} enable timeout")


def _slow_move(piper_left, piper_right, target: np.ndarray, duration: float, control_freq: int,
               grasp_close_val: float, key_state: KeyState | None = None,
               state_wait_time: float = 3.0, min_valid_joint_norm: float = 0.2,
               allow_zero_start_state: bool = False) -> str:
    start = _wait_for_current_position(
        piper_left,
        piper_right,
        timeout=state_wait_time,
        min_joint_norm=min_valid_joint_norm,
        allow_zero_state=allow_zero_start_state,
    )
    target = _sanitize_action(target, grasp_close_val)
    max_delta = float(np.max(np.abs(target - start)))
    print(f"current={np.array2string(start, precision=4, suppress_small=True)}")
    print(f"target ={np.array2string(target, precision=4, suppress_small=True)}")
    print(f"startup max joint/gripper delta={max_delta:.4f} over {duration:.1f}s")
    steps = max(1, int(float(duration) * int(control_freq)))
    period = 1.0 / int(control_freq)
    for i in range(steps):
        if key_state is not None and key_state.should_stop():
            return "stop"
        u = (i + 1) / steps
        u = 3 * u * u - 2 * u * u * u
        _send_position(piper_left, piper_right, start * (1.0 - u) + target * u)
        time.sleep(period)
    return "complete"


def _summarize(actions: np.ndarray, fps: int, grasp_close_val: float) -> None:
    clean = np.stack([_sanitize_action(a, grasp_close_val) for a in actions])
    print(f"frames={len(actions)} fps={fps} duration={len(actions) / max(fps, 1):.2f}s")
    print(f"first={np.array2string(clean[0], precision=4, suppress_small=True)}")
    print(f"last ={np.array2string(clean[-1], precision=4, suppress_small=True)}")
    for arm_name, offset in (("left", 0), ("right", 7)):
        mins = clean[:, offset:offset + 6].min(axis=0)
        maxs = clean[:, offset:offset + 6].max(axis=0)
        print(f"{arm_name} joint rad min={np.round(mins, 4).tolist()} max={np.round(maxs, 4).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True,
                    help="dataset directory, e.g. data/lerobot_dataset/fold_tshirt_side_fixed")
    ap.add_argument("--repo-id", default=None, help="default: basename of --dataset-root")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--task", default="fold_tshirt_side", help="used to derive home preset")
    ap.add_argument("--home-preset", default=None, help="overrides task robot_init.start_pose")
    ap.add_argument("--export", default=None, help="optional .npz export path for the action trajectory")
    ap.add_argument("--execute", action="store_true", help="actually send commands to the real robot")
    ap.add_argument("--yes", action="store_true", help="do not wait for ENTER before replay")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = dataset timing, 0.5 = half speed")
    ap.add_argument("--control-freq", type=int, default=200)
    ap.add_argument("--startup-move-duration", type=float, default=5.0)
    ap.add_argument("--state-wait-time", type=float, default=3.0,
                    help="seconds to wait for non-stale Piper joint feedback before moving")
    ap.add_argument("--min-valid-joint-norm", type=float, default=0.2,
                    help="reject startup feedback whose 12-joint norm is below this value")
    ap.add_argument("--allow-zero-start-state", action="store_true",
                    help="allow an all-zero startup joint state; normally unsafe for replay")
    ap.add_argument("--grasp-close-val", type=float, default=0.015)
    ap.add_argument("--can-left", default="can_left")
    ap.add_argument("--can-right", default="can_right")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.dataset_root))
    repo_id = args.repo_id or root.name
    actions, fps = _load_actions(root, repo_id, args.episode)
    _summarize(actions, fps, args.grasp_close_val)

    if args.export:
        out = Path(os.path.expanduser(args.export))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            actions=actions.astype(np.float32),
            fps=np.array(fps, dtype=np.int32),
            episode=np.array(args.episode, dtype=np.int32),
            motor_names=np.array([
                "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
                "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
                "left_gripper.pos", "right_joint_1.pos", "right_joint_2.pos",
                "right_joint_3.pos", "right_joint_4.pos", "right_joint_5.pos",
                "right_joint_6.pos", "right_gripper.pos",
            ]),
        )
        meta = {"repo_id": repo_id, "episode": args.episode, "fps": fps, "frames": int(len(actions))}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"exported {out}")

    if not args.execute:
        print("dry-run only; pass --execute on the real robot machine to replay")
        return
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    from piper_sdk import C_PiperInterface_V2

    piper_left = C_PiperInterface_V2(args.can_left)
    piper_right = C_PiperInterface_V2(args.can_right)
    piper_left.ConnectPort()
    piper_right.ConnectPort()
    _enable_arm(piper_left, "left")
    _enable_arm(piper_right, "right")
    piper_left.MotionCtrl_2(0x01, 0x01, 60, 0xAD)
    piper_right.MotionCtrl_2(0x01, 0x01, 60, 0xAD)

    home_preset = args.home_preset or _task_start_pose(args.task)
    home = np.asarray(_preset_home_14d(home_preset), dtype=np.float64) if home_preset else actions[0]
    print(f"moving to home ({home_preset or 'first_action'}) over {args.startup_move_duration:.1f}s")
    _slow_move(piper_left, piper_right, home, args.startup_move_duration,
               args.control_freq, args.grasp_close_val,
               state_wait_time=args.state_wait_time,
               min_valid_joint_norm=args.min_valid_joint_norm,
               allow_zero_start_state=args.allow_zero_start_state)
    if not args.yes:
        input("Press ENTER to start replay. Keys during replay: SPACE/h home, p pause, q/s stop. ")

    key_state = KeyState()
    stop_event = threading.Event()
    _start_key_listener(key_state, stop_event)

    frame_period = 1.0 / (float(fps) * float(args.speed))
    tick_period = 1.0 / int(args.control_freq)
    t0 = time.perf_counter()
    paused_total = 0.0
    pause_start = None
    completed = False
    try:
        for i in range(len(actions) - 1):
            start = _sanitize_action(actions[i], args.grasp_close_val)
            end = _sanitize_action(actions[i + 1], args.grasp_close_val)
            steps = max(1, int(round(frame_period * int(args.control_freq))))
            for j in range(steps):
                if key_state.should_stop():
                    print("replay stopped in place")
                    return
                if key_state.consume_home():
                    print("returning home and ending replay")
                    _slow_move(
                        piper_left,
                        piper_right,
                        home,
                        args.startup_move_duration,
                        args.control_freq,
                        args.grasp_close_val,
                        key_state=key_state,
                        state_wait_time=args.state_wait_time,
                        min_valid_joint_norm=args.min_valid_joint_norm,
                        allow_zero_start_state=args.allow_zero_start_state,
                    )
                    return
                while key_state.is_paused():
                    if pause_start is None:
                        pause_start = time.perf_counter()
                    if key_state.should_stop():
                        print("replay stopped in place")
                        return
                    if key_state.consume_home():
                        print("returning home and ending replay")
                        _slow_move(
                            piper_left,
                            piper_right,
                            home,
                            args.startup_move_duration,
                            args.control_freq,
                            args.grasp_close_val,
                            key_state=key_state,
                            state_wait_time=args.state_wait_time,
                            min_valid_joint_norm=args.min_valid_joint_norm,
                            allow_zero_start_state=args.allow_zero_start_state,
                        )
                        return
                    time.sleep(0.05)
                if pause_start is not None:
                    paused_total += time.perf_counter() - pause_start
                    pause_start = None
                u = j / steps
                _send_position(piper_left, piper_right, start * (1.0 - u) + end * u)
                target_t = t0 + paused_total + (i * steps + j + 1) * tick_period
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        _send_position(piper_left, piper_right, _sanitize_action(actions[-1], args.grasp_close_val))
        completed = True
    except KeyboardInterrupt:
        print("replay interrupted")
    finally:
        stop_event.set()
    print("replay done" if completed else "replay ended")


if __name__ == "__main__":
    main()
