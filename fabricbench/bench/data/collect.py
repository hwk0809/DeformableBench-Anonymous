"""Episode data collection in LeRobot v3.0 layout.

Records the LeRobot v3.0 feature set from the open newton env. The orchestration
(subprocess-per-episode, resume,
success filtering) lives in ``collect_data.py``; this module holds the per-episode
recording + dataset writing.

Schema (LeRobot v3.0, robot_type ``sim_dual_piper_deformable``):
  action / observation.state : float32 [14] = left(6 joints + gripper) + right(6+grip)
  observation.images.<cam>   : video [H, W, 3] (mp4), fps = control_fps

Action convention (one-frame buffer): ``action[t] = cmd[t+1]`` — the command that
drives ``obs[t] -> obs[t+1]`` (forward-policy pairing). The last frame pads with
``action = state``.

``save_render_state`` (optional, default off) additionally dumps a per-episode npz
(cloth vertices, robot link poses, gripper, raw state/action, times, stages) for
offline re-rendering / analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fabricbench.bench.data.schema import MOTOR_NAMES
from fabricbench.bench.render.scene import gripper_values, robot_link_poses

ROBOT_TYPE = "sim_dual_piper_deformable"


def state_14d(env) -> np.ndarray:
    """Proprioceptive state: left(6 joints, grip) + right(6 joints, grip)."""
    return np.array([
        *np.asarray(env.left_qpos, dtype=float)[:6], float(env.left_gripper_val),
        *np.asarray(env.right_qpos, dtype=float)[:6], float(env.right_gripper_val),
    ], dtype=np.float32)


def action_14d(action_dict: dict, env) -> np.ndarray:
    """Commanded action (falls back to current state when no command)."""
    if not action_dict:
        return state_14d(env)
    q_l = np.asarray(action_dict.get("q_l", env.left_qpos), dtype=float)[:6]
    q_r = np.asarray(action_dict.get("q_r", env.right_qpos), dtype=float)[:6]
    g_l = float(action_dict.get("g_l", env.left_gripper_val))
    g_r = float(action_dict.get("g_r", env.right_gripper_val))
    return np.array([*q_l, g_l, *q_r, g_r], dtype=np.float32)


def _fresh_rs() -> dict:
    return {k: [] for k in ("state", "action_cmd", "cloth", "link", "gripper", "time", "stage")}


# ----------------------------------------------------------------------------
class EpisodeCollector:
    """SimExecutor record hook: ``record_frame(env, action_dict)`` per physics step.

    Downsamples physics -> control_fps, grabs 14-D state/action + per-camera RGB,
    applies the action=cmd[t+1] one-frame buffer, and (optionally) buffers
    render-state. ``start()``/``stop()`` gate recording so settle/randomization
    frames are skipped."""

    def __init__(self, bridge, physics_fps: int = 50, control_fps: int = 25,
                 save_render_state: bool = False):
        self.bridge = bridge
        self.control_every = max(1, int(round(physics_fps / control_fps)))
        self.save_rs = bool(save_render_state)
        self.active = False
        self._step = 0
        self._pending: dict | None = None
        self.frames: list[dict] = []
        self._rs = _fresh_rs()
        self.stage = "default"

    def start(self) -> None:
        self.active = True
        self._step = 0
        self._pending = None
        self.frames = []
        self._rs = _fresh_rs()

    def stop(self) -> None:
        self.active = False

    def record_frame(self, env, action_dict) -> None:
        if not self.active:
            return
        if self._step % self.control_every == 0:
            state = state_14d(env)
            cmd = action_14d(action_dict, env)
            images = self.bridge.render(env) if self.bridge is not None else {}
            # action[t] = cmd[t+1]: fill the held-back frame with THIS frame's cmd.
            if self._pending is not None:
                self._pending["action"] = cmd
                self.frames.append(self._pending)
            self._pending = {"observation.state": state, "images": images, "action": None}
            if self.save_rs:
                self._rs["state"].append(state)
                self._rs["action_cmd"].append(cmd)
                self._rs["cloth"].append(env.get_sim_vertices().astype(np.float32))
                self._rs["link"].append(robot_link_poses(env))
                self._rs["gripper"].append(gripper_values(env))
                self._rs["time"].append(float(env.get_current_sim_time()))
                self._rs["stage"].append(self.stage)
        self._step += 1

    def take_episode(self):
        """Flush the last held frame (action=state pad); return (frames, render_state|None)."""
        if self._pending is not None:
            self._pending["action"] = self._pending["observation.state"]
            self.frames.append(self._pending)
            self._pending = None
        frames = self.frames
        rs = None
        if self.save_rs and self._rs["state"]:
            rs = {
                "cloth_positions": np.stack(self._rs["cloth"]),
                "link_poses_world": np.stack(self._rs["link"]),
                "gripper_values": np.stack(self._rs["gripper"]),
                "states": np.stack(self._rs["state"]),
                "action_cmds": np.stack(self._rs["action_cmd"]),
                "times": np.array(self._rs["time"], dtype=np.float64),
                "stage_names": np.array(self._rs["stage"]),
            }
        self.frames = []
        self._rs = _fresh_rs()
        return frames, rs


# ----------------------------------------------------------------------------
class LeRobotWriter:
    """Write episodes in LeRobot v3.0 layout (real lerobot, with interim fallback).

    One writer per worker process: create (new/overwrite) or resume (append to an
    existing valid dataset), add the episode(s), save + finalize, exit."""

    def __init__(self, root: str | Path, repo_id: str, fps: int,
                 camera_names: list[str], image_hw: tuple[int, int],
                 mode: str = "create"):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.camera_names = list(camera_names)
        self.h, self.w = int(image_hw[0]), int(image_hw[1])
        self.features = self._build_features()
        self.backend, self._ds = self._init_backend(mode)
        self._episodes = 0
        self._frames = 0

    def _build_features(self) -> dict:
        feats = {
            "action": {"dtype": "float32", "shape": (14,), "names": MOTOR_NAMES},
            "observation.state": {"dtype": "float32", "shape": (14,), "names": MOTOR_NAMES},
        }
        for cam in self.camera_names:
            feats[f"observation.images.{cam}"] = {
                "dtype": "video", "shape": (self.h, self.w, 3),
                "names": ["height", "width", "channels"],
            }
        return feats

    def _has_valid_data(self) -> bool:
        return (self.root / "meta" / "info.json").exists() and (self.root / "data").exists()

    def _init_backend(self, mode: str):
        try:
            from fabricbench.bench.data.inmemory_video import InMemoryVideoDataset
        except Exception:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "meta").mkdir(exist_ok=True)
            return "interim", None
        if mode == "resume" and self._has_valid_data():
            ds = InMemoryVideoDataset(repo_id=self.repo_id, root=str(self.root))
        else:
            if self.root.exists():
                import shutil
                shutil.rmtree(self.root)
            # InMemoryVideoDataset stream-encodes video features straight to mp4,
            # so no per-frame PNG is ever written (see inmemory_video.py). All our
            # camera features are dtype "video", hence image_writer_threads=0 --
            # the async PNG writer would have nothing to do.
            ds = InMemoryVideoDataset.create(
                repo_id=self.repo_id, fps=self.fps, features=self.features,
                root=str(self.root), robot_type=ROBOT_TYPE, use_videos=True,
                image_writer_threads=0,
            )
        return "lerobot", ds

    def add_episode(self, frames: list[dict], task: str, render_state: dict | None = None,
                    episode_idx: int | None = None) -> None:
        if not frames:
            return
        if self.backend == "lerobot":
            for f in frames:
                frame = {"task": task, "action": f["action"],
                         "observation.state": f["observation.state"]}
                for cam, img in f["images"].items():
                    frame[f"observation.images.{cam}"] = img
                self._ds.add_frame(frame)
            self._ds.save_episode()
        else:
            self._write_interim_episode(frames, task, episode_idx or self._episodes)
        if render_state is not None:
            idx = episode_idx if episode_idx is not None else self._episodes
            rs_dir = self.root / "render_state"
            rs_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(rs_dir / f"episode_{idx:06d}.npz", **render_state)
        self._episodes += 1
        self._frames += len(frames)

    def _write_interim_episode(self, frames, task, idx) -> None:
        from PIL import Image
        ep = self.root / "episodes" / f"episode_{idx:06d}"
        ep.mkdir(parents=True, exist_ok=True)
        np.savez(ep / "lowdim.npz",
                 **{"observation.state": np.stack([f["observation.state"] for f in frames]),
                    "action": np.stack([f["action"] for f in frames])})
        for cam in self.camera_names:
            (ep / cam).mkdir(exist_ok=True)
            for i, f in enumerate(frames):
                if f["images"].get(cam) is not None:
                    Image.fromarray(f["images"][cam]).save(ep / cam / f"{i:05d}.png")
        (ep / "meta.json").write_text(json.dumps({"task": task, "num_frames": len(frames)}))

    def finalize(self) -> None:
        if self.backend == "lerobot":
            if hasattr(self._ds, "finalize"):
                self._ds.finalize()
            return
        info = {
            "codebase_version": "v3.0", "robot_type": ROBOT_TYPE,
            "total_episodes": self._episodes, "total_frames": self._frames,
            "total_tasks": 1, "fps": self.fps, "features": self.features,
            "_backend": "interim (lerobot not installed)",
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2, default=list))
