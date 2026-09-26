"""Streaming LeRobot writer for passive dual-PiPER demonstrations.

The firmware master/slave linkage moves the follower arms.  We record follower
feedback only and pair frames with a one-step look-ahead:
``action[t] = follower_state[t + 1]``.  This matches the simulation collector's
forward-policy convention while never reading or commanding the master arms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fabricbench.bench.data.inmemory_video import InMemoryVideoDataset
from fabricbench.bench.data.schema import MOTOR_NAMES

ROBOT_TYPE = "dual_piper"
_LEROBOT_AUTO_FEATURES = {
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
}


def follower_state_14d(observation: dict[str, Any]) -> np.ndarray:
    """Extract the canonical radian/[0,1] follower state from a robot observation."""
    missing = [name for name in MOTOR_NAMES if name not in observation]
    if missing:
        raise KeyError(f"dual_piper observation is missing motor fields: {missing}")

    state = np.asarray([observation[name] for name in MOTOR_NAMES], dtype=np.float32)
    if state.shape != (14,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid dual_piper follower state: {state!r}")
    for index in (6, 13):
        if not 0.0 <= float(state[index]) <= 1.0:
            raise ValueError(
                f"gripper state {MOTOR_NAMES[index]}={state[index]} is outside normalized [0,1]"
            )
    return state


def real_dataset_features(robot) -> tuple[dict[str, dict], list[str]]:
    """Build the fixed 14-D schema plus configured RGB video features."""
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": (14,), "names": MOTOR_NAMES},
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": MOTOR_NAMES,
        },
    }
    camera_names = list(getattr(robot, "cameras", {}).keys())
    observation_features = robot.observation_features
    for camera_name in camera_names:
        shape = observation_features.get(camera_name)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"camera {camera_name!r} has unsupported feature shape {shape!r}")
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": tuple(int(value) for value in shape),
            "names": ["height", "width", "channels"],
        }
    return features, camera_names


def _validate_resume_dataset(dataset, expected_features: dict, fps: int) -> None:
    if dataset.meta.robot_type != ROBOT_TYPE:
        raise ValueError(
            f"cannot resume robot_type={dataset.meta.robot_type!r}; expected {ROBOT_TYPE!r}"
        )
    if int(dataset.fps) != int(fps):
        raise ValueError(f"cannot resume fps={dataset.fps}; requested fps={fps}")
    stored_features = set(dataset.meta.features) - _LEROBOT_AUTO_FEATURES
    if stored_features != set(expected_features):
        raise ValueError(
            "cannot resume dataset with different features: "
            f"stored={sorted(stored_features)}, requested={sorted(expected_features)}"
        )
    for key, expected in expected_features.items():
        stored = dataset.meta.features[key]
        if stored["dtype"] != expected["dtype"] or tuple(stored["shape"]) != tuple(
            expected["shape"]
        ):
            raise ValueError(
                f"cannot resume feature {key!r}: stored={stored}, requested={expected}"
            )


def open_real_dataset(
    *,
    robot,
    root: str | Path,
    repo_id: str,
    fps: int,
    resume: bool,
):
    """Open a new or resumable dataset without ever deleting existing data."""
    root = Path(root)
    features, camera_names = real_dataset_features(robot)
    has_dataset = (root / "meta" / "info.json").is_file()

    if resume:
        if not has_dataset:
            raise FileNotFoundError(f"--resume requested but no LeRobot dataset exists at {root}")
        dataset = InMemoryVideoDataset(repo_id=repo_id, root=str(root))
        _validate_resume_dataset(dataset, features, fps)
    else:
        if root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing path {root}; choose a new --repo-id or use --resume"
            )
        dataset = InMemoryVideoDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            root=str(root),
            robot_type=ROBOT_TYPE,
            features=features,
            use_videos=bool(camera_names),
            image_writer_threads=0,
        )
    return dataset, camera_names


class PassiveFollowerEpisode:
    """Stream one episode while retaining only the latest unpaired RGB frame."""

    def __init__(self, dataset, camera_names: list[str], task: str):
        self.dataset = dataset
        self.camera_names = list(camera_names)
        self.task = task
        self._pending: dict[str, Any] | None = None
        self._sealed = False
        self.num_frames = 0

    def _frame_from_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = follower_state_14d(observation)
        images: dict[str, np.ndarray] = {}
        for name in self.camera_names:
            if name not in observation:
                raise KeyError(f"dual_piper observation is missing camera {name!r}")
            image = np.asarray(observation[name])
            expected_shape = tuple(
                self.dataset.meta.features[f"observation.images.{name}"]["shape"]
            )
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"camera {name!r} returned shape/dtype {image.shape}/{image.dtype}; "
                    f"expected {expected_shape}/uint8"
                )
            images[name] = image.copy()
        return {"observation.state": state, "images": images}

    def _write_pending(self, action: np.ndarray) -> None:
        if self._pending is None:
            return
        frame: dict[str, Any] = {
            "task": self.task,
            "observation.state": self._pending["observation.state"],
            "action": np.asarray(action, dtype=np.float32),
        }
        for name, image in self._pending["images"].items():
            frame[f"observation.images.{name}"] = image
        self.dataset.add_frame(frame)
        self.num_frames += 1

    def add_observation(self, observation: dict[str, Any]) -> None:
        if self._sealed:
            raise RuntimeError("cannot add observations after sealing an episode")
        current = self._frame_from_observation(observation)
        # The current follower state is the action that moves the previous
        # observation forward, exactly matching bench.data.collect.EpisodeCollector.
        self._write_pending(current["observation.state"])
        self._pending = current

    def seal(self) -> int:
        """Pad the final action with its state; data remains discardable until commit()."""
        if not self._sealed and self._pending is not None:
            self._write_pending(self._pending["observation.state"])
            self._pending = None
        self._sealed = True
        return self.num_frames

    def commit(self) -> int:
        self.seal()
        if self.num_frames == 0:
            raise RuntimeError("refusing to save an empty real-robot episode")
        self.dataset.save_episode()
        return self.num_frames

    def discard(self) -> None:
        self._pending = None
        self._sealed = True
        self.dataset.clear_episode_buffer()
