"""In-memory streaming video encoder for LeRobotDataset.

Why this exists:
  LeRobot's default video path is
    numpy frame -> AsyncImageWriter writes PNG to disk -> save_episode reads the
    PNGs back, decodes with PIL and re-encodes to MP4 (ProcessPoolExecutor).
  That is two useless round trips (PNG write/read + decode/encode). For our
  1280x720 multi-camera episodes it costs 30-120s per episode and needs write +
  *delete* permission for thousands of files under the dataset root -- a problem
  on shared storage that disallows unlink.

What this module does:
  Subclass LeRobotDataset and stream every ``dtype == "video"`` feature straight
  into a pyav encoder from ``add_frame``; ``save_episode`` closes the stream and
  hands the finished MP4 to the parent's ``_save_episode_video`` for chunk
  management.

  - lerobot itself is untouched (pure subclass)
  - O(1) memory (streaming encode, the whole clip is never buffered)
  - a small sample buffer (~30 frames/camera) is kept for compute_episode_stats,
    which normally reads the PNGs back off disk

Note the per-episode temporary MP4 is *not* removable: lerobot v3 packs several
episodes into one chunk file, so an episode must exist as a self-contained MP4
before we know whether it is appended to the current chunk or starts a new one.
It is one already-compressed file per episode, versus one uncompressed PNG per
frame per camera.

Compatibility:
  - only ``dtype == "video"`` features take the streaming path
  - ``dtype == "image"`` features keep the parent's PNG-into-parquet path
  - state / action / other numerical features are unaffected
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import av
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer

log = logging.getLogger(__name__)

# Encoder settings, matching lerobot/datasets/video_utils.py (H.264 veryfast).
_H264_OPTIONS = {
    "preset": "veryfast",
    "crf": "30",
    "g": "2",
}
_PIX_FMT = "yuv420p"

# Downsampled frames kept per camera for stats. lerobot's compute_episode_stats
# samples the episode uniformly (sample_indices -> linspace over all frames), so
# the buffer must span the whole episode, not its tail. Episodes at or under the
# cap keep every frame and reproduce lerobot's stats exactly; longer ones decimate
# by a growing stride, which stays uniform but no longer bit-exact.
#
# Sized against our longest task: fold_tshirt_side episodes run 905-1383 frames
# (median 1019), so 2048 keeps the exact path for all of them. Frames are stored
# already downsampled (~150px, as lerobot does before computing stats), which at
# 1280x720 is 160x90x3 = 42 KB/frame -- so the cap costs 84 MB per camera, ~253 MB
# for a 3-camera rig. That is peak RAM within a single episode, released on
# save_episode; it does not accumulate across episodes.
_STATS_SAMPLE_CAP = 2048


class InMemoryVideoDataset(LeRobotDataset):
    """LeRobotDataset whose video features stream-encode, skipping PNG staging.

    Drop-in for LeRobotDataset -- construct via ``.create()`` or ``__init__``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_inmem_state()

    def _init_inmem_state(self) -> None:
        """Init streaming state. Idempotent (both create() and __init__ call it)."""
        # video_key -> (av.Container, av.VideoStream, temp_path)
        self._av_streams: dict[str, tuple[Any, Any, Path]] = {}
        # video_key -> list[np.ndarray], the stats sample (channel-first, downsampled)
        self._stats_samples: dict[str, list[np.ndarray]] = {}
        # video_key -> current decimation stride for that sample
        self._stats_stride: dict[str, int] = {}
        # video_key -> frames encoded so far
        self._frame_count: dict[str, int] = {}
        # all streaming tempdirs live here so cleanup has one root
        self._scratch_root: Path = Path(self.root) / ".inmem_scratch"
        # tempdirs owed cleanup for the current episode (rmtree is ignore_errors,
        # so it stays safe even after _save_episode_video already removed one)
        self._pending_temp_dirs: list[Path] = []
        # drop anything a previous crash left behind
        try:
            if self._scratch_root.exists():
                shutil.rmtree(self._scratch_root, ignore_errors=True)
        except Exception:
            pass

    @classmethod
    def create(cls, *args, **kwargs):
        """create() goes through __new__, not __init__, so init our state here too."""
        obj = super().create(*args, **kwargs)
        obj._init_inmem_state()
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_video_feature(self, key: str) -> bool:
        return self.features.get(key, {}).get("dtype") == "video"

    def _open_stream(self, key: str, img: np.ndarray) -> None:
        """Open a pyav container for this video_key on its first frame.

        The dir name carries the episode index so a leftover after a crash can be
        traced back to the episode that produced it.
        """
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        try:
            ep_idx = int(self.episode_buffer.get("episode_index", self.meta.total_episodes))
        except Exception:
            ep_idx = self.meta.total_episodes
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"ep{ep_idx:06d}_{key}_", dir=str(self._scratch_root)))
        self._pending_temp_dirs.append(tmp_dir)
        temp_path = tmp_dir / f"{key}.mp4"

        container = av.open(str(temp_path), mode="w")
        stream = container.add_stream("h264", rate=self.fps, options=_H264_OPTIONS)
        stream.pix_fmt = _PIX_FMT
        stream.width = int(img.shape[1])
        stream.height = int(img.shape[0])

        self._av_streams[key] = (container, stream, temp_path)
        self._frame_count[key] = 0
        self._stats_samples[key] = []
        self._stats_stride[key] = 1

    def _encode_one(self, key: str, img: np.ndarray) -> None:
        """Encode one (H, W, 3) uint8 RGB frame into the av stream."""
        if key not in self._av_streams:
            self._open_stream(key, img)

        container, stream, _ = self._av_streams[key]
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        vframe = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(vframe):
            container.mux(packet)

        self._collect_stats_sample(key, img)
        self._frame_count[key] += 1

    def _collect_stats_sample(self, key: str, img: np.ndarray) -> None:
        """Keep a uniform, whole-episode sample of frames for stats.

        Deterministic decimation: take every ``stride``-th frame; when the buffer
        fills, drop every other entry and double the stride. The kept indices stay
        an evenly spaced grid starting at frame 0, so the sample always spans the
        episode rather than drifting towards its tail.

        Frames are stored channel-first and downsampled with lerobot's own helper,
        matching what compute_episode_stats would have done after reading the PNGs.
        """
        from lerobot.datasets.compute_stats import auto_downsample_height_width

        idx = self._frame_count[key]
        stride = self._stats_stride[key]
        if idx % stride:
            return

        samples = self._stats_samples[key]
        samples.append(auto_downsample_height_width(img.transpose(2, 0, 1).copy()))
        if len(samples) > _STATS_SAMPLE_CAP:
            self._stats_samples[key] = samples[::2]
            self._stats_stride[key] = stride * 2

    def _close_streams(self) -> dict[str, Path]:
        """Flush + close every stream, returning {video_key: temp_mp4_path}."""
        temp_paths: dict[str, Path] = {}
        for key, (container, stream, temp_path) in self._av_streams.items():
            try:
                for packet in stream.encode():  # flush
                    container.mux(packet)
            finally:
                container.close()
            temp_paths[key] = temp_path
        self._av_streams.clear()
        return temp_paths

    def _abort_streams(self) -> None:
        """Failure path: close containers only. Tempdir removal belongs to
        clear_episode_buffer (via _pending_temp_dirs) so cleanup has one owner."""
        for _, (container, _, _) in list(self._av_streams.items()):
            try:
                container.close()
            except Exception:
                pass
        self._av_streams.clear()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """Single cleanup entry point for the streaming state.

        Runs on both the success path (parent's save_episode tail) and the
        failure path (collect_data calls it), so it must be idempotent.
        """
        self._abort_streams()
        for td in self._pending_temp_dirs:
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass
        self._pending_temp_dirs.clear()
        try:
            if self._scratch_root.exists():
                shutil.rmtree(self._scratch_root, ignore_errors=True)
        except Exception:
            pass
        # Prune the empty images/{video_key}/episode-X dirs the parent still makes
        # (add_frame mkdirs them before calling _save_image, which we no-op).
        try:
            images_root = Path(self.root) / "images"
            if images_root.exists():
                for video_key in self.meta.video_keys:
                    key_dir = images_root / video_key
                    if not key_dir.exists():
                        continue
                    for ep_dir in list(key_dir.iterdir()):
                        if ep_dir.is_dir() and not any(ep_dir.iterdir()):
                            ep_dir.rmdir()
                    if not any(key_dir.iterdir()):
                        key_dir.rmdir()
                if not any(images_root.iterdir()):
                    images_root.rmdir()
        except Exception:
            pass
        self._stats_samples.clear()
        self._stats_stride.clear()
        self._frame_count.clear()
        super().clear_episode_buffer(delete_images=delete_images)

    def _save_image(self, image, fpath, compress_level: int = 1) -> None:
        """Skip the PNG write for video features -- already stream-encoded.

        fpath is {root}/images/{image_key}/episode-XXXXXX/frame-XXXXXX.png, so
        parent.parent.name recovers the feature key.
        """
        try:
            if self.features.get(Path(fpath).parent.parent.name, {}).get("dtype") == "video":
                return
        except Exception:
            pass
        super()._save_image(image, fpath, compress_level)

    def add_frame(self, frame: dict) -> None:
        """Stream-encode video features; everything else passes through.

        The parent still appends a PNG path into episode_buffer[video_key] (the
        file never materialises, and save_episode below ignores those paths).

        Order matters: let the parent validate and buffer first, so a rejected
        frame does not leave an open av stream or a stray tempdir behind.
        """
        super().add_frame(frame)
        for key in list(frame.keys()):
            if self._is_video_feature(key):
                self._encode_one(key, frame[key])

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """Full override: use our pre-encoded MP4s instead of the PNG->MP4 pass.

          1. close av streams -> temp MP4 paths
          2. stage the buffer (same as the parent: stack non-video, tasks, indices)
          3. stats -- video keys from the in-memory sample, the rest as usual
          4. _save_episode_data: parent as-is (video keys are not in hf_features)
          5. _save_episode_video(temp_path=...): parent as-is (move/concat MP4)
          6. meta.save_episode, then clear_episode_buffer
        """
        from lerobot.datasets.compute_stats import (
            compute_episode_stats,
            get_feature_stats,
            sample_indices,
        )

        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        try:
            validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

            # ---- Stage buffer (mirrors the parent) ----
            episode_length = episode_buffer.pop("size")
            tasks = episode_buffer.pop("task")
            episode_tasks = list(set(tasks))
            episode_index = episode_buffer["episode_index"]

            episode_buffer["index"] = np.arange(
                self.meta.total_frames, self.meta.total_frames + episode_length
            )
            episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
            self.meta.save_episode_tasks(episode_tasks)
            episode_buffer["task_index"] = np.array([self.meta.get_task_index(t) for t in tasks])

            for key, ft in self.features.items():
                if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                    continue
                episode_buffer[key] = np.stack(episode_buffer[key])

            # ---- Stats ----
            # compute_episode_stats would call sample_images(paths) for video keys,
            # i.e. read the PNGs we never wrote. Run it over the non-video features
            # only, then fold in the video stats from our in-memory sample.
            non_video_features = {
                k: v for k, v in self.features.items() if v.get("dtype") != "video"
            }
            non_video_buffer = {
                k: v for k, v in episode_buffer.items() if k in non_video_features
            }
            ep_stats = compute_episode_stats(non_video_buffer, non_video_features)

            for key in self.meta.video_keys:
                samples = self._stats_samples.get(key)
                if not samples:
                    log.warning("[InMemoryVideo] no stats sample for %s, skipping", key)
                    continue
                # Same reduction lerobot applies to sample_images() output: sample
                # uniformly, reduce over (batch, H, W), normalise to [0, 1].
                arr = np.stack(samples)[sample_indices(len(samples))]
                stats = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
                ep_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in stats.items()
                }

            # ---- Parquet (parent; video keys are absent from hf_features) ----
            ep_metadata = self._save_episode_data(episode_buffer)

            # ---- Video (hand the parent our pre-encoded MP4) ----
            temp_paths = self._close_streams()
            for video_key in self.meta.video_keys:
                if video_key not in temp_paths:
                    log.warning("[InMemoryVideo] no stream for video_key=%s, skipping", video_key)
                    continue
                ep_metadata.update(
                    self._save_episode_video(video_key, episode_index, temp_path=temp_paths[video_key])
                )

            self.meta.save_episode(
                episode_index, episode_length, episode_tasks, ep_stats, ep_metadata
            )

            if not episode_data:
                self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

        except Exception:
            try:
                self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)
            except Exception:
                pass
            raise
