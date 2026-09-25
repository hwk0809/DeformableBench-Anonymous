"""LingBot-VA TCP inference client.

The wire protocol matches lingbot-va/evaluation/tcp_policy_server.py:
4-byte big-endian payload length followed by pickle(<dict>).
"""

from __future__ import annotations

import logging
import os
import pickle
import socket
import struct
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CAMS = ["static_cam", "left_hand_cam", "right_hand_cam"]


def _u8(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


class _TCPClient:
    """Length-prefixed pickle client with one reconnect retry."""

    def __init__(self, host, port, connect_timeout=60.0, request_timeout=600.0):
        self.host, self.port = host, int(port)
        self.connect_timeout, self.request_timeout = connect_timeout, request_timeout
        self._sock = None
        self._connect()

    def _connect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        s.settimeout(self.request_timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s

    @staticmethod
    def _recv_exactly(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server closed during recv")
            buf.extend(chunk)
        return bytes(buf)

    def round_trip(self, msg: dict) -> dict:
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        last = None
        for _ in range(2):
            try:
                self._sock.sendall(struct.pack(">I", len(payload)) + payload)
                (length,) = struct.unpack(">I", self._recv_exactly(self._sock, 4))
                return pickle.loads(self._recv_exactly(self._sock, length))
            except (ConnectionError, OSError) as e:
                last = e
                time.sleep(0.5)
                self._connect()
        raise ConnectionError(f"lingbot server round-trip failed: {last!r}")


class LingbotInferencePolicy:
    needs_point_cloud: bool = False
    needs_images: bool = True
    needs_language: bool = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29570,
        *,
        prompt: str = "",
        video_guidance_scale: float = 5.0,
        action_guidance_scale: float = 1.0,
        imagined_out: str | None = None,
        episode_offset: int = 0,
        state_key: str = "observation.state",
        image_key_prefix: str = "observation.images.",
    ):
        self.client = _TCPClient(host, port)
        self.prompt = prompt
        self.vgs = video_guidance_scale
        self.ags = action_guidance_scale
        self._dump_dir = imagined_out
        root = os.path.dirname(imagined_out) if imagined_out else None
        self._seed_dir = os.path.join(root, "seed_obs") if root else None
        self._fb_dir = os.path.join(root, "fb_latents") if root else None
        self._gb_dir = os.path.join(root, "gb_latents") if root else None
        self.state_key = state_key
        self.image_key_prefix = image_key_prefix
        self._episode_idx = int(episode_offset) - 1
        self._reset_state()

    def reset(self) -> None:
        self._reset_state()
        self._episode_idx += 1

    def _reset_state(self):
        self._queue: list[np.ndarray] = []
        self._key_frames: list[dict] = []
        self._first = True
        self._need_reset = True
        self._last_action_chunk = None
        self._action_per_frame = None
        self._steps_since_kf = 0
        self._chunk_idx = 0
        self._gb_idx = 0

    def _to_obs(self, obs_dict, task) -> dict:
        msg: dict[str, Any] = {
            self.state_key: np.asarray(obs_dict[self.state_key], np.float32).reshape(-1),
            "task": task or self.prompt,
        }
        for c in _CAMS:
            k = f"{self.image_key_prefix}{c}"
            if k not in obs_dict:
                raise KeyError(f"missing {k} in obs_dict (have {list(obs_dict)})")
            msg[k] = _u8(obs_dict[k])
        return msg

    @staticmethod
    def _dump_pickle(path: str, obj) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, obs_dict: dict, task_description: str = "") -> np.ndarray:
        prompt = task_description or self.prompt
        cur = self._to_obs(obs_dict, prompt)

        if not self._queue:
            ep = max(self._episode_idx, 0)
            if self._need_reset:
                self.client.round_trip(dict(reset=True, prompt=prompt))
                self._need_reset = False
                if self._seed_dir is not None:
                    self._dump_pickle(os.path.join(self._seed_dir, f"ep_{ep:04d}.pkl"),
                                      {"obs": cur, "prompt": prompt})
            elif not self._first:
                gret = self.client.round_trip(dict(
                    obs=self._key_frames, compute_kv_cache=True, imagine=False,
                    state=self._last_action_chunk,
                    return_grounding_latents=bool(self._gb_dir)))
                if self._gb_dir and isinstance(gret, dict) and gret.get("grounding_latents") is not None:
                    self._dump_pickle(
                        os.path.join(self._gb_dir, f"ep_{ep:04d}", f"chunk_{self._gb_idx:04d}.pkl"),
                        np.asarray(gret["grounding_latents"]))
                    self._gb_idx += 1
            self._key_frames = []

            ret = self.client.round_trip(dict(
                obs=cur, prompt=prompt, return_latents=bool(self._fb_dir),
                video_guidance_scale=self.vgs, action_guidance_scale=self.ags))
            if "error" in ret:
                raise RuntimeError(f"lingbot server error: {ret['error']}")
            if "action" not in ret:
                raise RuntimeError(f"lingbot server returned no action: {list(ret)}")

            action = np.asarray(ret["action"], dtype=np.float32)
            if action.ndim != 3 or action.shape[0] != 14:
                raise ValueError(f"expected (14,F,H) action, got {action.shape}")
            self._last_action_chunk = action
            if self._fb_dir and ret.get("latents") is not None:
                self._dump_pickle(
                    os.path.join(self._fb_dir, f"ep_{ep:04d}", f"chunk_{self._chunk_idx:04d}.pkl"),
                    np.asarray(ret["latents"]))
                self._chunk_idx += 1

            _, f_count, horizon = action.shape
            self._action_per_frame = max(1, horizon // 4)
            start_i = 1 if (self._first and f_count > 1) else 0
            self._queue = [action[:, i, j] for i in range(start_i, f_count) for j in range(horizon)]
            self._first = False
            self._steps_since_kf = 0

        out = self._queue.pop(0)
        self._steps_since_kf += 1
        if self._action_per_frame and self._steps_since_kf >= self._action_per_frame:
            self._key_frames.append(cur)
            self._steps_since_kf = 0
        return np.asarray(out, dtype=np.float32).reshape(14)

    def select_action(self, obs_dict, task_description=""):
        return self.predict(obs_dict, task_description)
