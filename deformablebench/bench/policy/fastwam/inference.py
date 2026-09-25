"""FastWAM TCP inference client (runs in deformablebench's .venv / eval env).

Thin client for ``policy/fastwam/server.py`` (which runs in the ``fastwam_eval``
conda env). Same wire protocol as lingbot: 4-byte big-endian length + pickle(dict).

FastWAM's text-embedding cache holds exactly ONE precomputed prompt, so we always
send the fixed instruction the checkpoint was cached with
("dual-arm grasp silk and lift" → DEFAULT_PROMPT → the single cache file), NOT the
per-call task string, to guarantee the cache hit.
"""

from __future__ import annotations

import logging
import pickle
import socket
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

_CAMS = ["static_cam", "left_hand_cam", "right_hand_cam"]


def _u8(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


class _TCPClient:
    """Length-prefixed pickle client with one reconnect retry (mirrors lingbot)."""

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
        raise ConnectionError(f"fastwam server round-trip failed: {last!r}")


class FastwamInferencePolicy:
    needs_point_cloud: bool = False
    needs_images: bool = True
    needs_language: bool = True

    def __init__(self, host="127.0.0.1", port=5570, *,
                 prompt="dual-arm grasp silk and lift",
                 state_key="observation.state",
                 image_key_prefix="observation.images."):
        self.client = _TCPClient(host, port)
        self.prompt = prompt
        self.state_key = state_key
        self.image_key_prefix = image_key_prefix

    def reset(self) -> None:
        r = self.client.round_trip({"cmd": "reset"})
        if r.get("error"):
            raise RuntimeError(f"fastwam reset failed: {r['error']}")

    def predict(self, obs_dict, task_description="") -> np.ndarray:
        msg = {
            "cmd": "predict",
            # Always the cached instruction, not the per-call task (single-prompt cache).
            "instruction": self.prompt,
            "obs": self._to_obs(obs_dict),
        }
        r = self.client.round_trip(msg)
        if r.get("error"):
            raise RuntimeError(f"fastwam predict failed: {r['error']}")
        return np.asarray(r["action"], dtype=np.float32).reshape(14)

    def _to_obs(self, obs_dict) -> dict:
        out = {self.state_key: np.asarray(obs_dict[self.state_key], np.float32).reshape(-1)}
        for c in _CAMS:
            k = f"{self.image_key_prefix}{c}"
            if k not in obs_dict:
                raise KeyError(f"missing {k} in obs_dict (have {list(obs_dict)})")
            out[k] = _u8(obs_dict[k])
        return out
