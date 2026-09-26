"""ZMQ-client adapter for a decoupled LeRobot policy server.

``--policy_type lerobot_remote`` — mirrors ``policy/gr00t``: the heavy /
version-sensitive policy env runs ``serve_lerobot.py`` (loads the checkpoint,
serves ``get_action`` over ZMQ), and this thin client — running inside fabricbench's
sim ``.venv`` — forwards the standardized ``obs_dict`` and gets back a 14-dim
action. Used for policies whose env can't coexist with the newton+Kit sim (e.g.
``pi05_evo``, which needs a patched ``transformers``). ``act`` doesn't need this
and runs in-process via ``policy/lerobot.py``.

The eval orchestrator (``bench/eval_sim.py``) launches + tears down the server when
``--policy_env`` is given; otherwise point ``--server_port`` at a server you
started yourself.
"""

from __future__ import annotations

import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _pack(d):
    return msgpack.packb(d, default=mnp.encode)


def _unpack(b):
    return msgpack.unpackb(b, object_hook=mnp.decode, raw=False)


class _Client:
    def __init__(self, host, port, timeout_ms=180000):
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._conn()

    def _conn(self):
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.s.setsockopt(zmq.LINGER, 0)
        self.s.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.s.send(_pack(req))
        r = _unpack(self.s.recv())
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(f"policy server error: {r['error']}")
        return r

    def ping(self):
        try:
            self.call("ping")
            return True
        except Exception:
            self._conn()
            return False


class LerobotRemote:
    """PolicyBase ZMQ client for a lerobot policy server."""

    needs_point_cloud = False
    needs_images = True
    needs_language = True  # pass task_description through; the server ignores it for non-VLA

    def __init__(self, host="127.0.0.1", port=5556, task=""):
        self.cli = _Client(host, port)
        self.task = task
        # wait (up to ~5 min) for the server to finish loading the model
        for _ in range(150):
            if self.cli.ping():
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"lerobot policy server not reachable at {host}:{port}")

    def reset(self):
        self.cli.call("reset")

    def predict(self, obs_dict, task_description="") -> np.ndarray:
        r = self.cli.call("get_action",
                          {"observation": obs_dict, "task": (task_description or self.task)})
        return np.asarray(r["action"], dtype=np.float32).reshape(14)


def add_arguments(parser) -> None:
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=5556)
    # Default empty on purpose: a wrong instruction does NOT raise, it just makes
    # the VLA dumber (that is how the fold pi05 eval silently scored 13%). The
    # real-robot entry point pushes the task yaml's `task_description` in here, so
    # an empty value at build() time means nobody resolved it -> fail loudly.
    parser.add_argument("--server_task", default="")

    # Optional one-command server launch from real_eval (auto-spawn + auto-kill).
    # When omitted, connect to a server you started yourself.
    parser.add_argument("--server_python", default=None,
                        help="Python of the policy's own lerobot env. Set => auto-spawn server.")
    parser.add_argument("--server_cwd", default=None,
                        help="Working dir for the server. Needed for GR00T-N1.7 on lerobot>=0.6.0: "
                             "its processor hardcodes the backbone hub id 'nvidia/Cosmos-Reason2-2B', "
                             "so point this at a dir holding that symlink to stay offline.")
    parser.add_argument("--server_cuda_visible_devices", default=None,
                        help="Pin the policy server to a GPU (kept off the render GPU).")
    parser.add_argument("--server_wait_s", type=float, default=420.0)


def build(args):
    task = getattr(args, "server_task", "") or ""
    if not task.strip():
        raise ValueError(
            "lerobot_remote got an empty --server_task. A VLA does not error on a wrong "
            "instruction, it just gets dumber, so refuse to start instead. Pass --task=<config "
            "name> (the instruction is resolved from config/task/<name>.yaml) or set "
            "--server_task explicitly."
        )
    return LerobotRemote(host=getattr(args, "server_host", "127.0.0.1"),
                         port=getattr(args, "server_port", 5556),
                         task=task)
