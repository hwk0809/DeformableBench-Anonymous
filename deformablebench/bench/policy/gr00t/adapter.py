"""GR00T eval adapter (backend = out-of-process ZMQ model server).

GR00T needs its own env (py3.10 / torch 2.7), incompatible with the sim/eval
env, so it runs as a standalone server (submodule's
``gr00t/eval/run_gr00t_server.py``, launched via ``serve.sh``) and this adapter
is a thin ZMQ client. The wire format matches GR00T's ``MsgSerializer``
(msgpack + msgpack_numpy). Actions are chunked: GR00T predicts a 16-step horizon
and we execute ``exec_horizon`` of them before re-querying.

This ZMQ decoupling is *why* the heavy GR00T env can conflict with the core
install and still be evaluated: the eval only speaks the wire protocol. Only
``pyzmq`` + ``msgpack`` + ``msgpack-numpy`` are needed here (the ``[gr00t]``
extra); the model itself lives in the separate GR00T env. See ``README.md``.
"""

from __future__ import annotations

from collections import deque
import logging
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _to_bytes(d):
    return msgpack.packb(d, default=mnp.encode)


def _from_bytes(b):
    return msgpack.unpackb(b, object_hook=mnp.decode, raw=False)


# sim camera name -> gr00t video modality key
# (see the submodule's examples/DeformableBimanual/modality.json)
_CAM_MAP = {
    "static_cam": "static",
    "left_hand_cam": "left_wrist",
    "right_hand_cam": "right_wrist",
}

# 14-D layout: [0:6]=left arm, 6=left gripper, [7:13]=right arm, 13=right gripper.
# The leash below is in RADIANS, so it applies to the 12 arm joints only — the
# grippers live in a normalized [0,1] and take the commanded value verbatim.
_ARM_DIMS = np.r_[0:6, 7:13]


class _Client:
    def __init__(self, host, port, timeout_ms=120000):
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._conn()

    def _conn(self):
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.s.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.s.send(_to_bytes(req))
        r = _from_bytes(self.s.recv())
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(f"GR00T server error: {r['error']}")
        return r

    def get_action(self, obs):
        return tuple(self.call("get_action", {"observation": obs, "options": None}))

    def reset(self):
        try:
            return self.call("reset", {"options": None})
        except Exception:
            return {}

    def ping(self):
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            self._conn()
            return False


class GR00TPolicy:
    needs_point_cloud = False
    needs_images = True
    needs_language = True

    def __init__(self, host, port, task, exec_horizon=16, anchor="state", anchor_leash=0.2):
        self.cli = _Client(host, port)
        self.task = task
        self.exec_horizon = int(exec_horizon)
        self.anchor = str(anchor)
        self.anchor_leash = float(anchor_leash)
        self._last_cmd = None
        self._queue: deque = deque()
        # Wait (up to ~4 min) for the server to finish loading the model.
        for _ in range(120):
            if self.cli.ping():
                break
            time.sleep(2)

    def reset(self):
        self._queue.clear()
        self._last_cmd = None
        self.cli.reset()

    def _anchor_state(self, measured):
        """The 14-D vector sent to the server as ``state``.

        GR00T uses it twice: as the proprioceptive input, and — because this
        embodiment declares the arms ``ActionRepresentation.RELATIVE`` — as the
        ORIGIN the predicted deltas are added to (``unapply_action``). Which
        vector is in-distribution therefore depends on how the training data
        defined ``state``:

        - ``state`` (default, = the official SO100 client): send the measured
          joints. Correct when the data was teleoperated on the same hardware,
          where ``action`` (leader) already leads ``state`` (follower) by the
          tracking lag, so the learned delta carries that lead.
        - ``command``: send the last action we issued. Our datasets are recorded
          in ``control_mode: kinematic``, where ``action[t] == state[t+1]``
          exactly (measured to 6e-8) — the demonstrator has zero tracking lag,
          so the learned delta is pure displacement with no lead term. On
          hardware the measured joints trail the last command (~80% of it is
          just the inference latency: the anchor frame is captured before the
          ~105 ms forward pass), and in relative mode that trail is silently
          dropped at every re-query. Rig log, exec_horizon=8: the command jumped
          BACKWARD 2.3–4.8 deg at each chunk boundary and the arm realized only
          60–76% of each chunk's commanded motion.

        ``anchor_leash`` caps how far the command origin may run ahead of
        reality, so a blocked arm cannot wind the open-loop origin up forever.
        """
        if self.anchor != "command" or self._last_cmd is None:
            return measured
        anchor = self._last_cmd.copy()
        if self.anchor_leash > 0:
            gap = anchor[_ARM_DIMS] - measured[_ARM_DIMS]
            anchor[_ARM_DIMS] = measured[_ARM_DIMS] + np.clip(
                gap, -self.anchor_leash, self.anchor_leash
            )
        return anchor

    def _infer(self, obs_dict, task_description):
        measured = np.asarray(obs_dict["observation.state"], dtype=np.float32).reshape(-1)
        st = self._anchor_state(measured)
        if self._last_cmd is not None:
            # e = how far the measurement trails the command we last issued. In
            # sim this is ~0 by construction; on the rig it is what relative
            # actions discard at every re-anchor. Logged in BOTH modes so the
            # baseline run measures it too (official metric_boundary_jump).
            gap = np.abs(self._last_cmd[_ARM_DIMS] - measured[_ARM_DIMS])
            i = int(np.argmax(gap))
            logging.info(
                "[gr00t] query: anchor=%s  e_max=%.4f rad (%.1f deg) @arm dim %d",
                self.anchor, float(gap[i]), float(np.degrees(gap[i])), int(_ARM_DIMS[i]),
            )
        obs = {
            "video": {},
            "state": {
                "left_arm": st[0:6].reshape(1, 1, 6),
                "left_gripper": st[6:7].reshape(1, 1, 1),
                "right_arm": st[7:13].reshape(1, 1, 6),
                "right_gripper": st[13:14].reshape(1, 1, 1),
            },
            "language": {
                "annotation.human.task_description": [[task_description or self.task]],
            },
        }
        for cam, key in _CAM_MAP.items():
            im = obs_dict.get(f"observation.images.{cam}")
            if im is None:
                continue
            obs["video"][key] = np.asarray(im, dtype=np.uint8)[None, None, ...]
        act, _ = self.cli.get_action(obs)
        # each value is (B=1, 16, D) -> take batch 0; concat to (16, 14) absolute joint targets
        chunk = np.concatenate(
            [
                np.asarray(act["left_arm"])[0],
                np.asarray(act["left_gripper"])[0],
                np.asarray(act["right_arm"])[0],
                np.asarray(act["right_gripper"])[0],
            ],
            axis=-1,
        ).astype(np.float32)
        return chunk

    def predict(self, obs_dict, task_description=""):
        if not self._queue:
            chunk = self._infer(obs_dict, task_description)
            for i in range(min(self.exec_horizon, chunk.shape[0])):
                self._queue.append(chunk[i])
        act = self._queue.popleft()
        # Remember what we actually hand out: it is the anchor for the NEXT
        # query in "command" mode, and the reference for the e log in both.
        self._last_cmd = np.asarray(act, dtype=np.float32).reshape(-1).copy()
        return act


def add_arguments(parser):
    parser.add_argument("--gr00t_host", default="127.0.0.1")
    parser.add_argument("--gr00t_port", type=int, default=5555)
    parser.add_argument("--gr00t_task", default="dual-arm grasp silk and lift")
    parser.add_argument("--gr00t_exec_horizon", type=int, default=16)
    parser.add_argument("--gr00t_anchor", default="state", choices=("state", "command"),
                        help="What to send as `state` = the origin the RELATIVE arm deltas are "
                             "added to. 'state' (default) = measured joints, matching the "
                             "official SO100 client. 'command' = the last action we issued, "
                             "matching how our kinematic-sim datasets define state "
                             "(action[t] == state[t+1]). See GR00TPolicy._anchor_state.")
    parser.add_argument("--gr00t_anchor_leash", type=float, default=0.2,
                        help="[--gr00t_anchor command] max radians the command origin may run "
                             "ahead of the measured arm joints (windup guard; grippers exempt).")

    # Optional one-command server launch from eval_sim/benchmark (auto-spawn +
    # auto-kill per cell). When omitted, connect to an already-running server.
    parser.add_argument("--gr00t_server_python", default=None,
                        help="Python of the gr00t conda env (py3.10). Set => auto-spawn server.")
    parser.add_argument("--gr00t_server_repo", default=None,
                        help="Isaac-GR00T repo dir (server cwd).")
    parser.add_argument("--gr00t_modality_config", default=None,
                        help="Override modality-config .py (default: <repo>/examples/DeformableBimanual/dual_piper_config.py).")
    parser.add_argument("--gr00t_cuda_visible_devices", default=None,
                        help="Pin the gr00t server to a GPU (kept off the render GPU).")
    parser.add_argument("--gr00t_server_wait_s", type=float, default=420.0)


def build(args):
    return GR00TPolicy(
        host=getattr(args, "gr00t_host", "127.0.0.1"),
        port=getattr(args, "gr00t_port", 5555),
        task=getattr(args, "gr00t_task", "dual-arm grasp silk and lift"),
        exec_horizon=getattr(args, "gr00t_exec_horizon", 16),
        anchor=getattr(args, "gr00t_anchor", "state"),
        anchor_leash=getattr(args, "gr00t_anchor_leash", 0.2),
    )
