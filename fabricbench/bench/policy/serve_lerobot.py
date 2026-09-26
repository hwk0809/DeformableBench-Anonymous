"""Standalone LeRobot policy server (ZMQ) — RUN IN THE POLICY'S OWN ENV.

Decouples a (patched, version-sensitive) policy env from the newton+Kit sim env,
the same pattern as ``policy/gr00t``. A VLA like ``pi05_evo`` needs a *patched*
``transformers`` (``transformers.models.siglip.check`` — openpi's "transformers
replace") that only lives in its trained conda env (e.g. ``lerobot_v042``);
cramming that into fabricbench's ``.venv`` would risk the sim stack. Instead this
server loads the checkpoint **in that env** and serves ``get_action`` over ZMQ;
the fabricbench eval connects as a client (``--policy_type lerobot_remote``).

Standalone by design — imports ONLY ``lerobot`` + ``zmq`` + ``msgpack`` (NOT
``fabricbench``), so it runs under the policy env's python. The policy conda env
(e.g. ``lerobot_v042``) must have ``pyzmq`` + ``msgpack`` installed (both are
innocuous comms libs)::

    conda run -n lerobot_v042 --no-capture-output \
      python .../policy/serve_lerobot.py --checkpoint <.../pretrained_model> --port 5556
"""

from __future__ import annotations

import os
import sys

# Run as `python .../policy/serve_lerobot.py`, python puts this policy/ dir (which
# contains lerobot.py / lerobot_remote.py) on sys.path[0] and would SHADOW the real
# installed `lerobot` package. Drop our own dir so `import lerobot` finds the env's.
_self_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _self_dir]

import argparse
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq


def _pack(d):
    return msgpack.packb(d, default=mnp.encode)


def _unpack(b):
    return msgpack.unpackb(b, object_hook=mnp.decode, raw=False)


def _resolve_checkpoint(path: str) -> str:
    p = Path(path)
    if (p / "config.json").exists():
        return str(p)
    if (p / "pretrained_model" / "config.json").exists():
        return str(p / "pretrained_model")
    raise FileNotFoundError(f"no lerobot config.json under {path!r}")


class _Server:
    def __init__(self, checkpoint: str, device: str = "cuda", n_action_steps=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        # lerobot 0.6.0 moved this module (lerobot.utils -> lerobot.common); the
        # function itself is unchanged (same 8 params, same order). Both paths are
        # kept so ONE server script serves the 0.4.2 envs (pi05_evo's lerobot_v042)
        # and the 0.6.0 env (groot N1.7) without branching anywhere else.
        try:
            from lerobot.utils.control_utils import predict_action  # lerobot <= 0.5.x
        except ImportError:
            from lerobot.common.control_utils import predict_action  # lerobot >= 0.6.0

        self._predict_action = predict_action
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = _resolve_checkpoint(checkpoint)
        config = PreTrainedConfig.from_pretrained(ckpt)
        if n_action_steps is not None:
            cap = getattr(config, "chunk_size", None) or getattr(config, "horizon", None) or n_action_steps
            config.n_action_steps = min(int(n_action_steps), int(cap))
        # pi05_evo (and similar) bake `device` into the saved config and run
        # `self.model.to(config.device)` inside from_pretrained — so the initial
        # load lands on the GPU used at training time (usually cuda:0), ignoring
        # our --device. Override it up front so the model loads straight onto the
        # target GPU (avoids OOM when the training GPU is busy on the eval box).
        if hasattr(config, "device"):
            config.device = str(self.device)
        self.policy = get_policy_class(config.type).from_pretrained(ckpt, config=config)
        self.policy.eval()
        self.policy.to(self.device)
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=ckpt)
        self.config = config
        self.policy.reset()
        print(f"[serve_lerobot] loaded {config.type} from {ckpt} on {self.device}", flush=True)

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def get_action(self, obs: dict, task: str) -> np.ndarray:
        a = self._predict_action(
            observation=obs, policy=self.policy, device=self.device,
            preprocessor=self.pre, postprocessor=self.post,
            use_amp=bool(getattr(self.config, "use_amp", False)), task=(task or None))
        return a.squeeze(0).float().cpu().numpy().astype(np.float32).reshape(14)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=None)
    args = ap.parse_args()

    srv = _Server(args.checkpoint, args.device, args.n_action_steps)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[serve_lerobot] ready on :{args.port}", flush=True)
    while True:
        req = _unpack(sock.recv())
        ep = req.get("endpoint")
        try:
            if ep == "ping":
                sock.send(_pack({"ok": True}))
            elif ep == "reset":
                srv.reset()
                sock.send(_pack({"ok": True}))
            elif ep == "get_action":
                d = req["data"]
                act = srv.get_action(d["observation"], d.get("task", ""))
                sock.send(_pack({"action": act}))
            else:
                sock.send(_pack({"error": f"unknown endpoint {ep!r}"}))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            sock.send(_pack({"error": str(e)}))


if __name__ == "__main__":
    main()
