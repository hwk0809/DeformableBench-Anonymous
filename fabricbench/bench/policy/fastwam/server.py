#!/usr/bin/env python
"""Decoupled FastWAM inference server (runs in the ``fastwam_eval`` conda env).

FastWAM needs transformers==4.49 and a big Wan2.2 stack that cannot coexist with
fabricbench's ``.venv`` — so, exactly like pi05 (serve_lerobot.py) and lingbot, it is
served out-of-process. This standalone script (stdlib only + the dynamically
imported model) loads ``FastWAMInferencePolicy`` from the FastWAM package
(``fastwam.inference``, vendored in the ``thirdparty/FastWAM`` submodule) and serves
it over a length-prefixed pickle protocol; the fabricbench-side client is
``policy/fastwam/inference.py``.

MUST be launched with the FastWAM repo root as CWD (base Wan2.2 / ActionDiT /
``./data/text_embeds_cache`` paths in config.yaml are CWD-relative). ``--fastwam_home``
(the ``thirdparty/FastWAM`` submodule) is put on the path so ``fastwam.*`` imports.
The eval-side ``_launch_fastwam_server`` in eval_sim.py wires this up.

Wire protocol (matches policy/lingbot): 4-byte big-endian length + pickle(dict).
  request {"cmd": "ping"}                              -> {"ok": True}
  request {"cmd": "reset"}                             -> {"ok": True}
  request {"cmd": "predict", "obs": {...}, "instruction": str} -> {"action": (14,) float32}
"""
from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import traceback


def _add_paths(fastwam_home: str) -> None:
    for p in (fastwam_home, os.path.join(fastwam_home, "src")):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_exactly(sock, 4))
    return pickle.loads(_recv_exactly(sock, n))


def _send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--fastwam_home", required=True,
                    help="FastWAM repo root (also the CWD; on sys.path for fastwam.*).")
    ap.add_argument("--ckpt_step", type=int, default=5835)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action_horizon", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--replan_steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_embeds_cache_dir", default=None)
    ap.add_argument("--task_name", default="silk_grasp")
    args = ap.parse_args()

    os.environ.setdefault("FASTWAM_HOME", args.fastwam_home)
    _add_paths(args.fastwam_home)
    os.chdir(args.fastwam_home)  # base-model / ActionDiT / text-cache paths are CWD-relative

    from fastwam.inference.robot_inference import FastWAMInferencePolicy  # noqa: E402

    print(f"[fastwam-server] loading policy from {args.run_dir} (ckpt_step={args.ckpt_step}) ...",
          flush=True)
    policy = FastWAMInferencePolicy(
        run_dir=os.path.abspath(args.run_dir),
        ckpt_step=args.ckpt_step,
        device=args.device,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        text_embeds_cache_dir=args.text_embeds_cache_dir,
        task_name=args.task_name,
    )
    print("[fastwam-server] policy ready.", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(8)
    print(f"[fastwam-server] listening on :{args.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                req = _recv_msg(conn)
                cmd = req.get("cmd")
                if cmd == "ping":
                    _send_msg(conn, {"ok": True})
                elif cmd == "reset":
                    policy.reset()
                    _send_msg(conn, {"ok": True})
                elif cmd == "predict":
                    import numpy as np
                    action = policy.predict(req["obs"], req.get("instruction", ""))
                    action = np.asarray(action, dtype=np.float32).reshape(14)
                    _send_msg(conn, {"action": action})
                else:
                    _send_msg(conn, {"error": f"unknown cmd {cmd!r}"})
        except (ConnectionError, EOFError):
            pass  # client (one episode worker) disconnected; wait for the next
        except Exception:
            traceback.print_exc()
            try:
                _send_msg(conn, {"error": traceback.format_exc()})
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
