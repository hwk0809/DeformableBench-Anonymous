"""FastWAM policy adapter for closed-loop sim eval (thin TCP client side).

FastWAM runs in its own ``fastwam_eval`` conda env (transformers 4.49 + Wan2.2),
incompatible with deformablebench's .venv, so it is served out-of-process by
``policy/fastwam/server.py`` and this adapter is just the client. When the
``--fastwam_server_python`` launch args are given, eval_sim auto-spawns the server
(and kills it per-cell); otherwise it connects to an already-running one.
"""

from __future__ import annotations


def add_arguments(parser) -> None:
    parser.add_argument("--fastwam_host", type=str, default="127.0.0.1")
    parser.add_argument("--fastwam_port", type=int, default=5570)
    parser.add_argument("--fastwam_task", type=str, default="dual-arm grasp silk and lift",
                        help="Instruction sent to the server; MUST match the single "
                             "precomputed text-embed cache entry of the checkpoint.")

    # Optional one-command server launch from eval_sim/benchmark (auto-spawn + kill
    # per cell). Omit to connect to an already-running server.
    parser.add_argument("--fastwam_server_python", default=None,
                        help="Python of the fastwam_eval conda env. Set => auto-spawn server.")
    parser.add_argument("--fastwam_home", default=None,
                        help="FastWAM repo root (server CWD + on sys.path). "
                             "Defaults to the thirdparty/FastWAM submodule.")
    parser.add_argument("--fastwam_run_dir", default=None,
                        help="Trained run dir (config.yaml + dataset_stats.json + checkpoints/).")
    parser.add_argument("--fastwam_ckpt_step", type=int, default=5835)
    parser.add_argument("--fastwam_action_horizon", type=int, default=32)
    parser.add_argument("--fastwam_num_inference_steps", type=int, default=20)
    parser.add_argument("--fastwam_replan_steps", type=int, default=32)
    parser.add_argument("--fastwam_seed", type=int, default=0)
    parser.add_argument("--fastwam_text_cache", default=None,
                        help="Explicit text_embeds_cache dir (the cache is NOT next to run_dir).")
    parser.add_argument("--fastwam_cuda_visible_devices", default=None,
                        help="Pin the fastwam server to a GPU (kept off the render GPU).")
    parser.add_argument("--fastwam_server_wait_s", type=float, default=600.0,
                        help="Max wait for the server (12G model load) to accept TCP.")


def build(args):
    from deformablebench.bench.policy.fastwam.inference import FastwamInferencePolicy

    return FastwamInferencePolicy(
        host=getattr(args, "fastwam_host", "127.0.0.1"),
        port=int(getattr(args, "fastwam_port", 5570)),
        prompt=getattr(args, "fastwam_task", "dual-arm grasp silk and lift"),
    )
