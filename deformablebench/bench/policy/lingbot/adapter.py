"""LingBot-VA policy adapter for closed-loop sim eval."""

from __future__ import annotations

import os


def add_arguments(parser) -> None:
    parser.add_argument("--lingbot_host", type=str, default="127.0.0.1",
                        help="Host/IP of the lingbot-va TCP policy server.")
    parser.add_argument("--lingbot_port", type=int, default=29570,
                        help="lingbot-va policy server port.")
    parser.add_argument("--lingbot_video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--lingbot_action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--imagined_out", type=str, default=None,
                        help="Optional root for LingBot latent/seed dumps.")

    # Optional one-command server launch from eval_sim/benchmark. When omitted,
    # eval connects to an already-running server.
    parser.add_argument("--lingbot_server_launcher", default=None,
                        help="Path to lingbot-va/evaluation/launch_tcp_server.sh.")
    parser.add_argument("--lingbot_server_python", default=None,
                        help="Alternative direct python launcher for tcp_policy_server.py.")
    parser.add_argument("--lingbot_server_repo", default=None,
                        help="lingbot-va repo root used as launcher cwd/REPO. "
                             "Defaults to the thirdparty/lingbot-va submodule.")
    parser.add_argument("--lingbot_dataset_dir", default=None,
                        help="LeRobot dataset root whose meta/stats.json supplies "
                             "the action quantiles used by the LingBot checkpoint.")
    parser.add_argument("--lingbot_config_name", default="silk_infer")
    parser.add_argument("--lingbot_master_port", type=int, default=29571)
    parser.add_argument("--lingbot_conda_env", default="lingbot-va")
    parser.add_argument("--lingbot_conda_base", default="~/miniconda3")
    parser.add_argument("--lingbot_cuda_visible_devices", default=None)
    parser.add_argument("--lingbot_save_root", default=None)
    parser.add_argument("--lingbot_server_wait_s", type=float, default=420.0)


def build(args):
    from deformablebench.bench.policy.lingbot.inference import LingbotInferencePolicy

    imagined_root = getattr(args, "imagined_out", None)
    imagined_dumps = os.path.join(imagined_root, "dumps") if imagined_root else None
    prompt = getattr(args, "task_description", None) or getattr(args, "task", "") or ""
    episode_offset = int(getattr(args, "_single_episode_id", 0) or 0)
    return LingbotInferencePolicy(
        host=getattr(args, "lingbot_host", "127.0.0.1"),
        port=int(getattr(args, "lingbot_port", 29570)),
        prompt=prompt,
        video_guidance_scale=getattr(args, "lingbot_video_guidance_scale", 5.0),
        action_guidance_scale=getattr(args, "lingbot_action_guidance_scale", 1.0),
        imagined_out=imagined_dumps,
        episode_offset=episode_offset,
    )
