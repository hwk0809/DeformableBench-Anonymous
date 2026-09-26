"""LeRobot eval adapter (in-process backend).

``--policy_type lerobot`` — loads an ACT / pi05_evo / diffusion / smolvla
checkpoint (lerobot v0.4.2) and runs closed-loop inference **in the same
process as the newton sim** (the fabricbench ``.venv`` ships both newton and
lerobot 0.4.2, so no out-of-process server is needed — unlike ``gr00t``).

The concrete algorithm is auto-detected from the checkpoint's ``config.type``.
All obs normalization / image transforms / de-normalization are delegated to
lerobot's baked-in pre/post processors + ``predict_action`` — the adapter only
maps the standardized ``obs_dict`` in and squeezes a 14-dim action out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def add_arguments(parser) -> None:
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="closed-loop replan length (clamped to model chunk); "
                             "default = the checkpoint's own value")


def _resolve_checkpoint(path: str) -> str:
    """Accept either a ``.../pretrained_model`` dir or its parent step dir."""
    p = Path(path)
    if (p / "config.json").exists():
        return str(p)
    if (p / "pretrained_model" / "config.json").exists():
        return str(p / "pretrained_model")
    raise FileNotFoundError(f"no lerobot config.json under {path!r} (or its pretrained_model/)")


def build(args) -> "LerobotPolicy":
    ckpt = _resolve_checkpoint(args.checkpoint)
    return LerobotPolicy(ckpt, device=(args.device or "cuda"),
                         n_action_steps=getattr(args, "n_action_steps", None))


class LerobotPolicy:
    """PolicyBase backend for any lerobot policy."""

    needs_point_cloud = False
    needs_images = True
    needs_language = False  # set per-checkpoint in __init__

    def __init__(self, checkpoint: str, device: str = "cuda",
                 n_action_steps: int | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if n_action_steps is not None:
            cap = getattr(config, "chunk_size", None) or getattr(config, "horizon", None) or n_action_steps
            config.n_action_steps = min(int(n_action_steps), int(cap))

        policy_cls = get_policy_class(config.type)  # ← auto-detect ACT / pi05_evo / diffusion / ...
        self.policy = policy_cls.from_pretrained(checkpoint, config=config)
        self.policy.eval()
        self.policy.to(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=checkpoint)

        self.config = config
        self.policy_type = config.type
        self.needs_language = config.type in ("pi0", "pi05", "pi05_evo", "pi0fast", "smolvla")
        self.policy.reset()
        print(f"[lerobot] loaded {config.type} from {checkpoint} on {self.device} "
              f"(n_action_steps={getattr(config, 'n_action_steps', '?')}, "
              f"needs_language={self.needs_language})", flush=True)

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_dict: dict[str, Any], task_description: str) -> np.ndarray:
        from lerobot.utils.control_utils import predict_action

        action = predict_action(
            observation=obs_dict, policy=self.policy, device=self.device,
            preprocessor=self.preprocessor, postprocessor=self.postprocessor,
            use_amp=bool(getattr(self.config, "use_amp", False)),
            task=(task_description or None))
        return action.squeeze(0).float().cpu().numpy().astype(np.float32).reshape(14)
