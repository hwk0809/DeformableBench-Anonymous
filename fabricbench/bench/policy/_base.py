"""Policy adapter contract for the closed-loop eval.

``--policy_type`` selects an inference *backend / runtime*, not an algorithm.
Each backend lives in ``policy/<name>.py`` (single file) or ``policy/<name>/``
(a subpackage, when it also ships a train/serve launcher + docs, e.g.
``gr00t/``) and exposes::

    def build(args) -> PolicyBase: ...

``build()`` returns an object implementing the ``PolicyBase`` protocol below.
The (future R5) eval loop never imports a specific backend — it calls
``load_policy(name, args)`` which dynamic-imports
``fabricbench.bench.policy.<name>``.

Backend-vs-algorithm distinction: one backend can host many algorithms (the
``lerobot`` backend serves ACT / DiffusionPolicy / Pi0 and auto-detects which
from the checkpoint), whereas a backend like ``gr00t`` is a single runtime that
talks to an out-of-process model server over ZMQ.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

_PKG = "fabricbench.bench.policy"


@runtime_checkable
class PolicyBase(Protocol):
    """Closed-loop control policy.

    The eval calls ``reset()`` at the start of every episode, then
    ``predict(obs_dict, task_description)`` once per control tick. The policy
    owns all internal state (action-chunk queues, replanning, normalization,
    history buffers).

    Capability flags (class attributes): the eval reads these to decide which
    observation modalities to populate, replacing eval-side
    ``if policy_type == "..."`` special-casing.
        - needs_point_cloud: populate ``observation.point_cloud``.
        - needs_images: populate ``observation.images.<cam>`` (default True).
        - needs_language: pass a non-empty ``task_description``.
    """

    needs_point_cloud: bool = False
    needs_images: bool = True
    needs_language: bool = False

    def reset(self) -> None:
        """Clear any per-episode internal state."""
        ...

    def predict(self, obs_dict: dict[str, Any], task_description: str) -> np.ndarray:
        """Return the next 14-dim joint-target action.

        Args:
            obs_dict: standardized observation keys for the current tick.
                Required:
                  - ``observation.state``: (14,) float32 robot joint state,
                    layout ``[left_arm_q1..q6, left_gripper,
                    right_arm_q1..q6, right_gripper]``.
                  - ``observation.images.<cam>`` for each eval camera:
                    (H, W, 3) uint8 RGB. Sim cams: ``static_cam``,
                    ``left_hand_cam``, ``right_hand_cam``.
            task_description: natural-language instruction (may be empty).

        Returns:
            (14,) float32 array, same layout as ``observation.state`` — absolute
            joint targets fed straight to the sim.
        """
        ...


def load_policy(name: str, args) -> PolicyBase:
    """Dynamic-dispatch factory.

    ``name`` matches ``policy/<name>.py`` or ``policy/<name>/``, which must
    expose ``build(args)``. ``args`` is the eval script's parsed argparse
    namespace.
    """
    if "." in name or name.startswith("_"):
        raise ValueError(f"Invalid policy name: {name!r}")
    try:
        mod = importlib.import_module(f"{_PKG}.{name}")
    except ImportError as e:
        raise ImportError(
            f"Could not import policy adapter `policy/{name}`. "
            f"Available adapters: see `ls src/fabricbench/bench/policy/`. "
            f"Original error: {e}"
        ) from e
    if not hasattr(mod, "build"):
        raise AttributeError(
            f"policy/{name} exists but does not expose a `build(args)` factory."
        )
    return mod.build(args)


def add_policy_arguments(parser, name: str) -> None:
    """If ``policy/<name>`` defines ``add_arguments(parser)``, call it.

    Lets each adapter declare its own CLI flags without bloating the eval loop.
    """
    if "." in name or name.startswith("_"):
        return
    try:
        mod = importlib.import_module(f"{_PKG}.{name}")
    except ImportError:
        return
    if hasattr(mod, "add_arguments"):
        mod.add_arguments(parser)
