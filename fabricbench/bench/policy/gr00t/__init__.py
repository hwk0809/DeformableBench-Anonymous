"""GR00T baseline (NVIDIA Isaac-GR00T N1.7, a 3B VLA).

One home for the whole baseline:
  - ``adapter.py`` — ZMQ eval adapter (the only part that runs in the sim env).
  - ``train.sh`` / ``serve.sh`` — thin facades over the submodule's launchers.
  - ``README.md`` — env setup + train + serve + eval, all in one place.

The eval dispatch (``policy._base.load_policy``) imports this package and calls
``build(args)`` / ``add_arguments(parser)``, re-exported here from ``adapter``.
"""

from .adapter import GR00TPolicy, add_arguments, build  # noqa: F401

__all__ = ["GR00TPolicy", "build", "add_arguments"]
