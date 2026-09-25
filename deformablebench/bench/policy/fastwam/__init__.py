"""FastWAM eval backend.

FastWAM runs in its own ``fastwam_eval`` conda env and is served over raw TCP by
``server.py``. The sim-side eval process only imports this package as a thin client.
"""

from .adapter import add_arguments, build  # noqa: F401
from .inference import FastwamInferencePolicy  # noqa: F401

__all__ = ["FastwamInferencePolicy", "build", "add_arguments"]
