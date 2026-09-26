"""LingBot-VA eval backend.

LingBot-VA runs in its own conda env and exposes a raw TCP policy server. The
sim-side eval process only imports this package as a thin client.
"""

from .adapter import add_arguments, build  # noqa: F401
from .inference import LingbotInferencePolicy  # noqa: F401

__all__ = ["LingbotInferencePolicy", "build", "add_arguments"]
