"""Newton physics integration layer for the deformable manipulation benchmark.

This is the only hand-written physics layer in the repo. It wraps the Style3D
fork of Newton (open-source ``SolverStyle3D``) behind a small, bench-facing
``DeformableEnv``. Three responsibilities, one each:

- ``builder``   build a ``newton.Model`` from assets
- ``solvers``   construct the cloth solver (factory; SDK branch reserved)
- ``env``       own Model/State, run the (co-)sim loop, expose the bench interface
"""

from fabricbench.sim.builder import RobotConfig, SceneConfig
from fabricbench.sim.env import DeformableEnv

__all__ = ["SceneConfig", "RobotConfig", "DeformableEnv"]
