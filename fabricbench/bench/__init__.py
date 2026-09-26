"""Benchmark pipeline (solver-agnostic).

Mounted incrementally:
- P1: ``common.asset`` (asset loading) + success/metric evaluation, no robot.
- P2: robot kinematics/planning/skills + ``common.sim_executor``.
- P3+: policy eval, Kit render bridge, LeRobot data collection.
"""
