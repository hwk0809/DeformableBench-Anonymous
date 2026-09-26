"""Cloth solver factory.

The benchmark core uses the open-source ``SolverStyle3D`` (fully reproducible).
"""

from __future__ import annotations

import newton

from fabricbench.sim.builder import SceneConfig
from fabricbench.sim.style3d_asset import configure_style3d_solver_collision


def make_cloth_solver(name: str, model, state, cfg: SceneConfig):
    """Construct the cloth solver selected by ``name``."""

    if name == "style3d":
        params = cfg.cloth_params
        solver = newton.solvers.SolverStyle3D(
            model,
            iterations=max(1, int(params.solver_iterations)),
            linear_iterations=max(1, int(params.linear_iterations)),
            enable_translation_preconditioner=bool(cfg.translation_preconditioner),
            # FUSE1/FUSE2: fabricbench default ON (fork ctor default False) -> always
            # passed when on; `physics.style3d_pcg_fused: false` omits the kwarg.
            **({"pcg_fused": True} if bool(params.style3d_pcg_fused) else {}),
        )
        # Self-collision radius/stiffness + BVH rebuild.
        configure_style3d_solver_collision(solver, state, params)
        return solver

    raise ValueError(f"unknown cloth solver backend: {name!r}")


def make_robot_solver(model, cfg: SceneConfig):
    """Construct the rigid-body solver for PD control mode (SolverMuJoCo).

    Only used when ``cfg.control_mode == 'pd'``; kinematic mode poses the robot
    via ``eval_fk`` and needs no robot solver. The robot's PD target gains/mode
    are set on the model by the env (robot.py) before this is constructed.
    """

    # njmax/nconmax: the auto-estimate from the initial state is too small once a
    # free rigid object sits in the gripper (box vs finger meshes -> many contact
    # rows; observed "nefc overflow - please increase njmax to 81" mid-grasp,
    # which silently drops contacts and NaNs the state). Memory-only cost.
    return newton.solvers.SolverMuJoCo(
        model, iterations=100, ls_iterations=50, njmax=512, nconmax=256,
    )
