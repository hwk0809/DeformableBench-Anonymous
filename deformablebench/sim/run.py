"""Unified scene/task runner — build a DeformableEnv and run it with a chosen backend.

    uv run python -m deformablebench.sim.run --viewer gl              # watch in OpenGL (debug)
    uv run python -m deformablebench.sim.run --viewer null            # headless, trajectory stats
    uv run python -m deformablebench.sim.run --viewer usd --output out.usd

``--viewer`` selects how to visualize; the simulation (``DeformableEnv``) is the
same regardless. This is the day-to-day DEBUG entry (native Newton viewers).
Photorealistic Kit observation for VLA eval plugs in at P3 via the eval entry,
not here. ``--task`` loads a bench task config once ``bench/`` is mounted (P1+);
until then it runs the default bag-drop scene.
"""

from __future__ import annotations

import argparse

import numpy as np

from deformablebench.sim import DeformableEnv, SceneConfig


def make_viewer(kind: str, frames: int, output: str):
    import newton.viewer as V

    if kind == "gl":
        return V.ViewerGL()
    if kind == "usd":
        return V.ViewerUSD(output_path=output, num_frames=frames)
    return V.ViewerNull(num_frames=frames)


def build_config(
    task: str | None, device: str | None, robot: bool = False, control_mode: str = "kinematic"
) -> SceneConfig:
    """Resolve a task name to a SceneConfig via the unified asset_meta framework.

    Every asset (cloth + bag) is a DeformableAsset under the top-level ``assets/``
    dir, described by its ``asset_meta.yaml`` (meters; scale from the meta). P1/P2
    use the asset geometry + scale; per-asset material/pose mapping (asset.physics)
    is a later calibration (§9.3).
    """
    # CLI convenience: resolve a task name via the bench asset framework. (This is
    # the one sim entry that reaches up to bench; the sim *library* stays clean.)
    import os

    from deformablebench.bench.common.asset import DeformableAsset, load_robot
    from deformablebench.paths import asset_dir
    from deformablebench.sim.builder import cloth_params_from_physics

    if task and os.path.isdir(task):
        # external asset dir (e.g. a generated-asset library outside the repo)
        resolved = task
    else:
        name = "bag/nonwoven_small_6" if task in (None, "bag_drop") else f"cloth/{task}"
        try:
            resolved = asset_dir(name)
        except FileNotFoundError as exc:
            raise SystemExit(f"--task {task!r}: {exc}") from exc

    asset = DeformableAsset.from_dir(str(resolved))
    robot_cfg = load_robot(str(asset_dir("robot/piper"))) if robot else None
    return SceneConfig(
        asset=asset.obj_path,
        scale=float(asset.scale),
        device=device,
        robot=robot_cfg,
        control_mode=control_mode,
        cloth_params=cloth_params_from_physics(asset.physics),  # asset-driven physics
        normalize_panel_islands=bool(asset.mesh_processing.get("normalize_panel_islands", False)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="bag_drop", help="(P1+) bench task name; default = bag drop scene")
    ap.add_argument("--viewer", default="gl", choices=["gl", "usd", "null"])
    ap.add_argument("--frames", type=int, default=600, help="frame budget for usd/null (gl runs until window closed)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="out.usd")
    ap.add_argument("--robot", action="store_true", help="add the dual-piper robot")
    ap.add_argument("--control", default="kinematic", choices=["kinematic", "pd"], help="robot control mode")
    args = ap.parse_args()

    cfg = build_config(args.task, args.device, robot=args.robot, control_mode=args.control)
    viewer = make_viewer(args.viewer, args.frames, args.output)
    env = DeformableEnv(viewer, cfg)

    frame = 0
    while True:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            break
        env.step()
        env.render()
        if frame % 30 == 0:
            q = env.state_0.particle_q.numpy()
            vmax = float(np.linalg.norm(env.state_0.particle_qd.numpy(), axis=1).max())
            print(
                "frame %4d  t=%.2fs  z[min/mean/max]=%.3f/%.3f/%.3f  vmax=%.3f"
                % (frame, env.sim_time, q[:, 2].min(), q[:, 2].mean(), q[:, 2].max(), vmax),
                flush=True,
            )
        frame += 1
        if args.viewer != "gl" and frame >= args.frames:
            break

    if hasattr(viewer, "close"):
        viewer.close()
    print("done: %d frames, t=%.2fs" % (frame, env.sim_time))


if __name__ == "__main__":
    main()
