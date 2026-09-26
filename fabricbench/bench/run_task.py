"""Run a benchmark task from its YAML config — the one task driver.

    # watch in a window / headless success-check / record USD
    uv run python -m fabricbench.bench.run_task --task grasp_silk_bimanual --viewer gl
    uv run python -m fabricbench.bench.run_task --task grasp_silk_bimanual --viewer null

    # feed the web viewer: drive the task and publish geometry to shared memory
    uv run python -m fabricbench.bench.run_task --task grasp_silk_bimanual --publish-shm <name>

Loads ``config/task/<task>.yaml``, builds env + planner, settles + folds the cloth,
then plans + executes the episode. One responsibility (drive a task), two output
sinks: a Newton viewer (gl/usd/null) OR a shared-memory geometry stream the web
viewer reads (``--publish-shm``, spawned by ``app/newton_bridge.py``). Because the
IK planner needs pinocchio — which can't load in the already-running Reflex backend
(LD_LIBRARY_PATH) — the web runs this as a separate process that scrubs the path on
launch (below) and re-execs clean.
"""

from __future__ import annotations

from fabricbench._env import apply_contact_backend_env, ensure_clean_ld_path

ensure_clean_ld_path()  # before importing pinocchio (transitively, below)
apply_contact_backend_env()  # tri-SDF backend switches from the contact profile, before `import newton`

import argparse  # noqa: E402
import importlib  # noqa: E402
import time  # noqa: E402

from fabricbench.bench.common.sim_executor import SimExecutor  # noqa: E402
from fabricbench.bench.robot.skills.primitive_skills import RobotSkills  # noqa: E402
from fabricbench.bench.task.base_task import BaseTask  # noqa: E402
from fabricbench.bench.task.spec import TaskSpec  # noqa: E402
from fabricbench.sim import DeformableEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Build + drive (the core; viewer and shm sinks both go through this)
# ---------------------------------------------------------------------------
def build_task_env(spec, *, device=None, episode=0, no_rand=False, viewer=None):
    """Build the env and bring it to the episode start state.

    TaskSpec.scene_config -> DeformableEnv, raise the arms to the start pose,
    settle the dropped cloth, then wrinkle/fold. ``env.render()`` is a no-op when
    ``viewer is None`` (the publisher path), so it is always safe to call. Returns
    ``(env, randomizer)``.
    """

    from fabricbench.bench.task.randomization import Randomizer, fixed_randomization_config

    rand_cfg = fixed_randomization_config(spec.randomization) if no_rand else spec.randomization
    randomizer = Randomizer(rand_cfg, episode_id=episode)
    env = DeformableEnv(viewer=viewer, cfg=spec.scene_config(device, randomizer))

    info = randomizer.apply_robot_init(env)
    if info:
        print(f"  robot_init: {info}")
    info = randomizer.apply_rigid_init(env)
    if info:
        print(f"  rigid_init: {info}")
    # two_point_swing_drop starts at the high spawn. Other modes disturb a cloth
    # that has first settled on the table.
    if randomizer.cloth_init_runs_before_settle:
        info = randomizer.apply_cloth_fold(env)
        if info:
            print(f"  cloth_init: {info}")
    else:
        for _ in range(spec.settle_frames):
            env.step()
            env.render()
        info = randomizer.apply_cloth_fold(env)
        if info:
            print(f"  cloth_init: {info}")
    return env, randomizer


def run_task_episode(spec, env, *, render_step_callback=None) -> bool:
    """Build planner + executor and run the episode once. Returns success.

    ``render_step_callback`` (no args) is invoked by the executor after every
    physics step — gl/usd use it to render, ``--publish-shm`` to push a geometry
    snapshot. Frequencies come from the task yaml ``sim:`` block.
    """

    planner = spec.build_planner()
    collision_trick = dict(spec.collision_trick or {})
    enable_collision_trick = bool(collision_trick.pop("enabled", True))
    executor = SimExecutor(
        env,
        physics_freq=spec.physics_fps,
        control_freq=spec.control_fps,
        enable_collision_trick=enable_collision_trick,
        collision_trick_config=collision_trick or None,
    )
    if render_step_callback is not None:
        executor._render_step_callback = render_step_callback

    mod_name, cls_name = spec.raw["task_class"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)

    if isinstance(cls, type) and issubclass(cls, BaseTask):
        # Standard task: generator yielding ExecutionStep, driven by the executor.
        task = cls(env, planner, collector=None, cfg=spec.raw)
        for step in task.plan_episode(0):
            executor.execute_plan(step)
        return bool(task.check_success())

    # Custom task with a .run(env, skills, executor) (e.g. grasp_silk_bimanual).
    task = cls(spec)
    return bool(task.run(env, RobotSkills(planner), executor))


# ---------------------------------------------------------------------------
# Shared-memory publisher (the web viewer sink)
# ---------------------------------------------------------------------------
class _ShmPublisher:
    """Decimates physics steps to the control rate, optionally paces to real time,
    and writes the env's geometry into the shared snapshot the web viewer reads."""

    def __init__(self, shm, physics_fps: float, control_fps: float, realtime: bool = True):
        self.shm = shm
        self.steps_per_frame = max(1, round(physics_fps / max(control_fps, 1e-6)))
        self.control_fps = float(control_fps)
        self.realtime = realtime
        self._n = 0
        self._frames = 0
        self._wall0: float | None = None

    def write(self, env) -> None:
        body_q = env.state_0.body_q.numpy() if self.shm.n_bodies else None
        self.shm.write(env.get_sim_vertices(), body_q)

    def tick(self, env) -> None:
        """Executor render-step hook (once per physics step)."""
        self._n += 1
        if self._n % self.steps_per_frame:
            return
        self.write(env)
        if self.realtime:
            now = time.perf_counter()
            self._frames += 1
            if self._wall0 is None:
                self._wall0 = now
                return
            delay = self._wall0 + self._frames / self.control_fps - now
            if delay > 0:
                time.sleep(min(delay, 0.1))


def _reset_for_loop(env) -> None:
    """Reset cloth to its initial drop + arms to home, for a fresh looped episode."""
    st = env.state_0
    snaps = getattr(env, "initial_cloth_states", None)
    if snaps:
        st.particle_q.assign(snaps[0])
    try:
        st.particle_qd.zero_()
    except Exception:
        pass
    if getattr(env, "has_robot", False):
        env.set_robot_control(env.robot.home_left, env.robot.home_right, 1.0, 1.0)


def _start_parent_watchdog() -> None:
    """Exit if the parent (the web backend that spawned us) dies.

    The producer is spawned with ``start_new_session=True``, so if the Reflex
    backend crashes/restarts without calling close(), this process would otherwise
    be reparented to init and run forever (orphan zombie). Watch ``getppid()`` and
    self-terminate the moment it changes."""
    import os
    import threading
    import time

    ppid0 = os.getppid()

    def _watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != ppid0:   # parent gone -> reparented (usually to init)
                os._exit(0)

    threading.Thread(target=_watch, daemon=True).start()


def _run_publish(spec, args) -> None:
    """Drive the task and stream geometry to ``args.publish_shm`` until killed.

    With ``--loop`` (the web viewer uses it) the episode repeats forever so the
    motion is continuous and never missed; otherwise it runs once and holds the
    final frame.
    """
    from loguru import logger

    from fabricbench.bench.render.geom_shm import GeomSnapshotShm

    _start_parent_watchdog()   # die with the web backend — never orphan
    env, _ = build_task_env(spec, device=args.device, episode=args.episode, no_rand=args.no_rand)
    shm = GeomSnapshotShm.attach(args.publish_shm)
    pub = _ShmPublisher(shm, spec.physics_fps, spec.control_fps, realtime=not args.no_realtime)
    logger.success(f"[run_task] publishing {spec.name} -> {args.publish_shm} "
                   f"({shm.n_verts} verts, {shm.n_bodies} bodies){' [loop]' if args.loop else ''}")
    pub.write(env)  # show the settled start pose immediately
    try:
        while True:
            try:
                success = run_task_episode(spec, env, render_step_callback=lambda: pub.tick(env))
                logger.info(f"[run_task] {spec.name} episode done success={success}")
            except Exception as e:
                logger.warning(f"[run_task] episode error (continuing): {e}")
            if not args.loop:
                logger.info("[run_task] holding final frame (no --loop)")
                while True:
                    pub.write(env)
                    time.sleep(0.2)
            _reset_for_loop(env)                       # fresh cloth + home arms
            for _ in range(min(spec.settle_frames, 60)):  # brief re-settle, keep streaming
                env.step()
                pub.tick(env)
    finally:
        shm.close()


# ---------------------------------------------------------------------------
def make_viewer(kind: str, frames: int, output: str):
    import newton.viewer as V

    if kind == "gl":
        return V.ViewerGL()
    if kind == "usd":
        return V.ViewerUSD(output_path=output, num_frames=frames)
    return V.ViewerNull(num_frames=frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="grasp_silk_bimanual")
    ap.add_argument("--viewer", default="null", choices=["gl", "usd", "null"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="task.usd")
    ap.add_argument("--control", default=None, choices=["kinematic", "pd"], help="override control_mode")
    ap.add_argument("--episode", type=int, default=0, help="episode id (seeds randomization)")
    ap.add_argument("--no-rand", action="store_true", help="disable task randomization")
    ap.add_argument("--publish-shm", default=None,
                    help="stream geometry to this shared-memory segment (web viewer sink)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="with --publish-shm: run free instead of pacing to real time")
    ap.add_argument("--loop", action="store_true",
                    help="with --publish-shm: repeat the episode forever (web viewer uses this)")
    args = ap.parse_args()

    spec = TaskSpec.from_name(args.task)
    if args.control:
        spec.control_mode = args.control
    print(f"== task: {spec.name} | cloth={spec.cloth} robot={spec.robot} "
          f"planner={spec.planner} control={spec.control_mode} ==")

    if args.publish_shm:
        _run_publish(spec, args)
        return

    viewer = make_viewer(args.viewer, spec.settle_frames + 2000, args.output)
    env, _ = build_task_env(
        spec, device=args.device, episode=args.episode, no_rand=args.no_rand, viewer=viewer)
    success = run_task_episode(
        spec, env, render_step_callback=(env.render if viewer is not None else None))

    print(f"\n== {spec.name}: success={success} ==")
    if hasattr(viewer, "close"):
        viewer.close()


if __name__ == "__main__":
    main()
