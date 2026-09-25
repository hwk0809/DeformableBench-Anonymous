"""Collect episodes for a task into a LeRobot v3.0 dataset.

Orchestrator + worker:

    # collect 50 episodes (real LeRobot v3.0 parquet+mp4) into data/lerobot_dataset/
    uv run python -m deformablebench.bench.collect_data --task flatten_silk_easy \
        --num-episodes 50 --overwrite

    # resume / append
    uv run python -m deformablebench.bench.collect_data --task flatten_silk_easy \
        --num-episodes 50 --resume

Each episode runs in its OWN subprocess (Style3D leaks across env builds, so one
build per process). The orchestrator spawns one worker per episode (first
--overwrite/--resume, the rest --resume), enforces a per-episode timeout, drops
failed episodes, and writes experiment_config.yaml + a success report.

Frequencies + randomization + task params all come from the task yaml (same
TaskSpec as run_task — single source). Collection-only knobs are CLI flags.

Render backend is pluggable: ``--render newton`` (headless ViewerGL, no Kit;
default, for fast pipeline testing) or ``--render kit`` (Kit path-traced via the
shared-memory bridge — the canonical sink for real data).
"""

from __future__ import annotations

import os  # noqa: E402

# All datasets are local (data/lerobot_dataset/); never touch the HF Hub (compaction
# via aggregate_datasets otherwise 401s trying to resolve repo_ids online).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from deformablebench._env import apply_contact_backend_env, ensure_clean_ld_path

ensure_clean_ld_path()
apply_contact_backend_env()  # tri-SDF backend switches from the contact profile, before `import newton`

import argparse  # noqa: E402
import importlib  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


# ----------------------------------------------------------------------------
def _make_bridge(kind: str, width: int, height: int, attach: dict | None = None,
                 rigid_objects: list | None = None, show_window: bool = False):
    if kind == "newton":
        from deformablebench.bench.render.bridge import NewtonOffscreenBridge
        return NewtonOffscreenBridge(width=width, height=height)
    if kind == "kit":
        from deformablebench.bench.render.kit_bridge import ShmKitBridge
        return ShmKitBridge(width=width, height=height, attach=attach,
                            rigid_objects=rigid_objects, show_window=show_window)
    raise ValueError(f"unknown --render backend {kind!r}")


def _dataset_root(lerobot_root: str, repo_id: str) -> str:
    return os.path.join(lerobot_root, repo_id)


def _saved_episode_count(root: str | Path) -> int:
    import json

    info_p = Path(root) / "meta" / "info.json"
    if not info_p.exists():
        return 0
    try:
        return int(json.loads(info_p.read_text()).get("total_episodes", 0))
    except Exception:
        return 0


def _collect_state_path(root: str | Path) -> Path:
    return Path(root) / "meta" / "deformablebench_collect_state.json"


def _read_next_episode_id(root: str | Path) -> int | None:
    import json

    state_p = _collect_state_path(root)
    if not state_p.exists():
        return None
    try:
        value = json.loads(state_p.read_text()).get("next_episode_id")
        return int(value) if value is not None else None
    except Exception:
        return None


def _write_collect_state(root: str | Path, *, next_episode_id: int, last_episode_id: int | None,
                         saved_episodes: int, args) -> None:
    import json
    from datetime import datetime, timezone

    state_p = _collect_state_path(root)
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "task": args.task,
        "repo_id": args.repo_id,
        "next_episode_id": int(next_episode_id),
        "last_episode_id": None if last_episode_id is None else int(last_episode_id),
        "saved_episodes": int(saved_episodes),
        "seed": args.seed,
        "episode_offset_floor": int(args.episode_offset),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_p.write_text(json.dumps(state, indent=2, sort_keys=True))


def _cloth_vert_count(spec) -> int:
    """Cloth particle count == OBJ vertex count, read without building physics
    (orchestrator must size the Kit SHM before any worker runs).

    Uses DeformableAsset's order-preserving OBJ parser — the same count the env
    ends up with. (trimesh disagrees on textured OBJs: it splits vertices per
    uv/normal seam, e.g. 20740 vs the bag's true 20274, and the SHM cloth view
    then rejects the env's array.)"""
    from deformablebench.bench.common.asset import DeformableAsset
    from deformablebench.paths import asset_dir

    return int(DeformableAsset.from_dir(str(asset_dir(spec.cloth))).num_vertices)


# ----------------------------------------------------------------------------
def run_worker(args) -> int:
    """Run ONE episode (args._single_episode_id) and write it to the dataset.

    Returns 0 on success (episode saved), 2 if the task failed (episode dropped)."""
    from loguru import logger

    from deformablebench.bench.common.sim_executor import SimExecutor
    from deformablebench.bench.data.collect import EpisodeCollector, LeRobotWriter
    from deformablebench.bench.task.base_task import BaseTask
    from deformablebench.bench.task.randomization import Randomizer
    from deformablebench.bench.task.spec import TaskSpec
    from deformablebench.sim import DeformableEnv

    ep_id = int(args._single_episode_id)
    spec = TaskSpec.from_name(args.task, args.randomization)
    if args.seed is not None:
        spec.randomization["seed"] = int(args.seed)  # ONE collection seed for all axes
    if args.control:
        spec.control_mode = args.control

    randomizer = Randomizer(spec.randomization, episode_id=ep_id)
    attach = None
    if args.render == "kit" and args._shm_name:   # attach to orchestrator's Kit
        import json
        attach = {"shm_name": args._shm_name, "zmq_endpoint": args._zmq_endpoint,
                  "w2l": json.loads(args._w2l) if args._w2l else None,
                  "robot_root_offset": json.loads(args._robot_root_offset) if args._robot_root_offset else [0, 0, 0],
                  "num_verts": args._num_verts}
    rigid_objects = None
    if args.render == "kit" and spec.rigid_objects:
        from deformablebench.bench.render.kit_bridge import rigid_render_objects
        rigid_objects = rigid_render_objects(spec)
    bridge = _make_bridge(args.render, args.width, args.height, attach=attach,
                          rigid_objects=rigid_objects, show_window=args.kit_window)
    env = DeformableEnv(viewer=None, cfg=spec.scene_config(args.device, randomizer))
    bridge.reset(env)
    # Visual DR (Kit): apply this episode's table-texture variant + lighting before
    # any frames are rendered. No-op unless randomization.texture/light are enabled.
    bridge.start_episode(ep_id,
                         lighting=randomizer.get_isaac_lighting_params(),
                         texture=randomizer.get_isaac_texture_params())

    randomizer.apply_robot_init(env)
    randomizer.apply_rigid_init(env)
    if randomizer.cloth_init_runs_before_settle:
        randomizer.apply_cloth_fold(env)
    else:
        for _ in range(spec.settle_frames):
            env.step()
        randomizer.apply_cloth_fold(env)

    collector = EpisodeCollector(bridge, physics_fps=spec.physics_fps,
                                 control_fps=spec.control_fps,
                                 save_render_state=args.save_render_state)
    collision_trick = dict(spec.collision_trick or {})
    collision_trick.pop("enabled", None)
    executor = SimExecutor(env, collector=collector, physics_freq=spec.physics_fps,
                           control_freq=spec.control_fps,
                           enable_collision_trick=bool(args.enable_collision_trick),
                           collision_trick_config=collision_trick or None)

    mod, cls_name = spec.raw["task_class"].split(":")
    task_module = importlib.import_module(mod)
    cls = getattr(task_module, cls_name)

    # Side-fold collection used to fall back to BaseFoldTask.check_success(), a
    # legacy shoulder-to-hem distance threshold.  Closed-loop evaluation uses
    # TshirtSideFoldEvaluator instead (left/right/forward staged gates), so valid
    # demonstrations could be dropped during collection.  Reuse that exact
    # evaluator here and sample it throughout the scripted trajectory.
    evaluator = None
    evaluator_step = 0
    if cls_name == "TshirtFoldV2Task":
        evaluator_cls = getattr(task_module, "TshirtSideFoldEvaluator", None)
        if evaluator_cls is not None:
            evaluator = evaluator_cls(spec)
            evaluator.reset_eval()
            evaluator.observe(env)  # freeze the garment-local frame before motion
            poll_steps = max(
                1,
                int(round(spec.physics_fps * float(spec.eval.get("check_interval", 2.0)))),
            )

            def _observe_side_fold() -> None:
                nonlocal evaluator_step
                evaluator_step += 1
                if evaluator_step % poll_steps == 0:
                    evaluator.observe(env)

            executor._render_step_callback = _observe_side_fold

    collector.start()
    legacy_success = None
    try:
        if isinstance(cls, type) and issubclass(cls, BaseTask):
            task = cls(env, spec.build_planner(randomizer), collector=None, cfg=spec.raw)
            for step in task.plan_episode(ep_id):
                executor.execute_plan(step)
            legacy_success = bool(task.check_success())
            success = legacy_success
        else:
            from deformablebench.bench.robot.skills.primitive_skills import RobotSkills
            task = cls(spec)
            success = bool(task.run(env, RobotSkills(spec.build_planner(randomizer)), executor))
    finally:
        collector.stop()

    if evaluator is not None:
        # Match eval_sim's terminal release+settle before taking the final
        # measurement, but do it after collector.stop() so training episodes do
        # not gain an extra tail of idle/open-gripper frames.
        settle_ticks = int(getattr(evaluator, "terminal_settle_ticks", 0) or 0)
        control_ratio = max(1, spec.physics_fps // spec.control_fps)
        if settle_ticks:
            q_l = env.left_qpos.copy()
            q_r = env.right_qpos.copy()
            for _ in range(settle_ticks):
                env.set_robot_control(q_l, q_r, 1.0, 1.0)
                for _ in range(control_ratio):
                    env.step()
                    if executor.collision_ctrl is not None:
                        executor.collision_ctrl.update(
                            float(env.left_gripper_val), float(env.right_gripper_val)
                        )
        evaluator.observe(env)
        success = bool(evaluator.check_success())
        metrics = evaluator.eval_metrics()
        logger.info(
            f"[worker ep{ep_id}] staged side-fold check: success={success}, "
            f"flags={metrics.get('flags')}, end_gates_ok={metrics.get('end_gates_ok')}, "
            f"legacy_success={legacy_success}"
        )

    if not success and not args.keep_failed:
        logger.warning(f"[worker ep{ep_id}] task failed -> episode dropped")
        return 2
    if not success:
        logger.warning(f"[worker ep{ep_id}] task failed but --keep-failed -> saving anyway")

    frames, render_state = collector.take_episode()
    if not frames:
        logger.warning(f"[worker ep{ep_id}] no frames recorded -> dropped")
        return 2

    mode = "resume" if args.resume else "create"
    writer = LeRobotWriter(root=_dataset_root(args.lerobot_root, args.repo_id),
                           repo_id=args.repo_id, fps=spec.control_fps,
                           camera_names=bridge.camera_names,
                           image_hw=(args.height, args.width), mode=mode)
    writer.add_episode(frames, spec.raw.get("task_description", spec.name),
                       render_state=render_state, episode_idx=ep_id)
    writer.finalize()
    bridge.close()
    logger.success(f"[worker ep{ep_id}] saved {len(frames)} frames (backend={writer.backend})")
    return 0


# ----------------------------------------------------------------------------
def _write_experiment_config(root: str, args, spec) -> None:
    import yaml
    cfg = {
        "task": args.task, "repo_id": args.repo_id, "num_episodes": args.num_episodes,
        "target_episodes": args.target_episodes,
        "render": args.render, "image_hw": [args.height, args.width],
        "sim": {"physics_fps": spec.physics_fps, "substeps": spec.substeps,
                "control_fps": spec.control_fps},
        "save_render_state": args.save_render_state,
        "enable_collision_trick": args.enable_collision_trick,
        "contact_profile": spec.raw.get("contact_profile"),
        "randomization": spec.randomization,
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(root, "experiment_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _compact_dataset(lerobot_root: str, repo_id: str) -> bool:
    """Merge per-episode parquet/mp4 (one file per worker process) into consolidated
    v3.0 large files (one mp4 per camera). Preserves custom files (render_state,
    experiment_config). Returns True on success."""
    import json
    import shutil

    import packaging.version
    from loguru import logger
    from lerobot.datasets import utils as _lru
    from lerobot.datasets.aggregate import aggregate_datasets

    root = Path(lerobot_root)
    src = root / repo_id
    # Local-only datasets: resolve the codebase version from the local info.json so
    # aggregate/metadata never reaches the HF Hub (get_repo_versions otherwise hits
    # list_repo_refs -> 401 / OfflineModeIsEnabled).
    ver = json.loads((src / "meta" / "info.json").read_text())["codebase_version"].lstrip("v")
    _lru.get_repo_versions = lambda _rid: [packaging.version.parse(ver)]
    tmp_id = repo_id + "__compact_tmp"
    tmp = root / tmp_id
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        # roots/aggr_root are the datasets' OWN dirs (containing meta/), not parents.
        aggregate_datasets([repo_id], tmp_id, roots=[src], aggr_root=tmp)
    except Exception as e:
        logger.warning(f"[compact] aggregate failed ({e}); leaving per-episode files")
        if tmp.exists():
            shutil.rmtree(tmp)
        return False
    # migrate custom (non-lerobot) files from src into the consolidated copy
    for name in ("render_state",):
        s = src / name
        if s.exists():
            shutil.move(str(s), str(tmp / name))
    shutil.rmtree(src)
    shutil.move(str(tmp), str(src))
    logger.success(f"[compact] consolidated {repo_id} into v3.0 large-file format")
    return True


def run_orchestrator(args) -> None:
    from loguru import logger

    from deformablebench.bench.task.spec import TaskSpec

    import json

    spec = TaskSpec.from_name(args.task, args.randomization)
    if args.seed is not None:
        spec.randomization["seed"] = int(args.seed)  # ONE collection seed for all axes
    root = _dataset_root(args.lerobot_root, args.repo_id)
    if args.overwrite and Path(root).exists():
        # wipe here, not in the first worker: with --target-episodes the saved
        # count below and the per-iteration recount would both see the old
        # dataset and exit "already-target" before any worker runs
        import shutil
        shutil.rmtree(root)
        logger.info(f"[overwrite] removed existing dataset at {root}")
    saved_before = _saved_episode_count(root)
    next_ep = int(args.episode_offset)
    if args.resume and not args.overwrite:
        state_ep = _read_next_episode_id(root)
        if state_ep is not None:
            next_ep = max(next_ep, state_ep)
            logger.info(f"[resume] restoring next_episode_id={next_ep} from collect state")
        elif saved_before > 0:
            next_ep = max(next_ep, saved_before)
            logger.info(f"[resume] no collect state; using saved count {saved_before} as next_episode_id")
    target_saved = int(args.target_episodes) if args.target_episodes is not None else None
    if target_saved is not None:
        print(f"== collect: {spec.name} | target={target_saved} saved ep | render={args.render} -> {root} ==")
    else:
        print(f"== collect: {spec.name} | {args.num_episodes} attempt ep | render={args.render} -> {root} ==")
    print(f"   freq {spec.physics_fps}fps x{spec.substeps} = {spec.physics_fps*spec.substeps}Hz "
          f"physics, record {spec.control_fps}Hz | save_render_state={args.save_render_state}")

    success, fail = 0, 0
    base = [sys.executable, "-m", "deformablebench.bench.collect_data",
            "--task", args.task, "--repo-id", args.repo_id,
            "--lerobot-root", args.lerobot_root, "--render", args.render,
            "--width", str(args.width), "--height", str(args.height)]
    if args.device:
        base += ["--device", args.device]
    if args.randomization:
        base += ["--randomization", args.randomization]
    if args.seed is not None:
        base += ["--seed", str(args.seed)]
    if args.control:
        base += ["--control", args.control]
    if args.save_render_state:
        base += ["--save-render-state"]
    if args.keep_failed:
        base += ["--keep-failed"]
    if args.kit_window:
        base += ["--kit-window"]
    # pass the explicit state to the worker so its argparse default can't diverge
    base += ["--collision-trick"] if args.enable_collision_trick else ["--no-collision-trick"]

    if target_saved is not None:
        remaining = max(0, target_saved - saved_before)
        attempt_limit = int(args.max_attempts) if args.max_attempts is not None else max(
            remaining, int(math.ceil(remaining * float(args.attempt_factor)))
        )
        print(f"   existing saved={saved_before}, next_episode_id={next_ep}, max_attempts={attempt_limit}")
        if remaining <= 0 or attempt_limit <= 0:
            print(f"== done: +0 saved, 0 failed, total_saved={saved_before} at {root} (already-target) ==")
            return
    else:
        attempt_limit = int(args.num_episodes)

    # Kit: launch Kit ONCE in the orchestrator; workers attach (avoids ~71s Kit
    # startup per episode). Pass the SHM/ZMQ/transform handoff to each worker.
    kit_owner = None
    if args.render == "kit":
        from deformablebench.bench.common.asset import DeformableAsset
        from deformablebench.bench.render.kit_bridge import ShmKitBridge
        from deformablebench.paths import asset_dir
        from deformablebench.bench.render.kit_bridge import rigid_render_objects
        n_verts = _cloth_vert_count(spec)
        _cloth_asset = DeformableAsset.from_dir(str(asset_dir(spec.cloth)))
        kit_owner, handoff = ShmKitBridge.launch_owner(
            n_verts, width=args.width, height=args.height,
            cloth_usd=_cloth_asset.usd_path,
            cloth_mesh_path=_cloth_asset.usd_prim_path,
            rigid_objects=rigid_render_objects(spec) if spec.rigid_objects else None,
            show_window=args.kit_window)
        base += ["--_shm_name", handoff["shm_name"], "--_zmq_endpoint", handoff["zmq_endpoint"],
                 "--_num_verts", str(handoff["num_verts"]),
                 "--_w2l", json.dumps(handoff["w2l"]),
                 "--_robot_root_offset", json.dumps(handoff["robot_root_offset"])]

    target_limit = target_saved
    last_ep_id = None
    while True:
        if target_limit is not None:
            current_saved = _saved_episode_count(root)
            if current_saved >= target_limit:
                break
        if success + fail >= attempt_limit:
            break

        ep_id = next_ep
        next_ep += 1
        last_ep_id = ep_id
        cmd = base + ["--_single_episode_id", str(ep_id)]
        # First worker honours the user flag (overwrite/resume/fresh); rest resume.
        if success + fail == 0:
            if args.overwrite:
                cmd.append("--overwrite")
            elif args.resume:
                cmd.append("--resume")
        else:
            cmd.append("--resume")

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=args.timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"Episode {ep_id} TIMEOUT ({args.timeout}s) -> dropped")
            rc = 124
        dt = time.time() - t0
        if rc == 0:
            success += 1
            logger.success(f"Episode {ep_id} done ({dt:.1f}s) | {success} ok / {fail} fail")
        else:
            fail += 1
            logger.warning(f"Episode {ep_id} failed (rc={rc}, {dt:.1f}s) | {success} ok / {fail} fail")
        _write_collect_state(root, next_episode_id=next_ep, last_episode_id=ep_id,
                             saved_episodes=_saved_episode_count(root), args=args)

    if kit_owner is not None:
        kit_owner.close()   # SHUTDOWN Kit + unlink SHM (after all workers done)

    if success > 0 and not args.no_compact:
        _compact_dataset(args.lerobot_root, args.repo_id)   # merge per-episode -> one mp4/parquet
    if success > 0:
        _write_experiment_config(root, args, spec)
    if last_ep_id is not None:
        _write_collect_state(root, next_episode_id=next_ep, last_episode_id=last_ep_id,
                             saved_episodes=_saved_episode_count(root), args=args)
    final_saved = _saved_episode_count(root)
    print(f"== done: +{success} saved, {fail} failed, total_saved={final_saved} at {root} "
          f"({'compacted' if (success>0 and not args.no_compact) else 'no-compact'}) ==")


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="grasp_silk_bimanual")
    ap.add_argument("--randomization", default=None,
                    help="difficulty preset in config/randomization/<name>.yaml "
                         "(e.g. clean/hard), deep-merged over the task's randomization block")
    ap.add_argument("--repo-id", default=None, help="default: <task>")
    ap.add_argument("--num-episodes", type=int, default=1)
    ap.add_argument("--target-episodes", type=int, default=None,
                    help="keep collecting until the dataset has at least this many saved episodes")
    ap.add_argument("--lerobot-root", default="data/lerobot_dataset")
    ap.add_argument("--render", default="newton", choices=["newton", "kit"])
    ap.add_argument("--kit-window", action="store_true",
                    help="launch Kit with a visible window (default: headless --no-window)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-render-state", action="store_true",
                    help="dump per-episode npz (cloth verts/link poses/gripper/state) for re-render")
    ap.add_argument("--collision-trick", action=argparse.BooleanOptionalAction,
                    default=True, dest="enable_collision_trick",
                    help="gripper/cloth collision release trick during collection "
                         "(default: on, matches run_task/deformable_bench; "
                         "--no-collision-trick to disable)")
    ap.add_argument("--keep-failed", action="store_true",
                    help="save episodes even if the task check_success() is False (debug/testing)")
    ap.add_argument("--no-compact", action="store_true",
                    help="skip final compaction (default: compact per-episode files into one mp4/parquet)")
    ap.add_argument("--timeout", type=int, default=400, help="per-episode timeout (s)")
    ap.add_argument("--attempt-factor", type=float, default=3.0,
                    help="when --target-episodes is set, max_attempts = remaining * attempt_factor")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="hard cap on worker attempts when using --target-episodes")
    ap.add_argument("--seed", type=int, default=None,
                    help="collection seed overriding randomization.seed for ALL axes "
                         "(geometric + texture + light), combined with episode_id for a "
                         "per-episode RNG. Keep DISJOINT from the eval seed (eval_sim "
                         "defaults to 42) so train/eval initial conditions don't leak "
                         "(convention: data uses 0/43). Default None = use the task yaml's "
                         "randomization.seed (or 0).")
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--control", default=None, choices=["kinematic", "pd"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--_single_episode_id", type=int, default=None,
                    help="internal: run this single episode as a worker")
    # internal Kit-attach handoff (orchestrator launches Kit once; workers attach)
    ap.add_argument("--_shm_name", default=None)
    ap.add_argument("--_zmq_endpoint", default=None)
    ap.add_argument("--_w2l", default=None)
    ap.add_argument("--_robot_root_offset", default=None)
    ap.add_argument("--_num_verts", type=int, default=None)
    args = ap.parse_args()
    if args.repo_id is None:
        args.repo_id = args.task

    if args._single_episode_id is not None:
        sys.exit(run_worker(args))
    run_orchestrator(args)


if __name__ == "__main__":
    main()
