"""Closed-loop policy evaluation for a task (R5 eval).

Port of deformable_bench's ``scripts/eval_isaac_render.py`` onto deformablebench's
newton + Kit stack. The *generic* seams are reused verbatim — the ``PolicyBase``
adapter contract (``policy/<name>.py``), the standardized ``obs_dict`` schema
(``observation.state`` + ``observation.images.<cam>``), the ``SuccessChecker``,
and the ``eval_results.json`` metrics format. Only the sim/render loop is
deformablebench-native: it clones ``collect_data``'s orchestrator+worker + Kit-attach
(one Kit launch, one subprocess per episode — Style3D leaks across env builds),
and drives the robot from ``policy.predict(obs)`` instead of a planner.

    # eval a checkpoint, 10 episodes, Kit render
    .venv/bin/python -m deformablebench.bench.eval_sim --task grasp_silk_bimanual \
        --policy_type lerobot --checkpoint <.../checkpoints/030000/pretrained_model> \
        --num_episodes 10 --render kit --output_dir data/eval_out/<name>

Writes ``<output_dir>/eval_results.json`` plus ``episodes/episode_XXXX.json``.
With ``--save_video``, MP4s are stored under ``videos/<camera>/episode_XXXX.mp4``.
The summary is rewritten incrementally after every episode (crash/resume-safe).
"""

from __future__ import annotations

import os  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from deformablebench._env import apply_contact_backend_env, ensure_clean_ld_path

ensure_clean_ld_path()
apply_contact_backend_env()  # tri-SDF backend switches from the contact profile, before `import newton`

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# reuse collect_data's Kit-launch + bridge factory (single source of truth)
from deformablebench.bench.collect_data import _cloth_vert_count, _make_bridge  # noqa: E402


# ----------------------------------------------------------------------------
def _selected_video_cameras(requested: str | None, available: list[str]) -> list[str]:
    if not requested:
        return [available[0]] if available else []
    wanted = [c.strip() for c in requested.split(",") if c.strip()]
    if wanted == ["all"]:
        return list(available)
    return [c for c in wanted if c in available]


def _write_mp4(frames: list, path: Path, fps: int | float) -> None:
    import cv2
    import numpy as np

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _episode_json_path(output_dir: str | Path, ep_id: int) -> Path:
    return Path(output_dir) / "episodes" / f"episode_{ep_id:04d}.json"


def _legacy_episode_json_path(output_dir: str | Path, ep_id: int) -> Path:
    return Path(output_dir) / f"episode_{ep_id:04d}.json"


def _episode_done(output_dir: str | Path, ep_id: int) -> bool:
    return _episode_json_path(output_dir, ep_id).exists() or _legacy_episode_json_path(output_dir, ep_id).exists()


def _episode_json_files(output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    by_id: dict[int, Path] = {}
    for p in sorted(out.glob("episodes/episode_*.json")):
        try:
            ep_id = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        by_id[ep_id] = p
    for p in sorted(out.glob("episode_*.json")):
        try:
            ep_id = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        by_id.setdefault(ep_id, p)
    return [by_id[k] for k in sorted(by_id)]


def _video_path(output_dir: str | Path, camera: str, ep_id: int) -> Path:
    return Path(output_dir) / "videos" / camera / f"episode_{ep_id:04d}.mp4"


def _clamp_ground_q(kin, q_arm, prev_q, min_ground_z):
    """Ground-penetration clamp for one arm's policy joint target.

    Mirrors deformable_bench's real-robot FK+IK ground clamp AND the collector's
    demo ``_grasp_pose`` z = max(., 0) floor: FK the TCP from the commanded q; if
    its world z dips below ``min_ground_z``, re-solve IK with the TCP z raised to
    the floor (keeping xy + orientation). IK failure -> hold the previous q so a
    bad target can't drive the gripper through the table/cloth.
    """
    import numpy as _np
    pose = kin.solve_fk(_np.asarray(q_arm, dtype=float))
    if float(pose[2]) >= float(min_ground_z):
        return q_arm
    target = pose.copy()
    target[2] = float(min_ground_z)
    q_corr = kin.solve_ik(target, seed_joints=_np.asarray(q_arm, dtype=float))
    if q_corr is not None:
        return _np.asarray(q_corr, dtype=_np.float32)
    return prev_q if prev_q is not None else q_arm


def _set_cloth_reference(spec, reference: str) -> None:
    spec.cloth = reference.removeprefix("assets/")
    spec.raw["cloth"] = spec.cloth


def _contact_provenance(spec) -> dict:
    profile = spec.raw.get("contact_profile")
    result = {
        "control_mode": spec.control_mode,
        "contact_profile": profile,
        "base_task": spec.raw.get("base"),
    }
    if profile:
        path = Path(__file__).parent / "config" / "contact" / f"{profile}.yaml"
        if path.is_file():
            result["contact_profile_digest"] = (
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            )
    return result


def run_worker(args) -> int:
    """Run ONE eval episode with the policy; write episode_<id>.json.

    Returns 0 if the task succeeded, 2 otherwise (so the orchestrator can count)."""
    from loguru import logger

    from deformablebench.bench.common.success_checker import SuccessChecker
    from deformablebench.bench.data.collect import state_14d
    from deformablebench.bench.policy._base import load_policy
    from deformablebench.bench.task.randomization import Randomizer
    from deformablebench.bench.task.spec import TaskSpec
    from deformablebench.sim import DeformableEnv

    ep_id = int(args._single_episode_id)
    spec = TaskSpec.from_name(args.task, args.randomization)
    if args.cloth:
        _set_cloth_reference(spec, args.cloth)
    apply_eval_seed(spec, args.eval_seed)
    if args.control:
        spec.control_mode = args.control
    randomizer = Randomizer(spec.randomization, episode_id=ep_id)

    # --- Kit attach (same handoff as collect_data's worker) ---
    attach = None
    if args.render == "kit" and args._shm_name:
        attach = {"shm_name": args._shm_name, "zmq_endpoint": args._zmq_endpoint,
                  "w2l": json.loads(args._w2l) if args._w2l else None,
                  "robot_root_offset": json.loads(args._robot_root_offset) if args._robot_root_offset else [0, 0, 0],
                  "num_verts": args._num_verts}
    rigid_objects = None
    if args.render == "kit" and spec.rigid_objects:
        from deformablebench.bench.render.kit_bridge import rigid_render_objects
        rigid_objects = rigid_render_objects(spec)
    bridge = _make_bridge(args.render, args.width, args.height, attach=attach,
                          rigid_objects=rigid_objects)
    # Live Newton viewer for INSPECTION (--viewer gl): an in-process OpenGL window
    # that shows the sim as the policy drives it. Independent of --render (Kit, a
    # separate process producing the policy's obs). Default null == no-op headless,
    # so batch/benchmark eval is unaffected. env.render() below pumps its frames.
    # (--viewer kit is NOT handled here — it's the orchestrator's native Kit
    # viewport window, launched once via ShmKitBridge(show_window=True).)
    live_viewer = None
    if getattr(args, "viewer", "null") in ("gl", "both"):
        import newton.viewer as _V
        live_viewer = _V.ViewerGL()
    env = DeformableEnv(viewer=live_viewer, cfg=spec.scene_config(args.device, randomizer))
    bridge.reset(env)
    # Visual DR (Kit): apply this episode's table-texture variant + lighting before
    # any frames are rendered. No-op unless randomization.texture/light are enabled.
    # Same wiring as collect_data so train/eval share one DR pipeline.
    bridge.start_episode(ep_id,
                         lighting=randomizer.get_isaac_lighting_params(),
                         texture=randomizer.get_isaac_texture_params())

    # --- reset / randomize / settle (identical to collect) ---
    randomizer.apply_robot_init(env)
    randomizer.apply_rigid_init(env)
    if randomizer.cloth_init_runs_before_settle:
        randomizer.apply_cloth_fold(env)
    else:
        for _ in range(spec.settle_frames):
            env.step()
        randomizer.apply_cloth_fold(env)

    # --- policy ---
    policy = load_policy(args.policy_type, args)
    policy.reset()
    task_desc = ""
    if getattr(policy, "needs_language", False):
        task_desc = spec.raw.get("task_description", spec.name)

    # Evaluation: grasp-style tasks own their check_success/get_score (RoboTwin
    # style) — the driver only calls them. Other tasks fall back to the
    # config-driven SuccessChecker (first-pass SR, no score).
    # Resolve evaluators defensively: a sibling task's evaluator missing (e.g. an
    # older checkout on an eval box) must not abort an unrelated task's run.
    import importlib
    _EVAL_SPECS = {
        "grasp_silk_bimanual": ("grasp_silk_bimanual", "SilkGraspBimanualTask"),
        "flatten_silk_easy": ("task_flatten_silk", "FlattenSilkEvaluator"),
        "flatten_silk_hard": ("task_flatten_silk", "FlattenSilkEvaluator"),
        "flatten_tshirt": ("task_flatten_tshirt", "FlattenTshirtEvaluator"),
        "flatten_tshirt_two_point": ("task_flatten_tshirt", "FlattenTshirtEvaluator"),
        "grasp_bag_strap": ("task_grasp_bag_strap", "BagStrapEvaluator"),
        "fold_tshirt_side": ("task_side_fold_tshirt_v2", "TshirtSideFoldEvaluator"),
        "fold_tshirt_side_no_drag_edge": (
            "task_side_fold_tshirt_v2", "TshirtSideFoldEvaluator"
        ),
        "fold_tshirt_side_v3": ("task_side_fold_tshirt_v2", "TshirtSideFoldEvaluator"),
        "fold_tshirt_side_right_nearest_hem": (
            "task_side_fold_tshirt_v2", "TshirtSideFoldEvaluator"
        ),
        "pack_snickers": ("task_pack_snickers", "PackSnickersEvaluator"),
    }
    EVALUATORS = {}
    for _task, (_mod, _cls) in _EVAL_SPECS.items():
        try:
            EVALUATORS[_task] = getattr(
                importlib.import_module(f"deformablebench.bench.task.{_mod}"), _cls)
        except (ImportError, AttributeError):
            pass
    # A `base:` variant (e.g. flatten_tshirt_contact) is the same task with the
    # contact stack switched on -- it evaluates with its base task's evaluator.
    EvCls = EVALUATORS.get(args.task) or EVALUATORS.get(spec.raw.get("base") or "")
    evaluator = EvCls(spec) if EvCls else None
    if evaluator is not None:
        evaluator.reset_eval()
    checker = None if evaluator is not None else SuccessChecker(spec.success_check_cfg, env)
    cams = bridge.camera_names
    video_cams = _selected_video_cameras(args.video_camera, cams) if args.save_video else []
    video_frames = {c: [] for c in video_cams}
    control_ratio = max(1, spec.physics_fps // spec.control_fps)
    max_ticks = int(args.max_steps) if args.max_steps else int(args.sim_duration * spec.control_fps)
    check_every = max(1, int(round(args.check_interval * spec.control_fps)))

    # Some evaluators (flatten) score the SETTLED end-state, so they must run the
    # full episode (no mid-episode early-exit) and be measured after a terminal
    # gripper release + settle. Default True keeps grasp's early-exit behaviour.
    early_exit = bool(getattr(evaluator, "early_exit", True)) if evaluator is not None else True

    # Gripper/cloth collision trick — MUST match data collection so the policy
    # sees the same gripper<->cloth contact it was trained on. collect_data.py and
    # run_task.py enable it from spec.collision_trick (default enabled=True); eval
    # drives env directly (no SimExecutor), so it must replicate it here or learned
    # grasps won't engage the cloth the same way. Update() is called after EVERY
    # physics step, exactly like SimExecutor.
    _ct = dict(spec.collision_trick or {})
    _ct_enabled = bool(_ct.pop("enabled", True))
    collision_ctrl = env.get_gripper_collision_controller(_ct or None) if _ct_enabled else None
    logger.info(f"[eval] gripper collision trick "
                f"{'ENABLED (cfg=' + str(_ct or 'default') + ')' if collision_ctrl else 'DISABLED'}")

    # Ground-penetration clamp (--clamp_ground): FK+IK correct the policy's joint
    # targets so the gripper TCP never dips below --min_ground_z. Built once here
    # (reuses the sim Model), applied per control step below.
    kin_l = kin_r = None
    if getattr(args, "clamp_ground", False) and env.has_robot:
        from deformablebench.bench.robot.kinematics.newton_kinematics import NewtonKinematics
        kin_l = NewtonKinematics.from_deformable_env(env, "left")
        kin_r = NewtonKinematics.from_deformable_env(env, "right")
        logger.info(f"[eval] ground clamp ENABLED (min_ground_z={args.min_ground_z})")
    prev_q_l = prev_q_r = None

    # Per-frame arm-target interpolation (default ON; --no_interp_action restores
    # the original "set once + control_ratio empty steps" path verbatim).
    import numpy as _np_interp
    interp_action = env.has_robot and not bool(getattr(args, "no_interp_action", False))
    logger.info(f"[eval] action interpolation "
                f"{'ENABLED (' + str(control_ratio) + ' frames per control step)' if interp_action else 'DISABLED'}")

    success, max_z, mean_z, prev = False, -1e9, 0.0, None
    action = None
    t0 = time.time()
    t = 0
    for t in range(max_ticks):
        # assemble standardized obs (render 3 cams + 14-dim state)
        imgs = bridge.render(env)
        for c, frames in video_frames.items():
            if c in imgs:
                frames.append(imgs[c].copy())
        obs = {"observation.state": state_14d(env)}
        if getattr(policy, "needs_images", True):
            for c in cams:
                obs[f"observation.images.{c}"] = imgs[c]
        action = policy.predict(obs, task_desc).reshape(14)
        if args.ema_alpha and prev is not None:
            action = args.ema_alpha * action + (1.0 - args.ema_alpha) * prev
        prev = action

        q_l, g_l = action[:6], float(action[6])
        q_r, g_r = action[7:13], float(action[13])
        if args.clip_gripper:
            g_l = args.grasp_close_val if g_l < 0.5 else g_l
            g_r = args.grasp_close_val if g_r < 0.5 else g_r
        if kin_l is not None:
            q_l = _clamp_ground_q(kin_l, q_l, prev_q_l, args.min_ground_z)
            q_r = _clamp_ground_q(kin_r, q_r, prev_q_r, args.min_ground_z)
            prev_q_l, prev_q_r = q_l, q_r
        if interp_action:
            # Spread the arm action linearly over the frames of the control step:
            # one target per env.step, exactly what SimExecutor feeds the env during
            # data collection. q0 = the target currently commanded (the reset start
            # pose on the first step, the previous action afterwards); the gripper
            # command stays a step. Without this the PD arm gets a control_ratio-frame
            # target jump every control step (data/eval_out/TUP3G/TUP3G_GRASP.md).
            _jq0 = _np_interp.asarray(env._joint_q, dtype=float)
            _q0_l, _q0_r = _jq0[env.idx_l_arm].copy(), _jq0[env.idx_r_arm].copy()
            _q1_l = _np_interp.asarray(q_l, dtype=float)
            _q1_r = _np_interp.asarray(q_r, dtype=float)
        else:
            env.set_robot_control(q_l, q_r, g_l, g_r)
        for _i in range(control_ratio):
            if interp_action:
                _a = (_i + 1) / control_ratio
                env.step(_q0_l + _a * (_q1_l - _q0_l), _q0_r + _a * (_q1_r - _q0_r), g_l, g_r)
            else:
                env.step()
            if collision_ctrl is not None:
                collision_ctrl.update(float(env.left_gripper_val), float(env.right_gripper_val))
            if live_viewer is not None:
                env.render()

        cz = env.get_cloth_position()[:, 2]
        max_z, mean_z = max(max_z, float(cz.max())), float(cz.mean())
        if (t + 1) % check_every == 0:
            if evaluator is not None:
                evaluator.observe(env)          # records the per-poll coverage series
                if early_exit and evaluator.check_success():
                    success = True
                    break
            elif checker.check():               # other tasks: original first-pass
                success = True
                break

    # Terminal release + settle (flatten-style): open both grippers, hold the
    # last arm pose, let the cloth fall & settle, capturing it into the same
    # video, then take the final on-ground measurement used for scoring.
    settle_ticks = int(getattr(evaluator, "terminal_settle_ticks", 0) or 0) if evaluator is not None else 0
    if settle_ticks and action is not None:
        q_l, q_r = action[:6], action[7:13]
        for _ in range(settle_ticks):
            env.set_robot_control(q_l, q_r, 1.0, 1.0)   # 1.0 == grippers fully open
            for _ in range(control_ratio):
                env.step()
                if collision_ctrl is not None:
                    collision_ctrl.update(float(env.left_gripper_val), float(env.right_gripper_val))
                if live_viewer is not None:
                    env.render()
            imgs = bridge.render(env)
            for c, frames in video_frames.items():
                if c in imgs:
                    frames.append(imgs[c].copy())
        cz = env.get_cloth_position()[:, 2]
        max_z, mean_z = max(max_z, float(cz.max())), float(cz.mean())
        evaluator.observe(env)                  # settled state == last series record

    if evaluator is None and not success:
        success = bool(checker.check())
    # Score + raw metrics come from the task's own evaluator (recomposable offline).
    if evaluator is not None:
        if not success:
            success = bool(evaluator.check_success())
        score, metrics = evaluator.get_score(), evaluator.eval_metrics()
    else:
        score, metrics = None, None

    result = {"episode_id": ep_id, "success": bool(success), "steps": int(t + 1),
              "max_steps": max_ticks, "max_z": round(max_z, 4), "mean_z": round(mean_z, 4),
              "score": score, "metrics": metrics,
              "interp_action": bool(interp_action),
              "duration_sec": round(time.time() - t0, 1)}
    result.update(_contact_provenance(spec))
    if args._asset_entry_json:
        result["asset"] = json.loads(args._asset_entry_json)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if video_frames:
        video_paths = {}
        for cam, frames in video_frames.items():
            path = _video_path(args.output_dir, cam, ep_id)
            _write_mp4(frames, path, fps=spec.control_fps)
            video_paths[cam] = str(path)
        if video_paths:
            result["video_path"] = next(iter(video_paths.values()))
            result["video_paths"] = video_paths
    ep_json = _episode_json_path(args.output_dir, ep_id)
    ep_json.parent.mkdir(parents=True, exist_ok=True)
    ep_json.write_text(json.dumps(result))
    # Raw particle dump (init + final + canonical) for full offline re-scoring.
    if evaluator is not None and hasattr(evaluator, "eval_raw"):
        import numpy as _np
        raw = evaluator.eval_raw()
        if raw.get("final") is not None:
            raw_dir = Path(args.output_dir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            _np.savez_compressed(
                raw_dir / f"episode_{ep_id:04d}.npz",
                **{k: _np.asarray(v) for k, v in raw.items() if v is not None})
    # Inspection mode: freeze the final frame in the live window so it can be
    # eyeballed, until the user closes it (blocks the per-episode subprocess —
    # intended for --num_episodes 1). No-op headless.
    if live_viewer is not None:
        logger.info(f"[eval ep{ep_id}] viewer holding final frame "
                    f"(success={success} score={score}); close the window to continue.")
        try:
            while getattr(live_viewer, "is_running", lambda: False)():
                env.render()
        except KeyboardInterrupt:
            pass
        if hasattr(live_viewer, "close"):
            live_viewer.close()
    bridge.close()
    logger.info(f"[eval ep{ep_id}] success={success} score={score} steps={t + 1} max_z={max_z:.3f}")
    return 0 if success else 2


# ----------------------------------------------------------------------------
def _aggregate(args, spec) -> dict:
    eps = []
    for p in _episode_json_files(args.output_dir):
        try:
            eps.append(json.loads(p.read_text()))
        except Exception:
            pass
    eps = [e for e in eps if args.start_ep <= e["episode_id"] < args.start_ep + args.num_episodes]
    n = len(eps)
    k = sum(1 for e in eps if e["success"])
    scored = [e["score"] for e in eps if e.get("score") is not None]
    mean_score = round(sum(scored) / len(scored), 1) if scored else None
    result = {
        "policy_type": args.policy_type, "checkpoint": args.checkpoint, "task": args.task,
        "randomization": args.randomization, "eval_seed": args.eval_seed,
        "num_episodes": n, "success_count": k,
        "success_rate": round(100.0 * k / n, 1) if n else 0.0,
        "mean_score": mean_score,
        "success_check": spec.success_check_cfg,
        "max_steps": int(args.max_steps) if args.max_steps else int(args.sim_duration * spec.control_fps),
        "episodes": sorted(eps, key=lambda e: e["episode_id"]),
    }
    result.update(_contact_provenance(spec))
    schedule_path = Path(args.output_dir) / "asset_schedule.json"
    if schedule_path.is_file():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        result["asset_schedule"] = {
            key: schedule[key]
            for key in ("pool_id", "pool_digest", "asset_seed", "schedule_digest")
        }
        result["asset_schedule"]["path"] = str(schedule_path)
    return result


def _write_summary(args, spec) -> dict:
    summary = _aggregate(args, spec)
    out = Path(args.output_dir) / "eval_results.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out)
    return summary


# Out-of-process policy-server launchers live in policy/servers.py so the
# real-robot entry point can reuse them without importing the sim stack.
from deformablebench.bench.policy.servers import (  # noqa: E402
    _attach,
    _kill_server,
    _launch_fastwam_server,
    _launch_gr00t_server,
    _launch_lingbot_server,
    _spawn_server,
)


def _append_policy_worker_args(base: list[str], args) -> None:
    if args.policy_type == "lerobot_remote":
        base += ["--server_host", args.server_host, "--server_port", str(args.server_port),
                 "--server_task", args.server_task]
    elif args.policy_type == "gr00t":
        base += ["--gr00t_host", args.gr00t_host, "--gr00t_port", str(args.gr00t_port),
                 "--gr00t_task", args.gr00t_task,
                 "--gr00t_exec_horizon", str(args.gr00t_exec_horizon)]
    elif args.policy_type == "lingbot":
        base += ["--lingbot_host", args.lingbot_host, "--lingbot_port", str(args.lingbot_port),
                 "--lingbot_video_guidance_scale", str(args.lingbot_video_guidance_scale),
                 "--lingbot_action_guidance_scale", str(args.lingbot_action_guidance_scale)]
        if getattr(args, "imagined_out", None):
            base += ["--imagined_out", args.imagined_out]
    elif args.policy_type == "fastwam":
        base += ["--fastwam_host", args.fastwam_host, "--fastwam_port", str(args.fastwam_port),
                 "--fastwam_task", args.fastwam_task]


def apply_eval_seed(spec, eval_seed) -> None:
    """Override the task's randomization seed with the held-out eval seed.

    The task yaml's `randomization.seed` is what data collection used (collect_data
    falls back to it when --seed is absent), so an eval seed equal to it replays the
    training initial conditions verbatim.  That leaked silently for four tasks until
    2026-07-26 -- refuse to run instead of producing contaminated numbers.
    """
    if eval_seed is None:
        return
    collect_seed = spec.randomization.get("seed")
    if collect_seed is not None and int(eval_seed) == int(collect_seed):
        raise SystemExit(
            f"eval_seed {eval_seed} == task '{spec.name}' randomization.seed "
            f"{collect_seed} (the collection seed): eval would replay the training "
            f"initial conditions. Pick a disjoint eval seed (convention: data 0/42/43, "
            f"eval 41)."
        )
    spec.randomization["seed"] = int(eval_seed)  # ONE held-out seed for all axes


def _launch_kit_owner(args, spec):
    from deformablebench.bench.common.asset import DeformableAsset
    from deformablebench.bench.render.kit_bridge import ShmKitBridge, rigid_render_objects
    from deformablebench.paths import asset_dir

    n_verts = _cloth_vert_count(spec)
    cloth_path = Path(spec.cloth)
    if not cloth_path.is_dir():
        cloth_path = asset_dir(spec.cloth)
    cloth = DeformableAsset.from_dir(str(cloth_path))
    owner, handoff = ShmKitBridge.launch_owner(
        n_verts,
        width=args.width,
        height=args.height,
        cloth_usd=cloth.usd_path,
        cloth_mesh_path=cloth.usd_prim_path,
        rigid_objects=rigid_render_objects(spec) if spec.rigid_objects else None,
        show_window=(args.viewer in ("kit", "both")),
    )
    worker_args = [
        "--_shm_name", handoff["shm_name"],
        "--_zmq_endpoint", handoff["zmq_endpoint"],
        "--_num_verts", str(handoff["num_verts"]),
        "--_w2l", json.dumps(handoff["w2l"]),
        "--_robot_root_offset", json.dumps(handoff["robot_root_offset"]),
    ]
    return owner, worker_args


def run_orchestrator(args) -> None:
    from loguru import logger

    from deformablebench.bench.task.spec import TaskSpec

    if args.cloth and args.asset_pool:
        raise SystemExit("--cloth and --asset_pool are mutually exclusive")

    spec = TaskSpec.from_name(args.task, args.randomization)
    if args.cloth:
        _set_cloth_reference(spec, args.cloth)

    asset_pool = asset_schedule = asset_by_episode = None
    if args.asset_pool:
        from deformablebench.bench.asset_schedule import (
            load_or_create_asset_schedule,
            schedule_by_episode,
        )

        pool_task = spec.raw.get("base") or args.task
        asset_pool, asset_schedule = load_or_create_asset_schedule(
            args.asset_pool,
            args.output_dir,
            expected_task=pool_task,
            asset_seed=args.asset_seed,
            start_ep=args.start_ep,
            num_episodes=args.num_episodes,
            resume=args.resume,
        )
        asset_by_episode = schedule_by_episode(asset_schedule)

    apply_eval_seed(spec, args.eval_seed)
    if args.viewer in ("kit", "both") and args.render != "kit":
        raise SystemExit(f"--viewer {args.viewer} requires --render kit (the Kit viewport IS "
                         "the Kit renderer's window). Use --viewer gl for the Newton debug viewer.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"== eval: {spec.name} | {args.policy_type} | {args.num_episodes} ep "
          f"| render={args.render} | eval_seed={args.eval_seed} -> {args.output_dir} ==")
    print(f"   checkpoint: {args.checkpoint}")
    print(f"   control/contact: {spec.control_mode} / "
          f"{spec.raw.get('contact_profile') or 'default'}")
    if asset_schedule:
        print(f"   asset pool: {asset_pool['pool_id']} | seed={args.asset_seed} "
              f"| schedule={Path(args.output_dir) / 'asset_schedule.json'}")

    base = [sys.executable, "-m", "deformablebench.bench.eval_sim",
            "--task", args.task, "--policy_type", args.policy_type,
            "--render", args.render, "--viewer", args.viewer,
            "--output_dir", args.output_dir,
            "--width", str(args.width), "--height", str(args.height),
            "--sim_duration", str(args.sim_duration),
            "--check_interval", str(args.check_interval),
            "--ema_alpha", str(args.ema_alpha), "--grasp_close_val", str(args.grasp_close_val)]
    if args.cloth:
        base += ["--cloth", args.cloth]
    if args.checkpoint is not None:
        base += ["--checkpoint", args.checkpoint]
    if args.max_steps:
        base += ["--max_steps", str(args.max_steps)]
    if args.randomization:
        base += ["--randomization", args.randomization]
    if args.eval_seed is not None:
        base += ["--eval_seed", str(args.eval_seed)]
    if args.device:
        base += ["--device", args.device]
    if args.control:
        base += ["--control", args.control]
    if args.clip_gripper:
        base += ["--clip_gripper"]
    if getattr(args, "no_interp_action", False):
        base += ["--no_interp_action"]
    if args.clamp_ground:
        base += ["--clamp_ground", "--min_ground_z", str(args.min_ground_z)]
    if getattr(args, "n_action_steps", None) is not None:
        base += ["--n_action_steps", str(args.n_action_steps)]
    if args.save_video:
        base += ["--save_video"]
        if args.video_camera:
            base += ["--video_camera", args.video_camera]

    # Decoupled policy server (lerobot_remote): load the checkpoint ONCE in its own
    # env (patched transformers etc.) and serve over ZMQ; workers connect as clients.
    server_proc = None
    if args.policy_type == "lerobot_remote":
        if args.server_python:
            serve_py = str(Path(__file__).parent / "policy" / "serve_lerobot.py")
            senv = dict(os.environ)
            scmd = [args.server_python, serve_py, "--checkpoint", args.checkpoint,
                    "--port", str(args.server_port), "--device", args.device or "cuda"]
            if getattr(args, "n_action_steps", None) is not None:
                scmd += ["--n_action_steps", str(args.n_action_steps)]
            server_proc = _spawn_server(
                "lerobot_remote", scmd, host="127.0.0.1", port=args.server_port,
                wait_s=float(getattr(args, "server_wait_s", 420.0)), logger=logger, env=senv)
        else:
            _attach("lerobot_remote", args.server_port, logger)
    elif args.policy_type == "lingbot":
        server_proc = _launch_lingbot_server(args, logger)
    elif args.policy_type == "gr00t":
        server_proc = _launch_gr00t_server(args, logger)
    elif args.policy_type == "fastwam":
        server_proc = _launch_fastwam_server(args, logger)
    _append_policy_worker_args(base, args)

    # A fixed cloth keeps one Kit owner. Pool mode relaunches Kit per episode
    # because the cloth USD is fixed at owner startup.
    kit_owner = None
    if args.render == "kit" and not asset_schedule:
        kit_owner, kit_worker_args = _launch_kit_owner(args, spec)
        base += kit_worker_args

    success, fail = 0, 0
    try:
        for i in range(args.num_episodes):
            ep_id = args.start_ep + i
            if args.resume and _episode_done(args.output_dir, ep_id):
                logger.info(f"Episode {ep_id} already done (resume) — skipping")
                continue
            episode_owner = None
            cmd = list(base)
            if asset_schedule:
                asset = asset_by_episode[ep_id]
                provenance = {
                    "pool_id": asset_pool["pool_id"],
                    "pool_digest": asset_pool["pool_digest"],
                    "asset_seed": args.asset_seed,
                    "schedule_digest": asset_schedule["schedule_digest"],
                    **asset,
                }
                cmd += [
                    "--cloth", asset["path"],
                    "--_asset_entry_json", json.dumps(provenance, separators=(",", ":")),
                ]
                if args.render == "kit":
                    episode_spec = TaskSpec.from_name(args.task, args.randomization)
                    _set_cloth_reference(episode_spec, asset["path"])
                    apply_eval_seed(episode_spec, args.eval_seed)
                    if args.control:
                        episode_spec.control_mode = args.control
                    episode_owner, kit_worker_args = _launch_kit_owner(args, episode_spec)
                    cmd += kit_worker_args
            cmd += ["--_single_episode_id", str(ep_id)]
            t0 = time.time()
            try:
                rc = subprocess.call(cmd, timeout=args.timeout) if args.timeout else subprocess.call(cmd)
            finally:
                if episode_owner is not None:
                    try:
                        episode_owner.close()
                    except Exception:
                        pass
            dt = time.time() - t0
            if rc == 0:
                success += 1
            else:
                fail += 1
            summary = _write_summary(args, spec)
            logger.info(f"Episode {ep_id} rc={rc} ({dt:.1f}s) | "
                        f"{success} ok / {fail} fail | rate={summary['success_rate']}%")
    finally:
        if kit_owner is not None:
            try:
                kit_owner.close()
            except Exception:
                pass
        if server_proc is not None:
            # Kill the whole session, not just the direct child: it was started with
            # start_new_session, and lingbot's entry point is torchrun, whose workers
            # would otherwise survive and keep the port -- which is exactly how the
            # next eval ends up silently attached to a stale checkpoint.
            _kill_server(server_proc, logger)

    summary = _write_summary(args, spec)
    print(f"== done: {summary['success_count']}/{summary['num_episodes']} success "
          f"({summary['success_rate']}%) -> {args.output_dir}/eval_results.json ==")


# ----------------------------------------------------------------------------
def _build_parser(add_help=True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, add_help=add_help,
                                 conflict_handler="resolve")
    ap.add_argument("--task", default="grasp_silk_bimanual")
    ap.add_argument("--cloth", default=None,
                    help="override task cloth with an assets/ relative or absolute directory")
    ap.add_argument("--asset_pool", default=None,
                    help="fixed asset-pool YAML; assigns one validated asset per episode")
    ap.add_argument("--asset_seed", type=int, default=20260915,
                    help="seed for balanced shuffled asset-pool cycles")
    ap.add_argument("--policy_type", default="lerobot")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint path")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--start_ep", type=int, default=0)
    ap.add_argument("--randomization", default=None,
                    help="difficulty preset in config/randomization/<name>.yaml "
                         "(clean/seen/unseen), deep-merged over the task's randomization block. "
                         "Use the held-out (unseen) split at eval to avoid train/eval texture leak.")
    ap.add_argument("--eval_seed", type=int, default=41,
                    help="ONE eval seed overriding randomization.seed for ALL axes "
                         "(geometric + texture + light) -> reproducible. Default 41 is held "
                         "out from every collection seed in use (0 / 42 / 43). It was 42 until "
                         "2026-07-26, which silently collided with the four task yamls whose "
                         "randomization.seed is 42 (pack/flatten_tshirt/fold_v3/bag) -- eval "
                         "then replayed training initial conditions. apply_eval_seed() now "
                         "hard-fails on such a collision.")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose episode_<id>.json already exists in --output_dir "
                         "(idempotent restart; lets the benchmark supervisor relaunch after a force-kill)")
    ap.add_argument("--render", default="kit", choices=["newton", "kit"])
    ap.add_argument("--viewer", default="null", choices=["gl", "kit", "both", "null"],
                    help="live visualization for INSPECTION (same scene, two lenses). "
                         "gl = Newton OpenGL debug window (true physics geometry/contact). "
                         "kit = native Omniverse Kit viewport (photorealistic = what the "
                         "policy actually sees; needs --render kit; persists across the run). "
                         "both = gl + kit side by side. null = headless (default). "
                         "Use with --num_episodes 1 to eyeball a single episode.")
    ap.add_argument("--output_dir", default="data/eval_out/run")
    ap.add_argument("--save_video", action="store_true",
                    help="save per-episode MP4 from rendered eval frames")
    ap.add_argument("--video_camera", default="static_cam",
                    help="camera(s) to save as MP4, comma-separated or 'all'; default static_cam")
    ap.add_argument("--sim_duration", type=float, default=12.0,
                    help="max episode seconds (ticks = sim_duration * control_fps) unless --max_steps")
    ap.add_argument("--max_steps", type=int, default=None, help="max control ticks (overrides sim_duration)")
    ap.add_argument("--check_interval", type=float, default=1.0, help="success-check period (s)")
    ap.add_argument("--ema_alpha", type=float, default=0.0, help="action EMA (0 = off)")
    ap.add_argument("--clip_gripper", action="store_true", help="snap gripper to grasp_close_val when closing")
    ap.add_argument("--no_interp_action", action="store_true",
                    help="escape hatch: command the policy action once per control step and hold it for "
                         "control_ratio frames (the pre-2026-09-18 path) instead of interpolating the arm "
                         "target linearly per frame like data collection does")
    ap.add_argument("--grasp_close_val", type=float, default=0.0)
    ap.add_argument("--clamp_ground", action="store_true",
                    help="prevent the gripper TCP from dipping below --min_ground_z by "
                         "FK+IK correcting the policy's joint targets each control step "
                         "(mirrors the demo's z>=0 grasp floor; avoids table/cloth "
                         "penetration when the policy commands too low).")
    ap.add_argument("--min_ground_z", type=float, default=0.0,
                    help="world-frame z floor for the gripper TCP under --clamp_ground (m).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", default=None)
    ap.add_argument("--control", default=None, choices=["kinematic", "pd"])
    ap.add_argument("--timeout", type=int, default=400, help="per-episode timeout (s)")
    ap.add_argument("--n_action_steps", type=int, default=None)
    # decoupled policy server (--policy_type lerobot_remote)
    ap.add_argument("--server_host", default="127.0.0.1")
    ap.add_argument("--server_port", type=int, default=5556)
    ap.add_argument("--server_task", default="dual-arm grasp the silk and lift")
    ap.add_argument("--server_python", default=None,
                    help="policy env's python (e.g. .../envs/lerobot_v042/bin/python); if set with "
                         "lerobot_remote, the orchestrator launches+kills serve_lerobot.py in it")
    # hidden worker handoff
    ap.add_argument("--_single_episode_id", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_shm_name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_zmq_endpoint", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_w2l", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_robot_root_offset", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_num_verts", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_asset_entry_json", default=None, help=argparse.SUPPRESS)
    return ap


def main() -> None:
    pre = _build_parser(add_help=False)
    known, _ = pre.parse_known_args()
    parser = _build_parser()
    from deformablebench.bench.policy._base import add_policy_arguments
    add_policy_arguments(parser, known.policy_type)
    args = parser.parse_args()
    if args._single_episode_id is not None:
        sys.exit(run_worker(args))
    run_orchestrator(args)


if __name__ == "__main__":
    main()
