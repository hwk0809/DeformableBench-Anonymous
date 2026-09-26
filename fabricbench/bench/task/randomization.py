"""Task-level randomization: placement, robot init, and cloth disturbances.

Adapted to the open newton env. Two newton-specific points vs. a solver that
pins vertices directly:

- **No solver pin constraint.** The reference solver pins a vertex with
  ``cloth_obj.set_pin`` + ``set_positions``. Newton's ``SolverStyle3D`` bakes
  particle mass at ``precompute()`` (so ``inv_mass=0`` does NOT hold a particle),
  and a once-per-frame snap relaxes between substeps for stiff cloth. So we use the
  env's ``set_particle_pin`` primitive, which snaps the particle (pos + zero vel)
  after *every substep*; neighbours follow via cloth constraints, producing the
  fold. Keep drag speed moderate so stiff sheets (silk) don't tunnel.
- **dt == one viewer frame.** ``env.step()`` advances one ``frame_dt``; durations
  are converted with ``env.frame_dt``.

Flow (driven by ``run_task``):
  1. ``Randomizer.sample_cloth_placement()``  — build-time: base_pos + Z-rot offset.
  2. ``Randomizer.apply_robot_init(env)``      — post-build, pre-settle: arms to a
     preset start pose (e.g. ``lift_silk``) so they don't lie on the cloth.
  3. ``Randomizer.apply_cloth_fold(env)``      — post-settle: random_fold to wrinkle
     a flat sheet so flatten/fling have something to do.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

# Robot init presets (real-teleop-derived; from deformable_bench). gripper_open is
# the repo's gripper open fraction [0,1] (1.0 = open), same convention as the env.
_ROBOT_INIT_PRESETS = {
    "zero": {
        "home_left": np.zeros(6), "home_right": np.zeros(6),
        "gripper_open": 1.0, "rand_range": np.zeros(6), "mirror": np.ones(6),
    },
    "lift_silk": {
        "home_left": np.array([-0.1240, 0.7980, -1.1250, -0.2253, 0.9856, 0.0829]),
        "home_right": np.array([0.1240, 0.7980, -1.1250, 0.2253, 0.9856, -0.0829]),
        "gripper_open": 1.0,
        "rand_range": np.array([0.1765, 0.1325, 0.1055, 0.1839, 0.1291, 0.2283]),
        "mirror": np.array([-1, 1, 1, -1, 1, -1]),
    },
}


def make_episode_rng(seed: int, episode_id: int) -> np.random.Generator:
    """Per-episode RNG: stable across runs, decorrelated across episodes."""
    return np.random.default_rng([int(seed), int(episode_id)])


def spawn_child_rng(parent: np.random.Generator) -> np.random.Generator:
    """Derive an independent child RNG from ``parent`` (ported from deformable_bench).

    Used for per-dimension isolation: each DR dimension draws from its OWN child
    RNG, so toggling or adding a dimension never shifts another dimension's random
    stream. The spawn ORDER is what fixes each dimension's values — see
    ``Randomizer.__init__``; new dimensions must be appended at the end.
    """
    return np.random.default_rng(parent.integers(2**63))


def fixed_randomization_config(cfg: dict | None) -> dict:
    """Return a non-random version of a task randomization block.

    This keeps deterministic task initialization such as cloth_init.base_pos and
    robot_init.start_pose, while disabling stochastic offsets, cloth
    disturbances, visual domain randomization, and robot pose noise.
    """
    import copy

    out = copy.deepcopy(cfg or {})
    cloth = out.setdefault("cloth_init", {})
    for key in ("pos_offset", "rot_offset", "wrinkled", "random_fold",
                "two_point_swing_drop", "procedural_fold", "airborne_two_pin_drop"):
        block = cloth.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    cloth["sampling"] = "random"
    robot = out.setdefault("robot_init", {})
    robot["noise_scale"] = 0.0
    for key in ("texture", "light"):
        block = out.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    return out


# ----------------------------------------------------------------------------
# kinematic-pin helpers
# ----------------------------------------------------------------------------
def _hold_args(env):
    """Args to hold the robot in place during a step (None if cloth-only)."""
    if not getattr(env, "has_robot", False):
        return None
    return (env.left_qpos, env.right_qpos, env.left_gripper_val, env.right_gripper_val)


def _step(env, hold):
    if hold is None:
        env.step()
    else:
        env.step(*hold)
    # Render each fold frame so the wrinkle/fold animates in gl (no-op when
    # viewer is None, e.g. collect_data — so the recording is unaffected).
    env.render()


def _advance(env, n: int, hold, pin=None) -> None:
    """Step ``n`` frames holding the robot; optionally drive a pinned particle.

    ``pin=(idx, pos)`` sets the env's per-substep kinematic pin to ``pos`` before
    each frame so the particle is held through every substep of that frame."""
    for _ in range(max(0, n)):
        if pin is not None:
            env.set_particle_pin(pin[0], pin[1])
        _step(env, hold)


def _sample(rng: np.random.Generator, value, default: float) -> float:
    """Sample a scalar range or return a scalar value."""
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def _set_pins(env, pins: dict[int, np.ndarray]) -> None:
    for idx, pos in pins.items():
        env.set_particle_pin(int(idx), pos)


def _drive_pins(env, hold, fixed, starts, ends, duration: float) -> None:
    """Linearly drive one or more particles while keeping other pins fixed."""
    steps = max(1, int(float(duration) / float(env.frame_dt)))
    for step in range(steps):
        alpha = (step + 1) / steps
        _set_pins(env, fixed)
        for idx, start in starts.items():
            env.set_particle_pin(idx, start * (1.0 - alpha) + ends[idx] * alpha)
        _step(env, hold)


def _hold_pins(env, hold, pins, duration: float) -> None:
    for _ in range(max(0, int(float(duration) / float(env.frame_dt)))):
        _set_pins(env, pins)
        _step(env, hold)


def _resolve_keypoint(env, name: str) -> int:
    if name not in env.cloth_keypoints:
        raise ValueError(f"cloth keypoint {name!r} is not in the asset")
    return int(env.cloth_keypoints[name])


def _translate_cloth(env, offset) -> None:
    """Translate all cloth particles during episode initialization."""
    offset = np.asarray(offset, dtype=np.float64)
    if hasattr(env, "state_0"):
        q = env.state_0.particle_q.numpy()
        q[:, :3] += offset
        env.state_0.particle_q.assign(q)
        qd = env.state_0.particle_qd.numpy()
        qd[:] = 0.0
        env.state_0.particle_qd.assign(qd)
    elif hasattr(env, "positions"):  # lightweight test environment
        env.positions[:, :3] += offset


def procedural_fold(env, cfg: dict, rng: np.random.Generator) -> dict:
    """Run a named semantic fold program while protecting task grasp patches."""
    hold = _hold_args(env)
    positions = np.asarray(env.get_sim_vertices(), dtype=np.float64)

    protected_names = list(cfg.get(
        "protected_keypoints", ["left_shoulder", "left_hem"]))
    radius = float(cfg.get("protection_radius", 0.045))
    max_pins = max(1, int(cfg.get("max_patch_pins", 16)))
    protected: dict[int, np.ndarray] = {}
    for name in protected_names:
        target_idx = _resolve_keypoint(env, name)
        distances = np.linalg.norm(
            positions[:, :2] - positions[target_idx, :2], axis=1)
        local = np.where(distances <= radius)[0]
        local = local[np.argsort(distances[local])]
        if len(local) > max_pins:
            local = local[np.linspace(0, len(local) - 1, max_pins, dtype=int)]
        for idx in local:
            protected[int(idx)] = positions[int(idx)].copy()
    _set_pins(env, protected)

    programs = cfg.get("programs", {}) or {}
    if not programs:
        raise ValueError("procedural_fold requires at least one program")
    names = list(programs)
    program_name = str(cfg.get("program") or names[int(rng.integers(0, len(names)))])
    if program_name not in programs:
        raise ValueError(f"unknown fold program {program_name!r}; have {names}")

    applied = []
    durations = cfg.get("segment_durations", [0.25, 0.5, 0.25])
    final_z = float(cfg.get("final_z", 0.012))
    # Sample one coherent fold depth for the whole program. Operations may
    # provide ``deep_offset`` as the other interpolation endpoint; configs
    # without it retain the original fixed-offset behavior.
    deep_probability = float(cfg.get("fold_depth_deep_probability", 0.0))
    if not 0.0 <= deep_probability <= 1.0:
        raise ValueError(
            "fold_depth_deep_probability must be between 0 and 1, "
            f"got {deep_probability}")
    depth_range = cfg.get("fold_depth_range")
    if deep_probability > 0.0 and rng.random() < deep_probability:
        depth_range = cfg.get("fold_depth_deep_range", [0.8, 1.0])
    fold_depth = _sample(rng, depth_range, 0.0)
    for operation in programs[program_name]:
        handle_name = operation["handle"]
        handle_idx = _resolve_keypoint(env, handle_name)
        if handle_idx in protected:
            raise ValueError(f"fold handle {handle_name!r} overlaps a protected patch")

        positions = np.asarray(env.get_sim_vertices(), dtype=np.float64)
        target_name = operation.get("target", "centroid")
        if target_name == "centroid":
            target_xy = positions[:, :2].mean(axis=0)
        else:
            target_xy = positions[_resolve_keypoint(env, target_name), :2].copy()
        offset = np.asarray(operation.get("offset", [0.0, 0.0]), dtype=np.float64)
        if "deep_offset" in operation:
            deep_offset = np.asarray(operation["deep_offset"], dtype=np.float64)
            offset = offset + fold_depth * (deep_offset - offset)
        target_xy += offset

        start = positions[handle_idx].copy()
        height = start[2] + _sample(
            rng, operation.get("lift_height", cfg.get("lift_height", [0.08, 0.16])), 0.12)
        lifted = np.array([start[0], start[1], height])
        moved = np.array([target_xy[0], target_xy[1], height])
        lowered = np.array([target_xy[0], target_xy[1],
                            float(operation.get("final_z", final_z))])
        _drive_pins(env, hold, protected, {handle_idx: start}, {handle_idx: lifted}, durations[0])
        _drive_pins(env, hold, protected, {handle_idx: lifted}, {handle_idx: moved}, durations[1])
        _drive_pins(env, hold, protected, {handle_idx: moved}, {handle_idx: lowered}, durations[2])

        # Release the moving handle but immediately restore the protected patches.
        env.clear_particle_pins()
        _set_pins(env, protected)
        _hold_pins(env, hold, protected, float(operation.get("settle", 0.1)))
        applied.append(handle_name)

    _hold_pins(env, hold, protected, float(cfg.get("protected_settle_duration", 0.8)))
    env.clear_particle_pins()
    _advance(env, int(float(cfg.get("settle_duration", 0.4)) / env.frame_dt), hold)
    return {"mode": "procedural_fold", "program": program_name,
            "fold_depth": round(fold_depth, 4),
            "handles": applied, "protected_keypoints": protected_names}


def _choose_random_pin_pair(env, cfg: dict, rng: np.random.Generator) -> list[int]:
    positions = np.asarray(env.get_sim_vertices(), dtype=np.float64)
    stride = max(1, int(cfg.get("candidate_stride", 10)))
    pool = np.arange(0, len(positions), stride, dtype=int)
    semantic = np.asarray(list(env.cloth_keypoints.values()), dtype=int)
    pool = np.unique(np.concatenate([pool, semantic]))
    first = int(pool[int(rng.integers(0, len(pool)))])
    distances = np.linalg.norm(positions[pool, :2] - positions[first, :2], axis=1)
    lo, hi = cfg.get("pin_separation", [0.1, 0.45])
    valid = pool[(distances >= float(lo)) & (distances <= float(hi)) & (pool != first)]
    second = int(valid[int(rng.integers(0, len(valid)))]) if len(valid) else int(
        pool[int(np.argmax(distances))])
    return [first, second]


def airborne_two_pin_drop(env, cfg: dict, rng: np.random.Generator) -> dict:
    """Raise a settled cloth, hold two random particles, then release and settle."""
    hold = _hold_args(env)
    positions = np.asarray(env.get_sim_vertices(), dtype=np.float64)
    initial_xy = positions[:, :2].mean(axis=0)
    pin_indices = _choose_random_pin_pair(env, cfg, rng)
    spawn_z = _sample(rng, cfg.get("spawn_centroid_z", [0.32, 0.48]), 0.4)
    spawn_x = _sample(rng, cfg.get("spawn_x_offset", [0.15, 0.25]), 0.2)
    offset = np.array([
        spawn_x,
        0.0,
        spawn_z - float(positions[:, 2].mean()),
    ])
    _translate_cloth(env, offset)

    raised = np.asarray(env.get_sim_vertices(), dtype=np.float64)
    pins = {idx: raised[idx].copy() for idx in pin_indices}
    hold_duration = _sample(rng, cfg.get("hold_duration", [1.0, 3.0]), 2.0)
    _hold_pins(env, hold, pins, hold_duration)
    env.clear_particle_pins()
    _advance(env, int(float(cfg.get("settle_duration", 2.0)) / env.frame_dt), hold)

    if bool(cfg.get("recenter_after_settle", True)):
        current = np.asarray(env.get_sim_vertices(), dtype=np.float64)
        delta_xy = initial_xy - current[:, :2].mean(axis=0)
        _translate_cloth(env, [delta_xy[0], delta_xy[1], 0.0])
        _advance(env, int(float(cfg.get("recenter_settle_duration", 0.3)) / env.frame_dt), hold)

    return {"mode": "airborne_two_pin_drop", "pin_indices": pin_indices,
            "spawn_centroid_z": round(spawn_z, 3),
            "spawn_x_offset": round(spawn_x, 3),
            "hold_duration": round(hold_duration, 3)}


# ----------------------------------------------------------------------------
def random_fold(env, cfg: dict, rng: np.random.Generator) -> dict:
    """Pin a corner particle and drag it (up -> across -> down) to wrinkle the sheet.

    Random_fold in target_offset mode:
    pick a corner keypoint, perturb within ``source.radius``, drag toward the
    centroid by ``target_offset.{delta_x,delta_y}``, hold, release, settle.
    """
    dt = float(env.frame_dt)
    hold = _hold_args(env)

    pre = float(cfg.get("pre_settle_duration", 0.2))
    _advance(env, int(pre / dt), hold)

    keypoints = env.cloth_keypoints  # {name: vertex_index}
    cand_names = list(cfg.get("source", {}).get("candidates", []))
    radius = float(cfg.get("source", {}).get("radius", 0.02))
    resolved = [(n, int(keypoints[n])) for n in cand_names if n in keypoints]
    if not resolved:
        return {"skipped": True, "reason": "no_valid_candidates"}

    pos = env.get_sim_vertices()
    name, vtx = resolved[int(rng.integers(0, len(resolved)))]
    cand_xy = pos[vtx, :2]
    near = np.where(np.linalg.norm(pos[:, :2] - cand_xy, axis=1) < radius)[0]
    pin_idx = int(rng.choice(near)) if len(near) else vtx
    source = pos[pin_idx].copy()

    to = cfg.get("target_offset", {}) or {}
    cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
    sign_x = -1.0 if source[0] > cx else 1.0   # delta_x always toward centroid
    sign_y = -1.0 if source[1] > cy else 1.0   # delta_y signed (+ inward, - outward)
    dx = float(rng.uniform(*to.get("delta_x", [0.18, 0.32])))
    dy = float(rng.uniform(*to.get("delta_y", [-0.05, 0.18])))
    target_xy = np.array([source[0] + sign_x * dx, source[1] + sign_y * dy])

    fold_h = float(cfg.get("fold_height", 0.08))
    final_z = float(cfg.get("final_z", 0.005))
    waypoints = [
        source,
        np.array([source[0], source[1], fold_h]),
        np.array([target_xy[0], target_xy[1], fold_h]),
        np.array([target_xy[0], target_xy[1], final_z]),
    ]
    seg_durs = cfg.get("seg_durations", [0.2, 0.3, 0.2])

    for a, b, sd in zip(waypoints[:-1], waypoints[1:], seg_durs):
        steps = max(1, int(float(sd) / dt))
        for s in range(steps):
            alpha = (s + 1) / steps
            _advance(env, 1, hold, pin=(pin_idx, a * (1.0 - alpha) + b * alpha))

    hold_steps = max(1, int(float(cfg.get("pin_duration", 0.2)) / dt))
    _advance(env, hold_steps, hold, pin=(pin_idx, waypoints[-1]))

    env.clear_particle_pins()  # release
    _advance(env, int(float(cfg.get("settle_duration", 1.0)) / dt), hold)  # settle
    return {"candidate": name, "pin_idx": pin_idx,
            "source": source.tolist(), "target_xy": target_xy.tolist()}


def _recenter_cloth_xy(env, target_xy) -> None:
    """Translate every cloth particle in XY so the current centroid lands at
    ``target_xy``. Cancels the horizontal drift a hang/drop induces so the
    crumpled pile stays over the reachable spot (mirrors deformable_bench's
    ``_recenter_cloth_centroid_xy``). Z is untouched."""
    q = env.state_0.particle_q.numpy()
    if q is None or len(q) == 0:
        return
    centroid_xy = q[:, :2].mean(axis=0)
    offset = np.asarray(target_xy, dtype=np.float64) - centroid_xy
    q[:, 0] += offset[0]
    q[:, 1] += offset[1]
    env.state_0.particle_q.assign(q)
    qd = env.state_0.particle_qd.numpy()
    qd[:] = 0.0
    env.state_0.particle_qd.assign(qd)


def _drive_pins_pair(env, hold, idx_a, idx_b, start_a, end_a, start_b, end_b, duration):
    """Linearly drive two pinned particles from start->end over ``duration`` s.
    (env's per-substep pin interpolation further smooths each frame's step.)"""
    dt = float(env.frame_dt)
    steps = max(1, int(duration / dt))
    for s in range(steps):
        alpha = (s + 1) / steps
        env.set_particle_pin(idx_a, start_a * (1.0 - alpha) + end_a * alpha)
        env.set_particle_pin(idx_b, start_b * (1.0 - alpha) + end_b * alpha)
        _step(env, hold)


def _hold_pins_pair(env, hold, idx_a, idx_b, pos_a, pos_b, duration):
    """Hold two pinned particles at fixed positions for ``duration`` s."""
    dt = float(env.frame_dt)
    if duration <= 0:
        return
    for _ in range(max(1, int(duration / dt))):
        env.set_particle_pin(idx_a, pos_a)
        env.set_particle_pin(idx_b, pos_b)
        _step(env, hold)


def two_point_swing_drop(env, cfg: dict, rng: np.random.Generator) -> dict:
    """Port of deformable_bench ``apply_two_point_swing_drop`` (flatten_tshirt v3 init).

    Cloth spawns high (yaml base_pos.z≈0.55). Pin two keypoints (left_shoulder,
    left_hem) at their elevated positions, then: pull both out along +X (Y
    squeezed toward the midpoint) → hold (others droop) → expose/release (drag
    along X while descending to drop_final_z) → short final hold → un-pin →
    free-fall settle → recenter. The two grasp points land on the outer layer of
    the pile, visible for downstream grasp."""
    dt = float(env.frame_dt)
    hold = _hold_args(env)

    keypoints = env.cloth_keypoints  # {name: vertex_index}
    kp_names = list((cfg.get("source", {}) or {}).get("keypoints", []))
    if len(kp_names) != 2:
        return {"skipped": True, "reason": f"need 2 keypoints, got {len(kp_names)}"}
    resolved = []
    for name in kp_names:
        if name not in keypoints:
            return {"skipped": True, "reason": f"keypoint {name!r} not in asset"}
        resolved.append((name, int(keypoints[name])))
    (name_a, idx_a), (name_b, idx_b) = resolved

    def _s(val, default):
        if val is None:
            return float(default)
        if isinstance(val, (list, tuple)):
            return float(rng.uniform(val[0], val[1]))
        return float(val)

    pull_x_target = float(cfg.get("pull_x_target", 0.5))
    pull_x_fraction = _s(cfg.get("pull_x_fraction_range", [0.8, 1.0]), 0.9)
    pull_y_ratio = _s(cfg.get("pull_y_ratio_range", [0.1, 0.25]), 0.2)
    pull_duration = _s(cfg.get("pull_duration", 0.3), 0.3)
    hold_at_pull = _s(cfg.get("hold_at_pull_duration", [1.5, 3.0]), 2.0)
    release_x_target = float(cfg.get("release_x_target", 0.1))
    release_x_fraction = _s(cfg.get("release_x_fraction_range", [0.8, 1.0]), 0.9)
    release_duration = _s(cfg.get("release_duration", 0.5), 0.5)
    drop_final_z = float(cfg.get("drop_final_z", 0.05))
    expose_while_drop = bool(cfg.get("expose_while_drop", False))
    drop_then_expose = bool(cfg.get("drop_then_expose", False))
    drop_duration = _s(cfg.get("drop_duration", release_duration), release_duration)
    expose_x_target = float(cfg.get("expose_x_target", release_x_target))
    expose_duration = _s(cfg.get("expose_duration", release_duration), release_duration)
    final_hold_duration = _s(cfg.get("final_hold_duration", 0.3), 0.3)
    pre_settle = float(cfg.get("pre_settle_duration", 0.0))
    settle_dur = float(cfg.get("settle_duration", 2.0))
    recenter = bool(cfg.get("recenter_after_settle", True))

    expose_x_fraction = release_x_fraction
    expose_y_offset = 0.0
    if expose_while_drop or drop_then_expose:
        expose_x_fraction = _s(cfg.get("expose_x_fraction_range", [0.9, 1.0]), release_x_fraction)
        expose_y_offset = _s(cfg.get("expose_y_offset_range", 0.0), 0.0)

    if pre_settle > 0:
        _advance(env, int(pre_settle / dt), hold)

    pos = env.get_sim_vertices()
    pre_centroid_xy = pos[:, :2].mean(axis=0).astype(np.float64)
    src_a = pos[idx_a].copy()
    src_b = pos[idx_b].copy()
    mid = (src_a + src_b) / 2.0

    pull_a = np.array([src_a[0] + pull_x_fraction * (pull_x_target - src_a[0]),
                       mid[1] + pull_y_ratio * (src_a[1] - mid[1]), src_a[2]])
    pull_b = np.array([src_b[0] + pull_x_fraction * (pull_x_target - src_b[0]),
                       mid[1] + pull_y_ratio * (src_b[1] - mid[1]), src_b[2]])

    drop_a = drop_b = None
    if expose_while_drop:
        release_a = np.array([pull_a[0] + expose_x_fraction * (expose_x_target - pull_a[0]),
                              pull_a[1] + expose_y_offset, drop_final_z])
        release_b = np.array([pull_b[0] + expose_x_fraction * (expose_x_target - pull_b[0]),
                              pull_b[1] + expose_y_offset, drop_final_z])
    elif drop_then_expose:
        drop_a = np.array([pull_a[0], pull_a[1], drop_final_z])
        drop_b = np.array([pull_b[0], pull_b[1], drop_final_z])
        release_a = np.array([drop_a[0] + expose_x_fraction * (expose_x_target - drop_a[0]),
                              drop_a[1] + expose_y_offset, drop_final_z])
        release_b = np.array([drop_b[0] + expose_x_fraction * (expose_x_target - drop_b[0]),
                              drop_b[1] + expose_y_offset, drop_final_z])
    else:
        release_a = np.array([pull_a[0] + release_x_fraction * (release_x_target - pull_a[0]),
                              pull_a[1], drop_final_z])
        release_b = np.array([pull_b[0] + release_x_fraction * (release_x_target - pull_b[0]),
                              pull_b[1], drop_final_z])

    # pin both at spawn; seg1 pull -> seg2 hold -> seg3 release -> seg4 final hold
    env.set_particle_pin(idx_a, src_a)
    env.set_particle_pin(idx_b, src_b)
    _drive_pins_pair(env, hold, idx_a, idx_b, src_a, pull_a, src_b, pull_b, pull_duration)
    _hold_pins_pair(env, hold, idx_a, idx_b, pull_a, pull_b, hold_at_pull)
    if expose_while_drop:
        _drive_pins_pair(env, hold, idx_a, idx_b, pull_a, release_a, pull_b, release_b, expose_duration)
    elif drop_then_expose:
        _drive_pins_pair(env, hold, idx_a, idx_b, pull_a, drop_a, pull_b, drop_b, drop_duration)
        _drive_pins_pair(env, hold, idx_a, idx_b, drop_a, release_a, drop_b, release_b, expose_duration)
    else:
        _drive_pins_pair(env, hold, idx_a, idx_b, pull_a, release_a, pull_b, release_b, release_duration)
    _hold_pins_pair(env, hold, idx_a, idx_b, release_a, release_b, final_hold_duration)

    env.clear_particle_pins()
    _advance(env, int(settle_dur / dt), hold)
    if recenter:
        _recenter_cloth_xy(env, pre_centroid_xy)
        _advance(env, int(0.5 / dt), hold)

    return {"mode": "two_point_swing_drop", "keypoints": [name_a, name_b],
            "pin_indices": [idx_a, idx_b], "pull_x": round(pull_x_target, 2),
            "pull_frac": round(pull_x_fraction, 2), "hold": round(hold_at_pull, 2),
            "recentered": recenter}


# ----------------------------------------------------------------------------
# ════════════════════════════════════════════════════════════════════════════
#  Visual DR (Kit-side): texture (USD material variant) + lighting.
#  Param-generator classes — they only SAMPLE and emit an ``get_isaac_params()``
#  dict; the Kit render script (omniverse_kit_render_v7.py _apply_textures /
#  _apply_lighting) does the actual USD apply, driven per-episode via the
#  ShmKitBridge.start_episode(...) EPISODE_START message. Ported verbatim from
#  deformable_bench so the prim paths / variant names match the shared scene.
# ════════════════════════════════════════════════════════════════════════════

def _color_temp_to_rgb(temp_k: float) -> np.ndarray:
    """Color temperature (K) -> normalized RGB [0,1] (Tanner Helland approx)."""
    temp = temp_k / 100.0
    if temp <= 66:
        r = 1.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592) / 255.0
    if temp <= 66:
        g = (99.4708025861 * np.log(temp) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492) / 255.0
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = (138.5177312231 * np.log(temp - 10) - 305.0447927307) / 255.0
    return np.clip([r, g, b], 0.0, 1.0)


# Prim path of the table material variantSet in whole_scene_piper_thin_gripper_base.usda
# (references table_variants/table_all_11k.usda). Matches deformable_bench exactly.
_DEFAULT_TABLE_MATERIAL_PATH = "/World/table/geometry/Looks/material_0/material_0"
_DEFAULT_TABLE_VARIANT_SET = "Look"


class TextureRandomizer:
    """Per-episode table-texture DR — selects a "Look" variant on the table
    material prim. Kit-side apply; this class only samples params + metadata.

    Config (under ``randomization.texture``):
      enabled: bool
      split: "seen" | "unseen"     — half of the unified table_all_11k.usda
        variant set to sample from. Variant names: ``texture_<split>_<idx:04d>``.
      num_textures: int (defaults seen=10000, unseen=1000)
      held_out_range: [lo, hi] | None  — restrict idx to [lo, hi) (e.g. reserve a tail)
      sampling: "sobol" | "random";  sobol_seed: int
    """

    _DEFAULT_NUM = {"seen": 10000, "unseen": 1000}

    def __init__(self, config: dict, rng: np.random.Generator | None = None,
                 seed: int | None = None):
        self.enabled = config.get("enabled", False)
        self.split = config.get("split", "seen")
        self.num_textures = int(config.get("num_textures", self._DEFAULT_NUM.get(self.split, 0)))
        self.held_out_range = config.get("held_out_range", None)
        self.sampling = config.get("sampling", "random")
        # Unified seed: ONE randomization.seed drives geometric + texture + light.
        # `seed` (the global per-run seed passed by Randomizer) overrides any legacy
        # per-dim sobol_seed, so a single --eval_seed makes every axis reproducible.
        self.sobol_seed = int(seed) if seed is not None else int(config.get("sobol_seed", 42))
        self.material_path = config.get("material_path", _DEFAULT_TABLE_MATERIAL_PATH)
        self.variant_set = config.get("variant_set", _DEFAULT_TABLE_VARIANT_SET)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._last_params: dict = {}
        if self.enabled:
            lo, hi = self.held_out_range or [0, self.num_textures]
            logger.info(f"[TextureRandomizer] split={self.split} num={self.num_textures} "
                        f"range=[{lo},{hi}) sampling={self.sampling}")

    def _sample_idx(self, episode_id: int | None) -> int:
        lo, hi = self.held_out_range or [0, self.num_textures]
        if hi <= lo:
            raise ValueError(f"TextureRandomizer: invalid range [{lo}, {hi})")
        if self.sampling == "sobol" and episode_id is not None:
            from scipy.stats.qmc import Sobol
            engine = Sobol(d=1, scramble=True, seed=self.sobol_seed)
            if episode_id > 0:
                engine.fast_forward(episode_id)
            u = float(engine.random(1)[0][0])
            return int(lo + u * (hi - lo))
        return int(self.rng.integers(lo, hi))

    def generate(self, episode_id: int | None = None) -> dict:
        if not self.enabled:
            self._last_params = {}
            return {}
        idx = self._sample_idx(episode_id)
        self._last_params = {"split": self.split, "table_texture_idx": idx,
                             "variant_name": f"texture_{self.split}_{idx:04d}"}
        logger.info(f"[TextureRandomizer] idx={idx} variant={self._last_params['variant_name']}")
        return self._last_params

    def get_isaac_params(self) -> dict:
        if not self._last_params:
            return {}
        return {"table_material_path": self.material_path,
                "table_variant_set": self.variant_set,
                "table_variant_name": self._last_params["variant_name"]}

    def get_metadata(self) -> dict:
        return {"enabled": self.enabled, "params": self._last_params.copy()}


class LightRandomizer:
    """Per-episode lighting DR — exposure (stops) + color temp (K) + dome intensity.
    Kit-side apply; this class only samples. Config under ``randomization.light``:
      exposure_range: [lo, hi] stops; color_temp_range: [lo, hi] K;
      dome_intensity_range: [lo, hi]; sampling: "sobol" | "random"; sobol_seed: int
    """

    def __init__(self, config: dict, rng: np.random.Generator | None = None,
                 seed: int | None = None):
        self.enabled = config.get("enabled", False)
        self.exposure_range = config.get("exposure_range", [-1.5, 1.5])
        self.color_temp_range = config.get("color_temp_range", [3500, 6500])
        self.dome_intensity_range = config.get("dome_intensity_range", [0.0, 0.3])
        self.sampling = config.get("sampling", "random")
        # Unified seed (see TextureRandomizer): global seed overrides per-dim sobol_seed.
        self.sobol_seed = int(seed) if seed is not None else int(config.get("sobol_seed", 42))
        self.rng = rng if rng is not None else np.random.default_rng()
        self._last_params: dict = {}
        if self.enabled:
            logger.info(f"[LightRandomizer] exposure={self.exposure_range} "
                        f"color_temp={self.color_temp_range} dome={self.dome_intensity_range} "
                        f"sampling={self.sampling}")

    def generate(self, episode_id: int | None = None) -> dict:
        if not self.enabled:
            self._last_params = {}
            return {}
        if self.sampling == "sobol" and episode_id is not None:
            from scipy.stats.qmc import Sobol
            engine = Sobol(d=3, scramble=True, seed=self.sobol_seed)
            if episode_id > 0:
                engine.fast_forward(episode_id)
            s = engine.random(1)[0]
            exposure = self.exposure_range[0] + s[0] * (self.exposure_range[1] - self.exposure_range[0])
            color_temp = self.color_temp_range[0] + s[1] * (self.color_temp_range[1] - self.color_temp_range[0])
            dome = self.dome_intensity_range[0] + s[2] * (self.dome_intensity_range[1] - self.dome_intensity_range[0])
        else:
            exposure = self.rng.uniform(*self.exposure_range)
            color_temp = self.rng.uniform(*self.color_temp_range)
            dome = self.rng.uniform(*self.dome_intensity_range)
        color_rgb = _color_temp_to_rgb(color_temp)
        self._last_params = {"exposure_offset": float(exposure), "color_temp": float(color_temp),
                             "color_rgb": color_rgb.tolist(), "dome_intensity": float(dome)}
        logger.info(f"[LightRandomizer] exposure={exposure:+.2f}st color_temp={color_temp:.0f}K dome={dome:.3f}")
        return self._last_params

    def get_isaac_params(self) -> dict:
        if not self._last_params:
            return {}
        return {"rectlight_exposure_offset": self._last_params["exposure_offset"],
                "rectlight_color": self._last_params["color_rgb"],
                "dome_intensity": self._last_params["dome_intensity"],
                "dome_color": self._last_params["color_rgb"]}

    def get_metadata(self) -> dict:
        return {"enabled": self.enabled, "params": self._last_params.copy()}


class Randomizer:
    """Owns the per-episode RNG and applies the ``randomization:`` config block."""

    def __init__(self, cfg: dict | None = None, episode_id: int = 0):
        self.cfg = cfg or {}
        self.episode_id = int(episode_id)
        # ONE seed for the whole run: geometric (make_episode_rng) + texture + light
        # all derive from it. --eval_seed (eval) / --seed (collect) override
        # randomization.seed upstream.
        self.seed = int(self.cfg.get("seed", 0))
        episode_rng = make_episode_rng(self.seed, self.episode_id)

        # Per-dimension child RNGs, spawned in a LOCKED order (ported from
        # deformable_bench EnvRandomizer). Each dimension draws only from its own
        # child so toggling/adding a dimension never shifts another's stream. The
        # three reserved slots (camera_extrinsic / domain_rand / table_height) are
        # not implemented here yet — we still spawn+discard them so that wiring one
        # in later drops into its reserved position without perturbing cloth /
        # robot / light / rigid / texture. NEW dims append at the very end.
        self._cloth_rng = spawn_child_rng(episode_rng)
        self._robot_rng = spawn_child_rng(episode_rng)
        spawn_child_rng(episode_rng)   # reserved: camera_extrinsic (not implemented)
        spawn_child_rng(episode_rng)   # reserved: domain_rand (not implemented)
        spawn_child_rng(episode_rng)   # reserved: table_height (not implemented)
        light_rng = spawn_child_rng(episode_rng)
        self._rigid_rng = spawn_child_rng(episode_rng)
        texture_rng = spawn_child_rng(episode_rng)

        self.cloth_init = self.cfg.get("cloth_init", {}) or {}
        self.robot_init = self.cfg.get("robot_init", {}) or {}
        self.rigid_init = self.cfg.get("rigid_init", {}) or {}
        # Visual DR (Kit-side) — sample per-episode params now; get_isaac_*_params()
        # feed ShmKitBridge.start_episode(). No-op dicts when the blocks are absent/off.
        self.texture_rand = TextureRandomizer(self.cfg.get("texture", {}) or {}, texture_rng, seed=self.seed)
        self.light_rand = LightRandomizer(self.cfg.get("light", {}) or {}, light_rng, seed=self.seed)
        self.texture_rand.generate(episode_id=self.episode_id)
        self.light_rand.generate(episode_id=self.episode_id)

    # --- Kit-side visual DR params (fed to ShmKitBridge.start_episode) ------
    def get_isaac_texture_params(self) -> dict:
        return self.texture_rand.get_isaac_params()

    def get_isaac_lighting_params(self) -> dict:
        return self.light_rand.get_isaac_params()

    def _cloth_init_mode(self) -> str | None:
        """Resolve YAML-selected mode, with legacy enabled blocks as fallback."""
        mode = self.cloth_init.get("mode")
        supported = (
            "procedural_fold", "airborne_two_pin_drop",
            "random_fold", "two_point_swing_drop",
        )
        if mode is not None:
            mode = str(mode)
            if mode not in supported:
                raise ValueError(f"unknown cloth_init mode {mode!r}; have {supported}")
            return mode

        enabled = [name for name in ("random_fold", "two_point_swing_drop")
                   if (self.cloth_init.get(name, {}) or {}).get("enabled")]
        if len(enabled) > 1:
            raise ValueError(f"enable only one cloth-init disturbance, got {enabled}")
        return enabled[0] if enabled else None

    @property
    def cloth_init_runs_before_settle(self) -> bool:
        """Whether the disturbance must start from the high spawn state."""
        return self._cloth_init_mode() == "two_point_swing_drop"

    # --- build-time: cloth placement ---------------------------------------
    def sample_cloth_placement(self, default_base_pos) -> tuple[list[float], float]:
        """(base_pos[3], rot_z_deg). base_pos = cfg base + pos_offset; rot from rot_offset."""
        base = list(self.cloth_init.get("base_pos", default_base_pos))
        po = self.cloth_init.get("pos_offset", {}) or {}
        ro = self.cloth_init.get("rot_offset", {}) or {}
        sampling = str(self.cloth_init.get("sampling", "random")).lower()
        if sampling not in ("random", "sobol"):
            raise ValueError(f"cloth_init.sampling must be 'random' or 'sobol', got {sampling!r}")

        # Match deformable_bench's cloth-placement Sobol semantics exactly:
        # active dimensions only, scrambled with the collection seed, and the
        # episode id selects the point via fast_forward.  Keeping this local to
        # cloth placement means robot/visual RNG streams remain isolated.
        sobol = None
        if sampling == "sobol":
            ndim = (2 if po.get("enabled") else 0)
            if ro.get("enabled") and ro.get("axis_z_only", False):
                ndim += 1
            if ndim:
                from scipy.stats.qmc import Sobol
                sobol_seed = int(self.cloth_init.get("sobol_seed", self.seed))
                engine = Sobol(d=ndim, scramble=True, seed=sobol_seed)
                if self.episode_id > 0:
                    engine.fast_forward(self.episode_id)
                sobol = engine.random(1)[0]
        dim = 0

        if po.get("enabled"):
            rx = po.get("range_x", [0.0, 0.0])
            ry = po.get("range_y", [0.0, 0.0])
            if sobol is not None:
                base[0] += float(rx[0] + sobol[dim] * (rx[1] - rx[0]))
                base[1] += float(ry[0] + sobol[dim + 1] * (ry[1] - ry[0]))
                dim += 2
            else:
                base[0] += float(self._cloth_rng.uniform(*rx))
                base[1] += float(self._cloth_rng.uniform(*ry))
        rot_z = 0.0
        # base_quat orientation (Z angle). Previously dropped, so assets needing a
        # non-identity base_quat (e.g. the vertical tshirt fold, rotated 90 deg to
        # fit the workspace) were placed unrotated and sprawled out of reach.
        bq = self.cloth_init.get("base_quat")
        if bq is not None:
            rot_z += math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
        if ro.get("enabled"):
            lo, hi = ro.get("angle_range_deg", [0.0, 0.0])
            if sobol is not None and ro.get("axis_z_only", False):
                rot_z += float(lo + sobol[dim] * (hi - lo))
            else:
                rot_z += float(self._cloth_rng.uniform(lo, hi))
        return base, rot_z

    # --- post-build, pre-settle: robot start pose --------------------------
    def apply_robot_init(self, env) -> dict:
        """Set arms to a named preset (+noise) so they clear the cloth, and make it
        the env's home pose (tasks return here). No-op if cloth-only or unset."""
        name = self.robot_init.get("start_pose")
        if not name or not getattr(env, "has_robot", False):
            return {}
        if name not in _ROBOT_INIT_PRESETS:
            raise ValueError(f"unknown robot_init start_pose {name!r}; "
                             f"have {list(_ROBOT_INIT_PRESETS)}")
        p = _ROBOT_INIT_PRESETS[name]
        scale = float(self.robot_init.get("noise_scale", 1.0))
        rr = p["rand_range"] * scale
        noise = np.zeros(6) if scale == 0.0 else self._robot_rng.uniform(-rr, rr)
        q_l = p["home_left"] + noise
        q_r = p["home_right"] + noise * p["mirror"]
        g_open = float(p.get("gripper_open", 1.0))  # repo convention: 1 = open
        # PLACE the robot at the start pose, don't ask the servo to slew there:
        # in PD mode the settle loop steps without a command, so a target-only
        # init leaves the arm tens of degrees short (see DeformableEnv.set_robot_pose).
        place = getattr(env, "set_robot_pose", None) or env.set_robot_control
        place(q_l, q_r, g_open, g_open)
        # Make this the home pose so the task's own home references the raised arms.
        import dataclasses
        env.robot = dataclasses.replace(env.robot, home_left=tuple(q_l), home_right=tuple(q_r))
        return {"start_pose": name, "noise_max": float(np.max(np.abs(noise)))}

    # --- post-build, pre-settle: free rigid body poses ----------------------
    def apply_rigid_init(self, env) -> dict:
        """Offset each named free rigid body from its build-time (nominal) pose.

        Config mirrors deformable_bench's ``rigid_init`` block::

            rigid_init:
              snickers_body:
                pos_offset: {enabled: true, range_x: [-0.05, 0.05], ...}
                yaw_offset: {enabled: false, range_deg: [-15, 15]}

        Applied before settle so the object drops onto its randomized spot.
        """
        results: dict = {}
        for body_name, body_cfg in self.rigid_init.items():
            if body_name not in getattr(env, "rigid_body_index", {}):
                results[body_name] = {"skipped": "not in scene"}
                continue
            body_cfg = body_cfg or {}
            pose = env.get_rigid_body_pose(body_name)  # [px,py,pz, qx,qy,qz,qw]
            pos, quat = pose[:3].copy(), pose[3:7].copy()

            po = body_cfg.get("pos_offset", {}) or {}
            if po.get("enabled"):
                pos[0] += float(self._rigid_rng.uniform(*po.get("range_x", [0.0, 0.0])))
                pos[1] += float(self._rigid_rng.uniform(*po.get("range_y", [0.0, 0.0])))
                pos[2] += float(self._rigid_rng.uniform(*po.get("range_z", [0.0, 0.0])))

            yo = body_cfg.get("yaw_offset", {}) or {}
            if yo.get("enabled"):
                yaw = math.radians(float(self._rigid_rng.uniform(*yo.get("range_deg", [0.0, 0.0]))))
                h = yaw / 2.0
                # q_yaw (xyzw) * quat: rotate the nominal orientation about world Z
                zw = np.array([0.0, 0.0, math.sin(h), math.cos(h)])
                x1, y1, z1, w1 = zw
                x2, y2, z2, w2 = quat
                quat = np.array([
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ])

            env.set_rigid_body_pose(body_name, pos, quat)
            results[body_name] = {"pos": [round(float(v), 4) for v in pos]}
        return results

    # --- post-settle: cloth init disturbance (fold / hang-drop) ------------
    def apply_cloth_fold(self, env) -> dict:
        """Apply the single cloth-init mode selected by the task YAML."""
        mode = self._cloth_init_mode()
        cfg = self.cloth_init.get(mode, {}) or {} if mode else {}
        if mode == "procedural_fold":
            return procedural_fold(env, cfg, self._cloth_rng)
        if mode == "airborne_two_pin_drop":
            return airborne_two_pin_drop(env, cfg, self._cloth_rng)
        if mode == "random_fold":
            return random_fold(env, cfg, self._cloth_rng)
        if mode == "two_point_swing_drop":
            return two_point_swing_drop(env, cfg, self._cloth_rng)
        return {}
