"""``DeformableEnv`` — the runtime object the benchmark talks to.

Owns the ``newton.Model``/``State``, the cloth solver, and the collision
pipeline; the (co-)sim loop lives here (Newton has no separate "stepper" — the
loop belongs with the state it advances, like a Newton ``Example``).

Two interfaces on one object:
- example-compatible driving for the web (``step`` / ``render`` + ``state_0`` /
  ``sim_time`` / ``frame_dt``), so the existing Reflex/ViewerWeb bridge can drive it;
- a thin bench interface (``reset`` / ``step_to_time`` / ``get_sim_vertices`` /
  ``get_current_sim_time``) consumed by the benchmark pipeline.

P0: cloth only (no robot), open ``SolverStyle3D``.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import warp as wp

import newton

from deformablebench.sim.builder import build_cloth_scene, build_robot_cloth_scene
from deformablebench.sim.builder import SceneConfig
from deformablebench.sim.gripper_collision_controller import (
    GripperCollisionTrickConfig,
    NewtonGripperCollisionController,
)
from deformablebench.sim.solvers import make_cloth_solver, make_robot_solver


# --- Robot index resolution (by NAME, robot-agnostic; was sim/robot.py) ---
def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1] if label else label


def resolve_joint_qindices(model, names):
    """MJCF joint names -> joint_q indices via model.joint_label."""
    qstart = model.joint_q_start.numpy()
    by = {_basename(lbl): int(qstart[i]) for i, lbl in enumerate(model.joint_label) if lbl}
    return np.array([by[n] for n in names], dtype=int)


def resolve_joint_dofindices(model, names):
    """MJCF joint names -> joint_qd (DOF) indices (for PD target arrays)."""
    dstart = model.joint_qd_start.numpy()
    by = {_basename(lbl): int(dstart[i]) for i, lbl in enumerate(model.joint_label) if lbl}
    return np.array([by[n] for n in names], dtype=int)


def resolve_joint_child_bodies(model, names) -> np.ndarray:
    """Joint names -> child body ids, using the robot asset's joint contract.

    This is the generic way to discover finger bodies: robot assets already
    declare their gripper joints, so the contact solver must not duplicate that
    information as Piper-specific body labels.
    """
    by = {_basename(lbl): i for i, lbl in enumerate(model.joint_label) if lbl}
    child = model.joint_child.numpy()
    missing = [name for name in names if name not in by]
    if missing:
        raise KeyError(f"gripper joints not found in model.joint_label: {missing}")
    return np.asarray([int(child[by[name]]) for name in names], dtype=np.int32)


def resolve_gripper_bodies(model, robot) -> dict[str, tuple[int, ...]]:
    """Resolve each hand's moving finger bodies from ``RobotConfig`` joints."""
    return {
        "left": tuple(int(v) for v in resolve_joint_child_bodies(model, robot.left_gripper)),
        "right": tuple(int(v) for v in resolve_joint_child_bodies(model, robot.right_gripper)),
    }


def resolve_body_index(model, name: str) -> int:
    for i, lbl in enumerate(model.body_label):
        if _basename(lbl) == name:
            return i
    raise KeyError(f"body {name!r} not found in model.body_label")


def apply_pd_gains(model, spec) -> None:
    """Set PD target gains + POSITION_VELOCITY mode on the model for the robot's DOFs.

    POSITION_VELOCITY makes SolverMuJoCo install a SECOND (velocity) actuator per
    DOF: the position one drops its damping term (biasprm ``[0,-kp,0]``) and the
    velocity one supplies ``kd*(qd_des - qd)`` from ``control.joint_target_qd``.
    At ``qd_des = 0`` the two modes are algebraically identical
    (``kp*(q_des-q) - kd*qd``) and share the joint's ``actfrcrange`` clamp, so the
    mode by itself changes nothing — it opens the velocity feedforward channel
    that ``DeformableEnv._sync_pd_target`` drives.
    """
    from newton import JointTargetMode

    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    mode = model.joint_target_mode.numpy()

    def _set(names, kps, kds):
        for d, kp, kdv in zip(resolve_joint_dofindices(model, names), kps, kds):
            ke[d], kd[d], mode[d] = kp, kdv, int(JointTargetMode.POSITION_VELOCITY)

    _set(spec.left_arm, spec.arm_kp, spec.arm_kd)
    _set(spec.right_arm, spec.arm_kp, spec.arm_kd)
    _set(spec.left_gripper, (spec.gripper_kp, spec.gripper_kp), (spec.gripper_kd, spec.gripper_kd))
    _set(spec.right_gripper, (spec.gripper_kp, spec.gripper_kp), (spec.gripper_kd, spec.gripper_kd))
    model.joint_target_ke = wp.array(ke, dtype=model.joint_target_ke.dtype, device=model.device)
    model.joint_target_kd = wp.array(kd, dtype=model.joint_target_kd.dtype, device=model.device)
    model.joint_target_mode = wp.array(mode, dtype=model.joint_target_mode.dtype, device=model.device)

    # Gripper effort limit -> SolverMuJoCo writes it to the joint's actfrcrange,
    # which clamps the position+velocity actuator sum. Raising gripper_kp alone
    # makes a stiff jaw that cannot yield to the cloth; the rating is what turns
    # grip force from "geometry decides" into "motor decides".
    if spec.gripper_effort > 0.0:
        eff = model.joint_effort_limit.numpy()
        for d in resolve_joint_dofindices(model, tuple(spec.left_gripper) + tuple(spec.right_gripper)):
            eff[d] = spec.gripper_effort
        model.joint_effort_limit = wp.array(
            eff, dtype=model.joint_effort_limit.dtype, device=model.device)


def apply_robot_gravcomp(model, exclude_bodies=()) -> None:
    """Gravity-compensate every robot body for PD mode (model.mujoco.gravcomp=1).

    deformable_bench's PD law feeds ``qfrc_bias`` (gravity+Coriolis) forward, so
    its arms hold pose exactly; SolverMuJoCo's position actuators have no such
    feedforward and the arm sags ~cm under gravity — enough to press the fingers
    into the table at low grasp heights, where ground friction then stalls the
    gripper. MuJoCo's per-body gravcomp reproduces the gravity part 1:1. Free
    rigid objects (``exclude_bodies``) keep gravcomp=0 so they still fall.
    """
    attrs = getattr(model, "mujoco", None)
    gc_arr = getattr(attrs, "gravcomp", None) if attrs is not None else None
    if gc_arr is None:
        return
    gc = gc_arr.numpy()
    gc[:] = 1.0
    for name in exclude_bodies:
        gc[resolve_body_index(model, name)] = 0.0
    gc_arr.assign(gc)


@wp.kernel
def _pd_coriolis_ff_kernel(
    qfrc_bias: wp.array2d[wp.float32],       # MuJoCo C(q,qd)qd + g(q), [1, nv]
    qfrc_gravcomp: wp.array2d[wp.float32],   # MuJoCo per-body gravcomp, [1, nv]
    newton_dof: wp.array[wp.int32],
    mj_dof: wp.array[wp.int32],
    # outputs
    joint_f: wp.array[wp.float32],
):
    """Gather MuJoCo's Coriolis bias into Newton's ``control.joint_f``.

    ``joint_f`` is copied straight into ``qfrc_applied`` by SolverMuJoCo's
    ``apply_mjc_qfrc_kernel`` for 1-DOF joints, so this is a direct torque
    feedforward. With per-body gravcomp on, ``qfrc_gravcomp`` is (measured,
    exactly) the gravity part of ``qfrc_bias``, so the difference is the pure
    Coriolis/centrifugal term and gravity is not compensated twice.
    """
    i = wp.tid()
    m = mj_dof[i]
    joint_f[newton_dof[i]] = qfrc_bias[0, m] - qfrc_gravcomp[0, m]


@wp.kernel
def _lerp_dofs_kernel(
    start: wp.array[wp.float32],
    end: wp.array[wp.float32],
    alpha: float,
    out_a: wp.array[wp.float32],
    out_b: wp.array[wp.float32],
):
    """``out = start + (end - start) * alpha``, written to two destinations.

    The substep interpolation of the robot's frame endpoints, on the device.
    Doing it on the host (numpy + ``assign``) costs one pageable host->device
    copy per substep, and a single such copy makes the whole substep loop
    uncapturable as a CUDA graph — see ``DeformableEnv._simulate``.
    """
    i = wp.tid()
    v = start[i] + (end[i] - start[i]) * alpha
    out_a[i] = v
    out_b[i] = v


@wp.kernel
def _pin_lerp_kernel(
    pin_idx: wp.array[wp.int32],
    pin_start: wp.array[wp.vec3],
    pin_target: wp.array[wp.vec3],
    alpha_prev: float,
    alpha_curr: float,
    inv_dt: float,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    """Device version of ``DeformableEnv._apply_pins_substep``.

    Same arithmetic as the host loop it replaces (position lerps applied->target
    across the frame, velocity = per-substep delta / sim_dt), but it never
    touches the host: the old code did ``particle_q.numpy()`` +
    ``particle_qd.numpy()`` + two ``assign`` PER SUBSTEP, i.e. two full
    device->host readbacks of every particle in the cloth just to move one or two
    pinned vertices.  Besides the copies, a readback inside the substep loop
    makes the whole frame uncapturable as a CUDA graph, so every frame that had a
    live pin fell back to per-launch dispatch (T27 A.6: 148 of flatten's 666
    frames, at 2.25x the cost of a replayed frame).
    """
    t = wp.tid()
    i = pin_idx[t]
    s = pin_start[t]
    d = pin_target[t] - s
    p_prev = s + d * alpha_prev
    p_curr = s + d * alpha_curr
    q[i] = p_curr
    qd[i] = (p_curr - p_prev) * inv_dt


@wp.kernel
def _pin_zero_vel_kernel(
    pin_idx: wp.array[wp.int32],
    qd: wp.array[wp.vec3],
):
    """Frame end: zero each pin's velocity so a held pin doesn't drift."""
    t = wp.tid()
    qd[pin_idx[t]] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _lerp_dofs_kernel1(
    start: wp.array[wp.float32],
    end: wp.array[wp.float32],
    alpha: float,
    out: wp.array[wp.float32],
):
    """Single-destination :func:`_lerp_dofs_kernel` (the PD target ramp)."""
    i = wp.tid()
    out[i] = start[i] + (end[i] - start[i]) * alpha


@wp.kernel
def _zero_selected_body_f_kernel(
    body_ids: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
):
    """R5-B: zero the listed bodies' wrench before it is fed to SolverMuJoCo.

    With ``gripper_joint_in_solver`` the cloth-finger force balance is resolved
    inside the cloth solve, so feeding the finger reaction to MuJoCo as xfrc
    as well would count it twice. The accumulation buffer (``_cloth_body_f``)
    is left intact so the per-finger force telemetry (doc GRIPPER-CONTACT
    §3.1 ③b) keeps reading."""
    i = wp.tid()
    body_f[body_ids[i]] = wp.spatial_vector()


@wp.kernel
def _frame_rate_kernel(
    start: wp.array[wp.float32],
    end: wp.array[wp.float32],
    inv_frame_dt: float,
    out: wp.array[wp.float32],
):
    """Per-substep joint velocity of the FK ramp.

    ``(q_curr - q_prev) / sim_dt`` is the same for every substep of a frame
    (the ramp is linear and the substeps are equal), so it is computed once per
    frame: ``(end - start) / frame_dt``.
    """
    i = wp.tid()
    out[i] = (end[i] - start[i]) * inv_frame_dt


def apply_gripper_contact_override(model, cfg: SceneConfig) -> None:
    """Stiff finger contact for pinching free rigid objects (PD mode).

    1:1 with deformable_bench's scene-XML ``<pair>`` override for
    gripper<->snickers (solref="-50000 -1000", solimp="0.99 0.999 0.0001"):
    the raw params + priority go on the FINGER shapes, so they govern exactly
    the finger contacts (the light bar keeps its soft geom solref vs the
    ground — putting the stiff params there bounces/NaNs it at dt=2 ms).
    Without this the default contact (timeconst 0.002, dampratio ~1.6) lets a
    1.6 N pinch sink ~8 mm into the bar and it squirts out during transport.
    """
    if not cfg.rigid_bodies:
        return
    attrs = getattr(model, "mujoco", None)
    if attrs is None or getattr(attrs, "solref", None) is None:
        return
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

    gripper_body_ids = {
        body for bodies in resolve_gripper_bodies(model, cfg.robot).values()
        for body in bodies
    }
    solref = attrs.solref.numpy()
    mode = attrs.solref_mode.numpy()
    solimp = attrs.geom_solimp.numpy()
    prio = attrs.geom_priority.numpy()
    shape_body = model.shape_body.numpy()
    n = 0
    for sid, bid in enumerate(shape_body):
        if int(bid) in gripper_body_ids:
            solref[sid] = (-50000.0, -1000.0)
            mode[sid] = SOLREF_MODE_RAW
            solimp[sid] = (0.99, 0.999, 0.0001, 0.5, 2.0)
            prio[sid] = 1
            n += 1
    attrs.solref.assign(solref)
    attrs.solref_mode.assign(mode)
    attrs.geom_solimp.assign(solimp)
    attrs.geom_priority.assign(prio)
    print(f"[env] stiff gripper contact override on {n} finger shapes", flush=True)


def apply_shape_material_overrides(model, cfg: SceneConfig) -> None:
    """Tune rigid shape friction for cloth-body contact.

    SolverStyle3D mixes the global soft-contact friction with per-shape material
    friction, so keeping the table modest and only making gripper fingers sticky
    is more useful than raising ``soft_contact_mu`` globally.
    """

    if cfg.default_shape_mu is None and cfg.gripper_shape_mu is None:
        return
    if not hasattr(model, "shape_material_mu") or model.shape_material_mu is None:
        return

    mu = model.shape_material_mu.numpy()
    if cfg.default_shape_mu is not None:
        mu[:] = float(cfg.default_shape_mu)

    if cfg.gripper_shape_mu is not None and getattr(model, "body_count", 0) > 0:
        gripper_body_ids = {
            body for bodies in resolve_gripper_bodies(model, cfg.robot).values()
            for body in bodies
        }
        shape_body = model.shape_body.numpy()
        for shape_id, body_id in enumerate(shape_body):
            if int(body_id) in gripper_body_ids:
                mu[shape_id] = float(cfg.gripper_shape_mu)

    model.shape_material_mu.assign(mu)


# Viewers that run HOST code per substep to inject forces (picking + wind).
# Everything else in newton.viewer inherits ``ViewerBase.apply_forces``, which
# is an empty ``pass`` -- see _viewer_injects_forces.
_INTERACTIVE_FORCE_VIEWERS = ("ViewerGL", "ViewerRTX")


def _viewer_injects_forces(viewer) -> bool:
    """Does this viewer write into ``state.body_f``/``particle_f`` per substep?

    Only interactive backends do (mouse picking, wind).  The test CANNOT be
    ``hasattr(viewer, "apply_forces")``: that method is declared on
    ``ViewerBase`` as a no-op ``pass``, so every backend has it -- ViewerNull
    included -- and using it as the guard silently disabled ``cuda_graph`` in
    every headless run ever done (T26 §6.4; the measured cost was 1.7-1.8x
    wall clock).  A viewer defined outside newton can opt in or out explicitly
    by setting the class attribute ``interactive_forces``.
    """
    if viewer is None:
        return False
    flag = getattr(viewer, "interactive_forces", None)
    if flag is not None:
        return bool(flag)
    return any(c.__name__ in _INTERACTIVE_FORCE_VIEWERS for c in type(viewer).__mro__)


class DeformableEnv:
    def __init__(self, viewer=None, cfg: SceneConfig | None = None):
        self.viewer = viewer
        self.cfg = cfg or SceneConfig()

        self.frame_dt = 1.0 / float(self.cfg.view_fps)
        self.sim_substeps = max(1, int(self.cfg.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.has_robot = self.cfg.robot is not None
        self.control_mode = self.cfg.control_mode if self.has_robot else "kinematic"
        if self.has_robot:
            self.model, self.state_0 = build_robot_cloth_scene(self.cfg)
        else:
            self.model, self.state_0 = build_cloth_scene(self.cfg)
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._gripper_body_ids_by_side = (
            resolve_gripper_bodies(self.model, self.cfg.robot)
            if self.has_robot else {"left": (), "right": ()}
        )
        self._gripper_body_ids = {
            body for bodies in self._gripper_body_ids_by_side.values()
            for body in bodies
        }
        apply_shape_material_overrides(self.model, self.cfg)

        self.cloth_solver = make_cloth_solver(self.cfg.solver, self.model, self.state_0, self.cfg)
        collision = getattr(self.cloth_solver, "collision", None)
        gripper_shape_ids = []
        if collision is not None and self._gripper_body_ids:
            shape_body = self.model.shape_body.numpy()
            gripper_shape_ids = [
                shape for shape, body in enumerate(shape_body)
                if int(body) in self._gripper_body_ids
            ]
        if (
            collision is not None
            and hasattr(collision, "set_shape_contact_stiffness")
            and self.cfg.gripper_contact_ke > 0.0
            and gripper_shape_ids
        ):
            collision.set_shape_contact_stiffness(
                gripper_shape_ids, self.cfg.gripper_contact_ke,
            )
        # Table (ground PLANE) stiffness against the cloth, table shapes only.
        # A soft table (newton stock ke 15 N/m) lets the blade tip push the cloth
        # 2-3 mm INTO the ground, so the normal load the fingers should be
        # carrying goes into the table spring instead (T1/T4). The stop-surface
        # recipe needs a table the cloth cannot sink into. 0 = leave stock.
        if (
            collision is not None
            and hasattr(collision, "set_shape_contact_stiffness")
            and self.cfg.table_contact_ke > 0.0
        ):
            _plane = [
                i for i, t in enumerate(self.model.shape_type.numpy())
                if int(t) == int(newton.GeoType.PLANE)
            ]
            if not _plane:
                raise RuntimeError(
                    "table_contact_ke set but the scene has no PLANE shape")
            collision.set_shape_contact_stiffness(
                _plane, float(self.cfg.table_contact_ke))
            print(f"[env] table_contact_ke: shapes {_plane} -> "
                  f"{self.cfg.table_contact_ke:g} N/m", flush=True)
        # E4: give the gripper shapes their own penalty band, decoupled from
        # particle_radius. `_soft_contact_margin` below widens contact GENERATION
        # to match -- a pair is only emitted while d < margin + particle_radius,
        # so a 4 mm band on a 0.5 mm radius would otherwise never see a contact.
        self._gripper_contact_offset = 0.0
        if (
            collision is not None
            and hasattr(collision, "set_shape_contact_offset")
            and self.cfg.gripper_contact_offset > 0.0
            and gripper_shape_ids
        ):
            collision.set_shape_contact_offset(
                gripper_shape_ids, float(self.cfg.gripper_contact_offset),
            )
            self._gripper_contact_offset = float(self.cfg.gripper_contact_offset)
        # R9: hardening compression law on that band. Must come AFTER the offset
        # is set -- d_max = eps_max * band, and the band is read per shape inside
        # the kernel, so the two settings are independent, but the log line only
        # makes sense in this order.
        if (
            collision is not None
            and hasattr(collision, "set_shape_contact_hardening")
            and self.cfg.gripper_contact_hardening > 0.0
            and gripper_shape_ids
        ):
            collision.set_shape_contact_hardening(
                gripper_shape_ids, float(self.cfg.gripper_contact_hardening),
            )
        # T17: cap the tangent stiffness that law hands the EXPLICITLY coupled
        # finger.  The cloth solve is implicit, but the reaction wrench is
        # evaluated once per cloth substep and held constant for the whole
        # MuJoCo step, so the finger sees the contact spring at the position the
        # step STARTED from -- an explicit spring on an implicitfast integrator.
        # The stability boundary is therefore NOT the textbook 4m/dt^2: with
        # implicitfast the velocity-dependent terms (joint damping + the
        # velocity actuator's kv) move to the left-hand side, which buys
        #     K_crit = 4*M/dt^2 + 2*c/dt,   M = body mass + joint armature,
        #                                   c = dof_damping + kv
        # and the contact channel gets K_crit minus the position actuator's kp
        # (kp is a second explicit spring on the same DOF).  For this robot
        # (M = 0.025 + 0.005 kg, dt = 2 ms, c = 5 + 100, kp = 2e4) that is
        # k_crit = 1.15e5 N/m; closed form, a numpy reproduction of the scheme
        # and real MuJoCo implicitfast agree to 0.002% (T17 tools/
        # t17_kcrit_1dof.py, out/kcrit_1dof.json), and it brackets both measured
        # anchors: the linear-sum load 8e4 N/m runs stable, the hardened 1.6e7
        # spikes.  The budget is half of it (kcap 9.2 in stopface_h.yaml).
        # m is the mass of the BODY the shape hangs on (the MJCF finger link)
        # and dt is the CLOTH substep -- both read here, never hardcoded; the
        # armature that separates m from M is folded into kcap.  Must come after
        # set_shape_contact_hardening so the self-cert line below reads the
        # eps_max that is actually in force.
        if (
            collision is not None
            and hasattr(collision, "set_shape_hardening_kcap")
            and self.cfg.gripper_contact_hardening > 0.0
            and self.cfg.gripper_contact_hardening_kcap > 0.0
            and gripper_shape_ids
        ):
            _shape_body = self.model.shape_body.numpy()
            _body_mass = self.model.body_mass.numpy()
            _kcap = float(self.cfg.gripper_contact_hardening_kcap)
            _ids, _kmax = [], []
            for _sid in gripper_shape_ids:
                _bid = int(_shape_body[_sid])
                if _bid < 0:
                    continue
                _m = float(_body_mass[_bid])
                if _m <= 0.0:
                    continue
                _k = _kcap * _m / (self.sim_dt * self.sim_dt)
                _ids.append(int(_sid))
                _kmax.append(_k)
                print(f"[collision] T17 hardening KCAP: shape {int(_sid)} "
                      f"m={_m:.6g} kg dt={self.sim_dt:.6g} s "
                      f"kcap={_kcap:.6g} k_max={_k:.6g} N/m "
                      f"[k_crit = 4M/dt^2 + 2c/dt - kp, "
                      f"T17 tools/t17_kcrit_1dof.py]", flush=True)
            if _ids:
                collision.set_shape_hardening_kcap(_ids, _kmax)
        if (
            collision is not None
            and self.cfg.rigid_feature_contact
            and gripper_shape_ids
        ):
            collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
            shape_flags = self.model.shape_flags.numpy()
            feature_shape_ids = [
                shape for shape in gripper_shape_ids
                if int(shape_flags[shape]) & collide_particles
            ]
            collision.enable_rigid_feature_contacts(
                feature_shape_ids,
                contact_radius=float(np.max(self.model.particle_radius.numpy())),
                search_margin=float(self.cfg.cloth_params.soft_contact_margin),
            )
        if (
            collision is not None
            and self.cfg.rigid_untangling
            and gripper_shape_ids
            and hasattr(collision, "enable_rigid_untangling")
        ):
            # R2-3: ICM recovery against the gripper triangles. Only the gripper
            # shapes -- the table is a half-space the cloth is meant to lie on,
            # and an untangling force there has nothing to recover from.
            collide_particles_icm = int(newton.ShapeFlags.COLLIDE_PARTICLES)
            shape_flags_icm = self.model.shape_flags.numpy()
            icm_shapes = [
                shape for shape in gripper_shape_ids
                if int(shape_flags_icm[shape]) & collide_particles_icm
            ]
            collision.enable_rigid_untangling(
                icm_shapes,
                stiff_factor=float(self.cfg.rigid_untangling_stiff),
                thickness=float(self.cfg.rigid_untangling_thickness),
                candidate_capacity=int(self.cfg.rigid_untangling_capacity),
            )
        if (
            collision is not None
            and self.cfg.contact_material_modulus > 0.0
            and not self.cfg.tri_sdf_compliant
            and hasattr(collision, "enable_material_contact_stiffness")
        ):
            # (Skipped in compliant mode: there the material stiffness already
            # rides on the triangle constraint for the gripper, and the stock
            # per-particle path must keep its per-shape constant for the table.)
            # E3(b): the penalty is per-particle, so a constant ke makes the total
            # normal load scale with the particle count. Deriving k_i from the
            # cloth's own material (modulus x tributary area / thickness) keeps
            # the load invariant when the mesh is refined.
            collision.enable_material_contact_stiffness(
                modulus=float(self.cfg.contact_material_modulus),
                thickness=float(self.cfg.contact_material_thickness),
            )
        if (
            collision is not None
            and self.cfg.tri_sdf_contact
            and gripper_shape_ids
            and hasattr(collision, "enable_triangle_sdf_contacts")
        ):
            # E3: triangle-level detection against a baked gripper SDF. Only the
            # gripper shapes get it -- the table is a half-space the cloth is
            # supposed to slide on, and E0c measured what a position constraint
            # against it costs.
            collide_particles_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
            shape_flags_np = self.model.shape_flags.numpy()
            tri_sdf_shapes = [
                shape for shape in gripper_shape_ids
                if int(shape_flags_np[shape]) & collide_particles_flag
            ]
            # k_tri = E_t * A_tri / t0 uses the CLOTH's thickness, not the
            # numerical shell: the two coincide at h = t0/2 (every stack to
            # date), so this is the same number today and only keeps an h
            # ladder from rescaling the stiffness behind our back. Older forks
            # do not take the argument, hence the signature check.
            _t0_kw = {}
            try:
                import inspect as _inspect

                if "thickness" in _inspect.signature(
                    collision.enable_triangle_sdf_contacts
                ).parameters:
                    _t0_kw["thickness"] = float(self.cfg.contact_material_thickness)
                # T8: true static friction (tangential anchor) on the tri-SDF
                # channel. 0 = off = the IPC regularised law, bit-identical.
                # Signature-checked like `thickness`: an older fork silently
                # keeps the old behaviour instead of raising.
                if (
                    float(self.cfg.sdf_anchor_kt_ratio) > 0.0
                    and "anchor_kt_ratio" in _inspect.signature(
                        collision.enable_triangle_sdf_contacts
                    ).parameters
                ):
                    _t0_kw["anchor_kt_ratio"] = float(self.cfg.sdf_anchor_kt_ratio)
                elif float(self.cfg.sdf_anchor_kt_ratio) > 0.0:
                    print(
                        "[env] sdf_anchor_kt_ratio requested but this fork's "
                        "enable_triangle_sdf_contacts takes no anchor_kt_ratio "
                        "-- T8 anchor NOT active",
                        flush=True,
                    )
            except (TypeError, ValueError):
                pass
            collision.enable_triangle_sdf_contacts(
                tri_sdf_shapes,
                half_thickness=float(self.cfg.tri_sdf_half_thickness),
                voxel=float(self.cfg.tri_sdf_voxel),
                pad=float(self.cfg.tri_sdf_pad),
                refine_steps=int(self.cfg.tri_sdf_refine_steps),
                max_correction=float(self.cfg.tri_sdf_max_correction),
                compliant=bool(self.cfg.tri_sdf_compliant),
                modulus=float(self.cfg.contact_material_modulus),
                **_t0_kw,
            )
            if (self.cfg.tri_sdf_compliant and tri_sdf_shapes
                    and self.cfg.tri_sdf_disable_stock_penalty):
                # The compliant triangle contact now carries the gripper's normal
                # load. Leaving the stock per-particle penalty on those shapes
                # would double-count it (and drag particle_radius back in as a
                # force scale). The table keeps its stock contact.
                collision.set_shape_contact_stiffness(tri_sdf_shapes, 0.0)
        self._contact_projection = bool(
            collision is not None
            and self.cfg.contact_projection
            and hasattr(collision, "enable_contact_projection")
        )
        if self._contact_projection:
            collision.enable_contact_projection(
                slack=float(self.cfg.contact_projection_slack),
                iterations=int(self.cfg.contact_projection_iterations),
                **({"friction_scale": float(self.cfg.contact_projection_friction)}
                   if self.cfg.contact_projection_friction > 0.0 else {}),
                **({"interleaved": True}
                   if self.cfg.contact_projection_interleaved else {}),
            )
            if self.cfg.contact_projection_gripper_only and hasattr(
                collision, "set_projection_shapes"
            ):
                # The projection is here to stop the jaw penetrating the cloth.
                # Running it against the table too costs flatten coverage: the
                # cloth has to slide across the table to spread out, and a
                # position-level constraint fights that every iteration
                # (measured 0.51 vs 0.74 with the projection off).
                sb = self.model.shape_body.numpy()
                collision.set_projection_shapes(
                    [s for s in range(len(sb)) if int(sb[s]) in self._gripper_body_ids]
                )
        if (
            collision is not None
            and self.cfg.friction_anchor_kt_ratio > 0.0
            and hasattr(collision, "enable_friction_anchors")
        ):
            collision.enable_friction_anchors(float(self.cfg.friction_anchor_kt_ratio))

        if (
            collision is not None
            and self.cfg.friction_avbd_dual_k > 0.0
            and hasattr(collision, "enable_avbd_friction")
        ):
            collision.enable_avbd_friction(float(self.cfg.friction_avbd_dual_k))

        # R5-B: finger prismatic DOFs solved inside the cloth solver (default
        # off; every downstream branch hangs off these flags, so the default
        # path is untouched). Setup happens in the PD block below (_gjs_setup):
        # the MuJoCo actuator gains on those DOFs must be zeroed BEFORE
        # SolverMuJoCo is constructed.
        self._gjs_enabled = False
        self._gjs_finger_bodies_wp = None
        self._gjs_debug = bool(int(os.environ.get("DEFORMABLEBENCH_GJS_DEBUG", "0")))
        self._gjs_wanted = bool(
            self.cfg.gripper_joint_in_solver
            and self.control_mode == "pd"
            and self.has_robot
            and collision is not None
            and hasattr(collision, "enable_gripper_joint_solve")
        )

        self._projection_impulse = bool(
            self._contact_projection
            and self.cfg.contact_projection_impulse
            and hasattr(collision, "accumulate_projection_impulse")
        )

        self._gripper_collision_controller = None

        def _projection_prev_pose():
            """Extra kwargs for project_contacts (empty unless S2 friction is on).

            States are swapped before the projection call, so ``state_1`` holds
            the pose the substep started from. The position-level friction branch
            needs it to separate cloth slip from the gripper's own motion; the
            normal-only pass does not, and older fork builds do not accept it.
            """
            if self.cfg.contact_projection_friction <= 0.0:
                return {}
            return {"body_q_prev": self.state_1.body_q}

        self._projection_prev_pose = _projection_prev_pose

        # Kinematic particle pins via the ParticleFlags.ACTIVE bit (see
        # set_particle_pin). ``_particle_pins`` = per-idx TARGET for this frame;
        # ``_particle_pin_applied`` = where the pin sat at frame start. The substep
        # loop lerps applied->target so a pin driven at frame rate still moves
        # smoothly across substeps — the same treatment the robot joints get
        # (_joint_q_applied -> _joint_q). Without this, a fast pin drag snaps once
        # per frame and over-stretches stiff cloth. Used by task randomization
        # (random_fold / two_point_swing_drop) and grasp_silk_hanging_pull.
        self._particle_pins: dict[int, np.ndarray] = {}
        self._particle_pin_applied: dict[int, np.ndarray] = {}
        # Device mirror of the pin table, read by _pin_lerp_kernel.  Rebuilt (and
        # the graph invalidated) only when the SET of pinned indices changes;
        # moving an existing pin is an in-place assign, so a drag keeps its graph.
        # DEFORMABLEBENCH_PIN_HOST=1 restores the old host-loop path (A/B + escape hatch).
        self._pin_order: list[int] = []
        self._pin_idx_dev = None
        self._pin_start_dev = None
        self._pin_target_dev = None
        self._pin_host = os.environ.get("DEFORMABLEBENCH_PIN_HOST", "0") not in ("", "0")

        # Contact generation must cover the E4 penalty band: create_soft_contacts
        # emits a pair while ``d < margin + particle_radius``, so a band wider
        # than that would be starved of contacts (no pair, no force at all).
        # Only ever widens, and only when the band is on, so the default path is
        # untouched. Extra pairs outside the band cost nothing numerically --
        # depth <= 0 returns zero force and zero Hessian.
        self._soft_contact_margin = float(self.cfg.cloth_params.soft_contact_margin)
        if self._gripper_contact_offset > 0.0:
            self._soft_contact_margin = max(
                self._soft_contact_margin, self._gripper_contact_offset)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.cfg.broad_phase,
            soft_contact_margin=self._soft_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # CUDA graph capture of the substep loop (SceneConfig.cuda_graph): the
        # loop issues ~11k kernel launches per frame for kernels that are often
        # narrower than a warp, so the CPU cannot keep the GPU fed. Off when a
        # viewer can inject forces (interactive dragging runs host code per
        # substep) — see _simulate for the rest of the preconditions.
        self._viewer_injects_forces = _viewer_injects_forces(self.viewer)
        self._use_cuda_graph = (
            bool(self.cfg.cuda_graph) and self.model.device.is_cuda
            and not self._viewer_injects_forces)
        # Whole-frame capture bakes the state_0/state_1 DEVICE POINTERS in, and
        # the substep loop swaps that pair once per substep.  With an EVEN substep
        # count a frame ends with the pair back in its captured roles, so one
        # graph replays forever; with an ODD count the roles alternate frame to
        # frame, so we keep ONE GRAPH PER ROLE ASSIGNMENT, keyed by
        # (id(state_0), id(state_1)) -- exactly the trick _robot_step already uses
        # for the MuJoCo graph.  At most two entries either way.  (A replay runs
        # no host code, so the Python-level swap the loop would have done is
        # re-applied by _swap_after_replay.)
        self._frame_graphs: dict = {}
        self._graph_warm = 0
        self._robot_graph_only = False
        # Last reason a frame ran per-launch instead of replaying, so the
        # transition gets exactly one log line (see _note_plain).
        self._plain_reason: str | None = None
        self._mj_graphs: dict = {}
        self._mj_graph_warm = 0

        # Robot: resolve semantic joint groups by NAME (robot-agnostic; see robot.py),
        # init the kinematic target, and pose via eval_fk. SolverStyle3D treats robot
        # bodies as read-only kinematic colliders (one-way robot->cloth contact).
        self.robot_solver = None
        if self.has_robot:
            self.robot = self.cfg.robot   # RobotConfig from robot_meta.yaml (RobotAsset)
            self.idx_l_arm = resolve_joint_qindices(self.model, self.robot.left_arm)
            self.idx_r_arm = resolve_joint_qindices(self.model, self.robot.right_arm)
            self.idx_l_grip = resolve_joint_qindices(self.model, self.robot.left_gripper)
            self.idx_r_grip = resolve_joint_qindices(self.model, self.robot.right_gripper)
            # DOF-space indices for the PD target array: joint_target_q is sized
            # per-DOF, not per-coordinate. With only 1-DOF joints the two spaces
            # coincide, but a free joint has 7 coords vs 6 DOFs, so the robot's
            # target slots must be addressed in DOF space explicitly.
            self.idx_l_arm_d = resolve_joint_dofindices(self.model, self.robot.left_arm)
            self.idx_r_arm_d = resolve_joint_dofindices(self.model, self.robot.right_arm)
            self.idx_l_grip_d = resolve_joint_dofindices(self.model, self.robot.left_gripper)
            self.idx_r_grip_d = resolve_joint_dofindices(self.model, self.robot.right_gripper)
            self.eef_l_body = resolve_body_index(self.model, self.robot.eef_left)
            self.eef_r_body = resolve_body_index(self.model, self.robot.eef_right)
            self.base_l_body = resolve_body_index(self.model, self.robot.base_left)
            self.base_r_body = resolve_body_index(self.model, self.robot.base_right)
            self._joint_q = self.model.joint_q.numpy().copy()  # joint target (host)
            self._joint_q_applied = self._joint_q.copy()
            # Device mirrors of the frame's FK endpoints. Refreshed once per
            # frame in _simulate (outside any capture); the substep ramp then
            # runs entirely on the device (see _lerp_dofs_kernel).
            dev = self.model.device
            self._q_start_dev = wp.array(self._joint_q_applied, dtype=wp.float32, device=dev)
            self._q_end_dev = wp.array(self._joint_q, dtype=wp.float32, device=dev)
            self._q_rate_dev = wp.zeros(len(self._joint_q), dtype=wp.float32, device=dev)

            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

            if self.control_mode == "pd":
                # PD: SolverMuJoCo integrates the robot; SolverStyle3D only reads the
                # rigid bodies as colliders (it never writes body_q), so the co-sim
                # handoff is one-way. Gains/mode must be on the model before build.
                apply_pd_gains(self.model, self.robot)
                apply_robot_gravcomp(
                    self.model, exclude_bodies=[rb.name for rb in self.cfg.rigid_bodies])
                apply_gripper_contact_override(self.model, self.cfg)
                if self._gjs_wanted:
                    # R5-B: must run BEFORE SolverMuJoCo is constructed — it
                    # zeroes the finger DOFs' actuator gains on the model, and
                    # MuJoCo bakes gains into its actuators at build time.
                    self._gjs_setup(collision)
                self.robot_solver = make_robot_solver(self.model, self.cfg)
                self._pd_target = self.control.joint_target_q.numpy().copy()
                self._init_pd_feedforward()
                self._sync_pd_target(self._joint_q)

        # Free rigid bodies (cfg.rigid_bodies): name -> body index and the free
        # joint's joint_q offset, so tasks/randomization can read/write poses by
        # name (get_rigid_body_pose / set_rigid_body_pose).
        self.rigid_body_index: dict[str, int] = {}
        self._rigid_qstart: dict[str, int] = {}
        self._rigid_dstart: dict[str, int] = {}
        if self.cfg.rigid_bodies:
            joint_child = self.model.joint_child.numpy()
            joint_q_start = self.model.joint_q_start.numpy()
            joint_qd_start = self.model.joint_qd_start.numpy()
            for rb in self.cfg.rigid_bodies:
                bidx = resolve_body_index(self.model, rb.name)
                # the body's free joint = the (single) joint whose child it is
                jidx = int(np.nonzero(joint_child == bidx)[0][0])
                self.rigid_body_index[rb.name] = bidx
                self._rigid_qstart[rb.name] = int(joint_q_start[jidx])
                self._rigid_dstart[rb.name] = int(joint_qd_start[jidx])

        # Two-way cloth->rigid coupling (PD mode only; kinematic mode has no
        # rigid integrator). SolverStyle3D treats rigid bodies as read-only
        # colliders, so on its own a free rigid presses the cloth but is never
        # pushed back or carried by it. After each cloth substep we re-evaluate
        # the particle-body contact force at the converged positions and feed
        # the reaction wrench to SolverMuJoCo through state.body_f (applied on
        # the NEXT substep — one-substep lag at sim_dt is negligible). Robot
        # links are masked out: their PD tracking stays one-way as before.
        # With ``gripper_two_way_coupling`` the same wrench also goes to the
        # gripper finger links, which is NOT just "one more body": it closes the
        # control loop on the grasp. A position-controlled finger is otherwise
        # infinite-mass, so it drives to the commanded gap no matter what the
        # cloth does; fed the reaction it stalls where kp*(x - x_target) balances
        # the contact force, i.e. at the cloth's actual thickness.
        self._cloth_body_f = None
        self._rigid_body_enabled = None
        want_rigid = bool(self.cfg.rigid_bodies) and bool(self.cfg.rigid_two_way_coupling)
        want_gripper = bool(self.has_robot) and bool(self.cfg.gripper_two_way_coupling)
        if ((want_rigid or want_gripper) and self.control_mode == "pd"
                and getattr(self.cloth_solver, "collision", None) is not None):
            enabled = np.zeros(self.model.body_count, dtype=np.int32)
            if want_rigid:
                for bidx in self.rigid_body_index.values():
                    enabled[bidx] = 1
            n_finger = 0
            if want_gripper:
                for body in self._gripper_body_ids:
                    enabled[body] = 1
                    n_finger += 1
                if n_finger == 0:
                    raise RuntimeError(
                        "gripper_two_way_coupling is on but the robot asset's "
                        "gripper joints resolved no child bodies"
                    )
            self._rigid_body_enabled = wp.array(enabled, dtype=wp.int32, device=self.model.device)
            self._cloth_body_f = wp.zeros(
                self.model.body_count, dtype=wp.spatial_vector, device=self.model.device)
            print(f"[env] two-way cloth->rigid coupling on {int(enabled.sum())} bodies "
                  f"({n_finger} gripper fingers) "
                  f"(max_force={self.cfg.rigid_two_way_max_force:g} N)", flush=True)

        # Initial (≈flat, pre-fold) cloth snapshot, used by tasks as the coverage
        # reference (env contract: list, one entry per cloth).
        self.initial_cloth_states = [self.state_0.particle_q.numpy().copy()]

        if self.viewer is not None:
            self.viewer.show_triangles = True
            self.viewer.set_model(self.model)
            # Front view of the workspace (cloth lands ~(0.4,0,0.05)), matching the
            # off-screen `front` camera in render/bridge.py so gl and mp4 agree.
            self.viewer.set_camera(pos=wp.vec3(1.3, 0.0, 0.75), pitch=-38.0, yaw=180.0)

    def snapshot_runtime_physics(self) -> dict:
        """Return the runtime-tunable contact parameters for temporary stage edits."""

        snap = {
            "soft_contact_mu": float(self.cfg.cloth_params.soft_contact_mu),
            "soft_contact_margin": float(self.cfg.cloth_params.soft_contact_margin),
            "soft_contact_ke": float(self.cfg.cloth_params.soft_contact_ke),
            "soft_contact_kd": float(self.cfg.cloth_params.soft_contact_kd),
            "default_shape_mu": self.cfg.default_shape_mu,
            "gripper_shape_mu": self.cfg.gripper_shape_mu,
        }
        if hasattr(self.model, "shape_material_mu") and self.model.shape_material_mu is not None:
            snap["shape_material_mu"] = self.model.shape_material_mu.numpy().copy()
        return snap

    def set_runtime_physics(
        self,
        *,
        soft_contact_mu: float | None = None,
        soft_contact_margin: float | None = None,
        soft_contact_ke: float | None = None,
        soft_contact_kd: float | None = None,
        default_shape_mu: float | None = None,
        gripper_shape_mu: float | None = None,
    ) -> None:
        """Update contact/friction knobs without rebuilding the scene.

        This is intended for stage-local tuning. It only touches values that the
        solver/collision path reads at runtime; material stretch/bend/density
        still require rebuilding the model.
        """

        updates = {}
        for key, value in (
            ("soft_contact_mu", soft_contact_mu),
            ("soft_contact_margin", soft_contact_margin),
            ("soft_contact_ke", soft_contact_ke),
            ("soft_contact_kd", soft_contact_kd),
        ):
            if value is not None:
                updates[key] = float(value)
                if hasattr(self.model, key):
                    setattr(self.model, key, float(value))
        if updates:
            self.cfg.cloth_params = dataclasses.replace(self.cfg.cloth_params, **updates)
            if "soft_contact_margin" in updates:
                # keep the E4 band covered (see __init__)
                self._soft_contact_margin = max(
                    float(updates["soft_contact_margin"]), self._gripper_contact_offset)
            if "soft_contact_margin" in updates and hasattr(self.collision_pipeline, "soft_contact_margin"):
                try:
                    self.collision_pipeline.soft_contact_margin = float(updates["soft_contact_margin"])
                except Exception:
                    pass

        if default_shape_mu is not None:
            self.cfg.default_shape_mu = float(default_shape_mu)
        if gripper_shape_mu is not None:
            self.cfg.gripper_shape_mu = float(gripper_shape_mu)
        if default_shape_mu is not None or gripper_shape_mu is not None:
            apply_shape_material_overrides(self.model, self.cfg)
        self._invalidate_graph()

    def _invalidate_graph(self) -> None:
        """Drop any captured graph so the next frame re-captures.

        Contact knobs reach the kernels as plain Python scalars (the collision
        margin argument, ``model.soft_contact_*``), so a capture bakes their
        current values in. Any runtime change to them must invalidate it.
        """
        self._frame_graphs.clear()
        self._graph_warm = 0
        self._mj_graphs.clear()
        self._mj_graph_warm = 0

    def restore_runtime_physics(self, snapshot: dict) -> None:
        """Restore a snapshot produced by :meth:`snapshot_runtime_physics`."""

        contact_updates = {
            key: float(snapshot[key])
            for key in ("soft_contact_mu", "soft_contact_margin", "soft_contact_ke", "soft_contact_kd")
            if key in snapshot and snapshot[key] is not None
        }
        if contact_updates:
            self.cfg.cloth_params = dataclasses.replace(self.cfg.cloth_params, **contact_updates)
            if "soft_contact_margin" in contact_updates:
                self._soft_contact_margin = max(
                    float(contact_updates["soft_contact_margin"]), self._gripper_contact_offset)
            for key, value in contact_updates.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, float(value))
            if hasattr(self.collision_pipeline, "soft_contact_margin"):
                try:
                    self.collision_pipeline.soft_contact_margin = float(
                        self.cfg.cloth_params.soft_contact_margin
                    )
                except Exception:
                    pass

        self.cfg.default_shape_mu = snapshot.get("default_shape_mu")
        self.cfg.gripper_shape_mu = snapshot.get("gripper_shape_mu")
        if "shape_material_mu" in snapshot and hasattr(self.model, "shape_material_mu"):
            self.model.shape_material_mu.assign(snapshot["shape_material_mu"])
        else:
            apply_shape_material_overrides(self.model, self.cfg)
        self._invalidate_graph()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate(self):
        """Advance one viewer frame, replaying a captured graph when possible.

        The substep loop is the same kernel sequence every frame — only the
        contents of the buffers change — so it replays from a CUDA graph
        verbatim. That removes ~11k per-frame kernel launches from the critical
        path (the cloth solve alone issues ~990 per substep for kernels that
        finish in microseconds) and lets mjwarp's convergence loop run as a
        device-side conditional node instead of a per-iteration host readback.

        Everything that is NOT the same every frame stays outside the capture:
        the host->device refresh of the frame's FK endpoints below, the PD
        target sync in ``set_robot_control``, and the host bookkeeping at the
        end. Pinned-particle dragging (randomization / hanging-pull) still runs
        on the host per substep, so frames with live pins take the plain path;
        the captured graph contains no pin work, which is exactly what the plain
        path does when there are none.
        """
        kinematic_robot = self.has_robot and self.control_mode != "pd"
        if kinematic_robot:
            # Frame endpoints for the FK ramp (host targets may have been moved
            # by set_robot_control / set_rigid_body_pose since the last frame).
            self._q_start_dev.assign(self._joint_q_applied)
            self._q_end_dev.assign(self._joint_q)

        sim = self._simulate_pd if self.control_mode == "pd" else self._simulate_kinematic
        # Live pins only block the capture on the legacy host path (a per-substep
        # readback cannot be recorded).  The device path (_pin_lerp_kernel) is
        # just two more kernels in the loop, so it captures like anything else.
        pins_block_graph = bool(self._particle_pins) and self._pin_host
        if not self._use_cuda_graph:
            reason = ("cfg.cuda_graph is off" if not self.cfg.cuda_graph else
                      "disabled at runtime (gjs debug, or a capture failed)")
        elif pins_block_graph:
            # No pin COUNT in the string: the count wobbles frame to frame
            # while a randomisation drags pins, and _note_plain keys off the
            # text, so a count would turn one line into dozens.
            reason = ("live particle pin(s) on the HOST path "
                      "(DEFORMABLEBENCH_PIN_HOST=1): every substep reads the whole "
                      "particle array back, which cannot be captured")
        elif self._robot_graph_only:
            reason = "whole-frame capture unavailable; only the robot substep is graphed"
        else:
            reason = None
        self._note_plain(reason)
        key = (id(self.state_0), id(self.state_1))
        graph = None if self._robot_graph_only else self._frame_graphs.get(key)
        if not self._use_cuda_graph or pins_block_graph or self._robot_graph_only:
            sim()
        elif graph is not None:
            wp.capture_launch(graph)
            self._swap_after_replay()
        elif self._graph_warm < 2:
            # Warp module loads and the solvers' lazy allocations must happen
            # before the capture, not inside it.
            self._graph_warm += 1
            sim()
        else:
            try:
                with wp.ScopedCapture() as cap:
                    sim()
                # sim() already applied this frame's Python-level swaps; the
                # capture recorded the kernels but executed none of them.
                self._frame_graphs[key] = cap.graph
                wp.capture_launch(cap.graph)
                self._swap_after_replay()
            except Exception as e:  # noqa: BLE001
                print(f"[env] cuda_graph: whole-frame capture failed ({e}); "
                      "falling back to the robot substep only", flush=True)
                self._frame_graphs.clear()
                self._robot_graph_only = True
                sim()

        if kinematic_robot:
            self._joint_q_applied = self._joint_q.copy()
        if self._particle_pins and not self._pin_host:
            # Host half of _commit_pins.  It must live OUT here: a graph replay
            # runs no host code, so it cannot sit inside the substep function.
            for idx, target in self._particle_pins.items():
                self._particle_pin_applied[idx] = np.asarray(target, dtype=float).copy()

    def _note_plain(self, reason: str | None) -> None:
        """Log WHY a frame runs per-launch instead of replaying a graph.

        One line per change of reason, not per frame: the two states are sticky
        (a pin drag lasts hundreds of frames) so this is a handful of lines per
        episode.  It exists because the failure is otherwise INVISIBLE -- a
        single host readback anywhere inside the substep loop silently costs the
        whole frame its graph, and the only symptom is that the run is 2-4x
        slower (T27 B.7: 148 of flatten's 666 frames used to go this way, at
        4.2x the cost of a replayed frame, with nothing in the log to see it by).
        """
        if reason == self._plain_reason:
            return
        if reason is None:
            print("[env] cuda_graph: frames are replaying from a graph again "
                  f"(was: {self._plain_reason})", flush=True)
        else:
            print(f"[env] cuda_graph: frames run PLAIN -- {reason}", flush=True)
        self._plain_reason = reason

    def _swap_after_replay(self) -> None:
        """Re-apply the Python-level state swap a graph replay did not run.

        ``_simulate_kinematic`` / ``_simulate_pd`` swap ``state_0``/``state_1``
        once per substep.  Those are HOST statements: a captured graph holds the
        kernels, not the Python, so after a replay the attributes still point
        where they did at frame start.  An even substep count means the loop
        would have ended where it started (nothing to do); an odd count means one
        net swap.  Getting this wrong is silent -- the physics keeps running, it
        just reads the previous substep's buffer.
        """
        if self.sim_substeps % 2:
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _sync_pin_table(self) -> None:
        """Push the host pin table down to the arrays ``_pin_lerp_kernel`` reads.

        A captured graph bakes the launch dim and the array pointers, so a change
        of the pinned INDEX SET has to invalidate it; re-targeting an existing pin
        is an in-place ``assign`` and leaves the graph alone (that is the common
        case -- randomization re-targets the same one or two pins every frame for
        a few hundred frames)."""
        order = sorted(self._particle_pins)
        if order != self._pin_order:
            self._pin_order = order
            dev = self.model.device
            n = len(order)
            if n:
                self._pin_idx_dev = wp.array(np.asarray(order, dtype=np.int32),
                                             dtype=wp.int32, device=dev)
                self._pin_start_dev = wp.zeros(n, dtype=wp.vec3, device=dev)
                self._pin_target_dev = wp.zeros(n, dtype=wp.vec3, device=dev)
            else:
                self._pin_idx_dev = self._pin_start_dev = self._pin_target_dev = None
            self._invalidate_graph()
        if not order:
            return
        start = np.asarray([self._particle_pin_applied.get(i, self._particle_pins[i])
                            for i in order], dtype=np.float32)
        target = np.asarray([self._particle_pins[i] for i in order], dtype=np.float32)
        self._pin_start_dev.assign(start)
        self._pin_target_dev.assign(target)

    def set_particle_pin(self, idx: int, pos) -> None:
        """Kinematically pin particle ``idx``, targeting world ``pos`` this frame.

        Clears the particle's ``ParticleFlags.ACTIVE`` bit so SolverStyle3D skips
        integrating it (it stays wherever ``particle_q`` puts it, propagated through
        the state swap). ``pos`` is the TARGET for the coming frame; the substep
        loop lerps from where the pin currently sits to ``pos`` across substeps
        (``_apply_pins_substep``), so dragging a pin at frame rate stays smooth for
        stiff cloth. To drag, call again each frame with the next waypoint.
        (``inv_mass=0`` does NOT work: Style3D bakes mass at ``precompute()``.)"""

        idx = int(idx)
        if idx not in self._particle_pins:
            pf = self.model.particle_flags.numpy()
            pf[idx] &= ~int(newton.ParticleFlags.ACTIVE)
            self.model.particle_flags.assign(pf)
            # new pin: start the interpolation from where the particle is now.
            self._particle_pin_applied[idx] = self.state_0.particle_q.numpy()[idx].copy()
        self._particle_pins[idx] = np.asarray(pos, dtype=float)
        if not self._pin_host:
            self._sync_pin_table()

    def _apply_pins_substep(self, alpha_prev: float, alpha_curr: float) -> None:
        """Write each pin's substep-interpolated position/velocity into state_0.

        Called once per substep before the cloth solve, mirroring the robot's
        per-substep FK drive: position lerps applied->target across the frame,
        velocity = per-substep delta / sim_dt so cloth friction sees the pin
        actually moving (not teleporting)."""
        if not self._particle_pins:
            return
        if not self._pin_host:
            wp.launch(_pin_lerp_kernel, dim=len(self._pin_order),
                      inputs=[self._pin_idx_dev, self._pin_start_dev,
                              self._pin_target_dev, float(alpha_prev),
                              float(alpha_curr), 1.0 / self.sim_dt],
                      outputs=[self.state_0.particle_q, self.state_0.particle_qd],
                      device=self.model.device)
            return
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        for idx, target in self._particle_pins.items():
            start = self._particle_pin_applied.get(idx, target)
            p_prev = start + (target - start) * alpha_prev
            p_curr = start + (target - start) * alpha_curr
            q[idx] = p_curr
            qd[idx] = (p_curr - p_prev) / self.sim_dt
        self.state_0.particle_q.assign(q)
        self.state_0.particle_qd.assign(qd)

    def _commit_pins(self) -> None:
        """Frame end: the target is now the applied position (next frame lerps
        from here) and pin velocity is zeroed so a held pin doesn't drift."""
        if not self._particle_pins:
            return
        if not self._pin_host:
            # Device half only; `applied <- target` is done by _simulate, which
            # is the level a graph replay still executes on the host.
            wp.launch(_pin_zero_vel_kernel, dim=len(self._pin_order),
                      inputs=[self._pin_idx_dev],
                      outputs=[self.state_0.particle_qd], device=self.model.device)
            return
        qd = self.state_0.particle_qd.numpy()
        for idx, target in self._particle_pins.items():
            self._particle_pin_applied[idx] = np.asarray(target, dtype=float).copy()
            qd[idx] = 0.0
        self.state_0.particle_qd.assign(qd)

    def clear_particle_pins(self) -> None:
        """Release all pins: restore ACTIVE and zero their velocity (no fly-off)."""
        if not self._particle_pins:
            return
        pf = self.model.particle_flags.numpy()
        qd = self.state_0.particle_qd.numpy()
        for idx in self._particle_pins:
            pf[idx] |= int(newton.ParticleFlags.ACTIVE)
            qd[idx] = 0.0
        self.model.particle_flags.assign(pf)
        self.state_0.particle_qd.assign(qd)
        self._particle_pins = {}
        self._particle_pin_applied = {}
        if not self._pin_host:
            self._sync_pin_table()   # empties the device table + invalidates

    def _simulate_kinematic(self):
        """``sim_substeps`` cloth solves with kinematic robot FK.

        Collide PER substep (not once per frame): stiff cloth (e.g. silk) moves far
        within a frame, and stale contacts let it tunnel through the robot/itself and
        explode.

        SolverStyle3D's external-rigid path reads ``state_in.body_q`` as the
        previous rigid pose and ``state_out.body_q`` as the current pose when
        computing body-particle friction.  Keep that ordering intact; otherwise
        the cloth can show normal contact while losing the tangential motion that
        carries it with the gripper.
        """

        margin = float(self._soft_contact_margin)
        n_coord = 0
        if self.has_robot:
            n_coord = len(self._joint_q)
            # The ramp's joint velocity is constant across the frame's substeps
            # (linear ramp, equal substeps), so derive it once.
            wp.launch(_frame_rate_kernel, dim=n_coord,
                      inputs=[self._q_start_dev, self._q_end_dev, 1.0 / self.frame_dt],
                      outputs=[self._q_rate_dev], device=self.model.device)
            wp.copy(self.model.joint_qd, self._q_rate_dev)

        for substep in range(self.sim_substeps):
            alpha_prev = substep / self.sim_substeps
            alpha_curr = (substep + 1) / self.sim_substeps
            if self.has_robot:
                wp.launch(_lerp_dofs_kernel, dim=n_coord,
                          inputs=[self._q_start_dev, self._q_end_dev, alpha_prev],
                          outputs=[self.model.joint_q, self.state_0.joint_q],
                          device=self.model.device)
                wp.copy(self.state_0.joint_qd, self._q_rate_dev)
                newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

                wp.launch(_lerp_dofs_kernel, dim=n_coord,
                          inputs=[self._q_start_dev, self._q_end_dev, alpha_curr],
                          outputs=[self.model.joint_q, self.state_1.joint_q],
                          device=self.model.device)
                wp.copy(self.state_1.joint_qd, self._q_rate_dev)
                newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
                self.model.body_qd.assign(self.state_1.body_qd)

            self.state_0.clear_forces()
            if self._viewer_injects_forces:
                self.viewer.apply_forces(self.state_0)
            self._apply_pins_substep(alpha_prev, alpha_curr)  # substep-smooth pin drag
            self.collision_pipeline.collide(self.state_0, self.contacts, soft_contact_margin=margin)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            if self._contact_projection:
                self.cloth_solver.collision.project_contacts(
                    particle_q=self.state_0.particle_q,
                    particle_qd=self.state_0.particle_qd,
                    particle_q_prev=self.cloth_solver.x_prev,
                    contacts=self.contacts,
                    body_q=self.state_0.body_q,
                    dt=self.sim_dt,
                    **self._projection_prev_pose(),
                )

        self._commit_pins()
        if self.has_robot:
            # Land exactly on the frame target (the host copy of the applied
            # pose is advanced by _simulate, outside any capture).
            wp.copy(self.model.joint_q, self._q_end_dev)
            self.model.joint_qd.zero_()
            self.model.body_qd.zero_()
            wp.copy(self.state_0.joint_q, self._q_end_dev)
            self.state_0.joint_qd.zero_()
            self.state_0.body_qd.zero_()

    def _gjs_setup(self, collision) -> None:
        """R5-B: wire the finger prismatic DOFs into the cloth solver.

        Resolves everything by NAME from the robot asset's declared gripper
        joints (robot-agnostic, no body names): joint coord/DOF indices, child
        (blade) bodies, their collision shapes, the prismatic axis expressed in
        the child body frame, and the PD parameters the closed-form 1D solve
        uses. Then disables the MuJoCo actuator on those DOFs (zero target
        gains) so the drive is not applied twice — the PD force lives in the
        cloth solve and the converged q/qd are written back each substep.
        Must run before SolverMuJoCo is constructed (gains are baked at build).
        """
        from newton import JointType

        names = tuple(self.robot.left_gripper) + tuple(self.robot.right_gripper)
        by = {_basename(lbl): i for i, lbl in enumerate(self.model.joint_label) if lbl}
        missing = [n for n in names if n not in by]
        if missing:
            raise KeyError(
                f"gripper_joint_in_solver: joints not in model.joint_label: {missing}")
        joints = [by[n] for n in names]
        jtype = self.model.joint_type.numpy()
        for name, j in zip(names, joints):
            if int(jtype[j]) != int(JointType.PRISMATIC):
                raise ValueError(
                    f"gripper_joint_in_solver: joint {name!r} is not prismatic")
        qstart = self.model.joint_q_start.numpy()
        dstart = self.model.joint_qd_start.numpy()
        child = self.model.joint_child.numpy()
        coord = [int(qstart[j]) for j in joints]
        dof = [int(dstart[j]) for j in joints]
        bodies = [int(child[j]) for j in joints]

        # Prismatic axis in the CHILD body frame. FK is
        # X_wc = X_wp * X_pj * t(q*axis) * X_cj^{-1}, so dp_child/dq =
        # R(X_wp*X_pj) @ axis = R_wc @ (R_cj @ axis): the child-frame vector
        # R_cj @ axis is constant and the world axis is just the child rotation
        # applied to it (a prismatic joint never rotates its child).
        axis_np = self.model.joint_axis.numpy()
        xcj = self.model.joint_X_c.numpy()
        axis_child = [
            self._quat_rotate(xcj[j][3:7], np.asarray(axis_np[d], dtype=float))
            for j, d in zip(joints, dof)
        ]

        shape_body = self.model.shape_body.numpy()
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        shape_flags = self.model.shape_flags.numpy()
        finger_shapes = [
            [s for s in range(len(shape_body))
             if int(shape_body[s]) == b and int(shape_flags[s]) & collide_particles]
            for b in bodies
        ]
        body_mass = self.model.body_mass.numpy()
        lo = self.model.joint_limit_lower.numpy()
        hi = self.model.joint_limit_upper.numpy()
        n = len(names)
        collision.enable_gripper_joint_solve(
            joint_coord=coord,
            joint_dof=dof,
            child_bodies=bodies,
            finger_shapes=finger_shapes,
            axis_child=axis_child,
            kp=[float(self.robot.gripper_kp)] * n,
            kd=[float(self.robot.gripper_kd)] * n,
            effort=[float(self.robot.gripper_effort)] * n,
            mass=[float(body_mass[b]) for b in bodies],
            limit_lo=[float(lo[d]) for d in dof],
            limit_hi=[float(hi[d]) for d in dof],
        )
        # Disable the MuJoCo actuator on these DOFs (zero P and D gains). The
        # joint's passive terms (MJCF armature/damping) stay: the 1D solve
        # treats MuJoCo's passive integration as its inertia target.
        ke = self.model.joint_target_ke.numpy()
        kd = self.model.joint_target_kd.numpy()
        for d in dof:
            ke[d] = 0.0
            kd[d] = 0.0
        self.model.joint_target_ke = wp.array(
            ke, dtype=self.model.joint_target_ke.dtype, device=self.model.device)
        self.model.joint_target_kd = wp.array(
            kd, dtype=self.model.joint_target_kd.dtype, device=self.model.device)
        self._gjs_finger_bodies_wp = wp.array(
            np.asarray(bodies, dtype=np.int32), dtype=wp.int32, device=self.model.device)
        self._gjs_enabled = True
        if self._gjs_debug:
            # Per-substep host readbacks are incompatible with graph capture.
            self._use_cuda_graph = False
        print(f"[env] gripper_joint_in_solver: {n} finger DOFs moved into the "
              f"cloth solve (coord={coord}, bodies={bodies}); MuJoCo actuator "
              f"gains zeroed on those DOFs", flush=True)

    def _simulate_pd(self):
        """PD co-sim: each substep step the robot (SolverMuJoCo) then the cloth.

        SolverStyle3D treats rigid bodies as read-only colliders and never writes
        ``body_q``; SolverMuJoCo integrates them into ``state_1``.  Do not copy the
        current robot pose back into ``state_0`` before the cloth solve: Style3D
        uses ``state_0.body_q`` as previous pose and ``state_1.body_q`` as current
        pose for body-particle friction, matching the Franka Style3D example.
        """

        margin = float(self._soft_contact_margin)
        for substep in range(self.sim_substeps):
            alpha_prev = substep / self.sim_substeps
            alpha_curr = (substep + 1) / self.sim_substeps
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if self._cloth_body_f is not None:
                # reaction wrench from the previous substep's cloth solve;
                # SolverMuJoCo maps state_in.body_f -> xfrc_applied
                self.state_0.body_f.assign(self._cloth_body_f)
                if self._gjs_enabled:
                    # R5-B: the finger force balance is solved inside the cloth
                    # solve — feeding the finger reaction as xfrc too would
                    # double-count it. Only the FEED is masked; _cloth_body_f
                    # keeps accumulating so ③b telemetry stays readable.
                    wp.launch(
                        _zero_selected_body_f_kernel,
                        dim=len(self._gjs_finger_bodies_wp),
                        inputs=[self._gjs_finger_bodies_wp],
                        outputs=[self.state_0.body_f],
                        device=self.model.device,
                    )
            if self._viewer_injects_forces:
                self.viewer.apply_forces(self.state_0)
            self._apply_pd_feedforward(alpha_curr)

            # 1. robot dynamics: state_0 -> state_1 (body_q updated, particles untouched)
            self._robot_step()
            # 2. cloth vs previous->current robot motion. The contact list is
            # built from state_0, as in the upstream Franka Style3D co-sim path;
            # per-substep robot motion is small, and the solver evaluates contact
            # forces against state_1.body_q.
            self._apply_pins_substep(alpha_prev, alpha_curr)  # substep-smooth pin drag
            self.collision_pipeline.collide(self.state_0, self.contacts, soft_contact_margin=margin)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            if self._gjs_enabled and self._gjs_debug:
                # R5-B unit check: per-substep finger q [mm] and the contact
                # generalized force -S1 [N] along each finger's joint axis at
                # the last iterate. Host readback — _gjs_setup turned graphs off.
                col = self.cloth_solver.collision
                _q = col.gjs_q.numpy() * 1000.0
                _f = -col.gjs_s1.numpy()
                _t = col.gjs_q_inertia.numpy() * 1000.0
                print(f"[gjs] t={self.sim_time:.3f} sub={substep} "
                      f"q_mm={np.array2string(_q, precision=3)} "
                      f"q_inertia_mm={np.array2string(_t, precision=3)} "
                      f"F_contact_N={np.array2string(_f, precision=3)}", flush=True)

            reaction_pre = (
                self._contact_projection
                and bool(self.cfg.contact_projection_reaction_pre)
                and self._cloth_body_f is not None
            )
            if reaction_pre:
                # Lambda-proxy ordering: the reaction reads the solver's
                # CONVERGED depth (ke x depth = the force the implicit solve
                # actually balanced against the jaw) BEFORE the projection
                # trims positions. The force channel no longer depends on the
                # projection residual, so slack can go to ~0. Interleaved mode
                # projects inside the solver loop, so state_0 is already
                # trimmed and the snapshot the sweep kept is the only place
                # that depth still exists.
                pre_q = self.state_0.particle_q
                if self.cfg.contact_projection_interleaved:
                    pre_q = self.cloth_solver.collision.proj_pre_q
                self._cloth_body_f.zero_()
                self.cloth_solver.collision.accumulate_body_reaction(
                    dt=self.sim_dt,
                    particle_q_prev=self.cloth_solver.x_prev,
                    particle_q=pre_q,
                    contacts=self.contacts,
                    body_q=self.state_0.body_q,
                    body_q_prev=self.state_1.body_q,
                    body_enabled=self._rigid_body_enabled,
                    body_f=self._cloth_body_f,
                    max_force=float(self.cfg.rigid_two_way_max_force),
                )
            if self._contact_projection:
                # Bound the residual penetration. Historical (reaction_pre off)
                # order runs this BEFORE the reaction, which then reads
                # depth <= slack — the residual is what keeps the penalty
                # force channel alive in that mode.
                self.cloth_solver.collision.project_contacts(
                    particle_q=self.state_0.particle_q,
                    particle_qd=self.state_0.particle_qd,
                    particle_q_prev=self.cloth_solver.x_prev,
                    contacts=self.contacts,
                    body_q=self.state_0.body_q,
                    dt=self.sim_dt,
                    **self._projection_prev_pose(),
                )
            if self._cloth_body_f is not None:
                # post-swap: state_0 = new state, state_1 = previous. The cloth
                # solve updates state_in.particle_q in place, so the true
                # pre-step positions live in cloth_solver.x_prev.
                if not reaction_pre:
                    self._cloth_body_f.zero_()
                    self.cloth_solver.collision.accumulate_body_reaction(
                        dt=self.sim_dt,
                        particle_q_prev=self.cloth_solver.x_prev,
                        particle_q=self.state_0.particle_q,
                        contacts=self.contacts,
                        body_q=self.state_0.body_q,
                        body_q_prev=self.state_1.body_q,
                        body_enabled=self._rigid_body_enabled,
                        body_f=self._cloth_body_f,
                        max_force=float(self.cfg.rigid_two_way_max_force),
                    )
                if (self.cfg.tri_sdf_compliant
                        and hasattr(self.cloth_solver.collision, "accumulate_tri_sdf_reaction")):
                    # Equal-and-opposite of the compliant triangle contact. Runs
                    # after accumulate_body_reaction (which zeroes body_f) so both
                    # channels land in the same buffer.
                    self.cloth_solver.collision.accumulate_tri_sdf_reaction(
                        particle_q=self.state_0.particle_q,
                        body_q=self.state_0.body_q,
                        body_com=self.model.body_com,
                        body_enabled=self._rigid_body_enabled,
                        body_f=self._cloth_body_f,
                    )
                if self._projection_impulse:
                    # Isaac-style: the position solve itself carries the
                    # reaction, so the projection no longer needs to leave
                    # residual penetration behind to keep a penalty force alive.
                    self.cloth_solver.collision.accumulate_projection_impulse(
                        contacts=self.contacts,
                        body_q=self.state_0.body_q,
                        body_com=self.model.body_com,
                        body_enabled=self._rigid_body_enabled,
                        body_f=self._cloth_body_f,
                        dt=self.sim_dt,
                        max_force=float(self.cfg.rigid_two_way_max_force),
                    )

        self._commit_pins()

    def _robot_step(self) -> None:
        """One robot substep, replayed from a CUDA graph when enabled.

        SolverMuJoCo issues ~110 kernel launches per substep for this robot
        (nv=16, and the cloth contacts never reach MuJoCo, so nacon=0): every
        kernel is 16 threads wide and the launch cost dwarfs the arithmetic.
        Worse, mjwarp's convergence loop is a ``wp.capture_while``, which
        OUTSIDE a graph degrades to a device->host readback plus a stream sync
        per iteration — draining a queue that also holds ~1k cloth kernels, so
        the CPU can never run ahead and every launch lands on the critical path.
        Measured: 31 ms/frame, i.e. the whole of PD mode's +19% over kinematic.
        Captured, the launches collapse into one replay and the readback becomes
        a device-side conditional node.

        Two graphs, keyed by ``(state_0, state_1)``: the substep loop swaps the
        two states, so the pair alternates between two fixed combinations.
        """
        args = (self.state_0, self.state_1, self.control, None, self.sim_dt)
        if not self._use_cuda_graph or not self._robot_graph_only:
            # Either graphs are off, or the whole frame is already being
            # captured/replayed by _simulate — capturing here would nest.
            self.robot_solver.step(*args)
            return

        key = (id(self.state_0), id(self.state_1))
        graph = self._mj_graphs.get(key)
        if graph is None:
            if self._mj_graph_warm < 2:
                # Let the first substeps run plain: warp module loads and the
                # solver's lazy allocations must not happen inside a capture.
                self._mj_graph_warm += 1
                self.robot_solver.step(*args)
                return
            try:
                with wp.ScopedCapture() as cap:
                    self.robot_solver.step(*args)
                graph = self._mj_graphs[key] = cap.graph
            except Exception as e:  # noqa: BLE001  (driver/mempool refusal -> stay correct)
                print(f"[env] cuda_graph: robot capture failed ({e}); plain steps",
                      flush=True)
                self._use_cuda_graph = False
                self.robot_solver.step(*args)
                return
            # capture records without executing: run the substep we just recorded
        wp.capture_launch(graph)

    # ------------------------------------------------------------------
    # Example-compatible interface (drives the web bridge)
    # ------------------------------------------------------------------
    def step(self, q_l=None, q_r=None, g_l=None, g_r=None):
        """Advance one frame. With robot args, apply them first (SimExecutor calls
        step(q_l,q_r,g_l,g_r); the web bridge calls step() with no args)."""

        if self.has_robot and q_l is not None:
            self.set_robot_control(q_l, q_r, g_l, g_r)
        self._simulate()
        self.sim_time += self.frame_dt
        if self.control_mode == "pd":
            self._commit_pd_target()

    # ------------------------------------------------------------------
    # Robot (kinematic) interface
    # ------------------------------------------------------------------
    def set_robot_control(self, q_l, q_r, g_l=None, g_r=None):
        """Set the kinematic joint target (arms + grippers).

        GRIPPER CONVENTION (single source of truth for the whole repo): ``g`` in
        [0,1] is the OPEN fraction — ``1`` = fully open (fingers apart), ``0`` =
        fully closed (fingers together). The piper finger joints open at +lim/-lim
        and close at 0, so the joint target is ``g*lim``. This matches the ported
        skills/tasks (close=0.015, open=0.7), the robot init presets
        (gripper_open=1.0), and the real dual_piper ``gripper.pos`` — so no
        conversion is needed anywhere else."""

        jq = self._joint_vector(q_l, q_r, g_l, g_r)
        if self.control_mode == "pd":
            # PD: command is the desired joint position (SolverMuJoCo tracks it).
            self._sync_pd_target(jq)
        else:
            # kinematic: the joint target IS the pose (eval_fk in _simulate).
            self.model.joint_q.assign(jq)
        self._joint_q = jq

    def _joint_vector(self, q_l, q_r, g_l=None, g_r=None):
        """Arms + gripper open-fractions -> a full ``joint_q``-shaped vector."""
        jq = self._joint_q.copy()
        jq[self.idx_l_arm] = np.asarray(q_l, dtype=float)[: len(self.idx_l_arm)]
        jq[self.idx_r_arm] = np.asarray(q_r, dtype=float)[: len(self.idx_r_arm)]
        lim = self.robot.gripper_limit
        if g_l is not None:
            jq[self.idx_l_grip[0]] = float(g_l) * lim
            jq[self.idx_l_grip[1]] = -float(g_l) * lim
        if g_r is not None:
            jq[self.idx_r_grip[0]] = float(g_r) * lim
            jq[self.idx_r_grip[1]] = -float(g_r) * lim
        return jq

    def set_robot_pose(self, q_l, q_r, g_l=None, g_r=None) -> None:
        """Place the robot AT a joint pose (episode init), not a servo command.

        ``set_robot_control`` only moves the PD *target*, so in PD mode the arm
        has to slew to the episode's start pose — and because the settle loop
        steps with no command, both PD command channels stay frozen on that one
        jump (the per-substep target ramp replays 0 -> start pose every frame,
        and the sticky velocity feedforward sticks at the +-clamp). Measured on
        ``flatten_tshirt``: the arm parked 25 deg / 182 mm off the start pose
        with the jaw at 58% of its commanded opening, and the fold phase then
        latched that pose as the command, so the whole episode ran from it.

        deformable_bench writes ``data.qpos`` directly in ``reset()`` for this
        exact reason (and clears its sticky velFF alongside). This is the same
        thing through Newton's state: ``SolverMuJoCo`` re-reads
        ``state_in.joint_q/qd`` every step (``update_data_interval=1``), so
        writing the state IS a teleport. Both modes end up bit-identical at the
        start pose, which is what makes PD-vs-kinematic comparisons meaningful.
        """

        if not self.has_robot:
            return
        jq = self._joint_vector(q_l, q_r, g_l, g_r)
        self.model.joint_q.assign(jq)
        self._joint_q = jq
        # No FK ramp to run: the arm is already there (kinematic path).
        self._joint_q_applied = jq.copy()
        self._q_start_dev.assign(self._joint_q_applied)
        self._q_end_dev.assign(self._joint_q)

        dofs = np.r_[self.idx_l_arm_d, self.idx_r_arm_d,
                     self.idx_l_grip_d, self.idx_r_grip_d]
        for st in (self.state_0, self.state_1):
            st.joint_q.assign(jq)
            qd = st.joint_qd.numpy()
            qd[dofs] = 0.0
            st.joint_qd.assign(qd)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, st)

        if self.control_mode == "pd":
            # Twice: the first call leaves prev = the OLD target (so qd_des is
            # the jump we just teleported away); the second makes prev == target
            # -> flat ramp, zero velocity feedforward, nothing stale left.
            self._sync_pd_target(jq)
            self._sync_pd_target(jq)

    def _commit_pd_target(self) -> None:
        """Frame end: a command applies to ONE frame.

        ``_sync_pd_target`` only advances ``_pd_target_prev`` when a new command
        arrives, and ``_apply_pd_feedforward`` interpolates prev -> target across
        every substep. A frame stepped WITHOUT a command (the settle loop, the
        web bridge, any hold that passes no joints) would therefore replay the
        previous command's ramp — the servo chases a 50 Hz sawtooth whose mean is
        0.55x the target — and keep its stale velocity feedforward. Committing
        the endpoint here makes "no new command" mean "hold the last target",
        which is a no-op when the executor commands every frame."""

        self._pd_target_prev[:] = self._pd_target
        self._pd_target_prev_dev.assign(self._pd_target_prev)
        if self._pd_target_qd.any():
            self._pd_target_qd[:] = 0.0
            self.control.joint_target_qd.assign(self._pd_target_qd)

    def _sync_pd_target(self, jq) -> None:
        """Copy the robot's coord-space targets into the DOF-space PD target array
        (1-DOF joints: same scalar, different index once a free joint exists), and
        derive the velocity feedforward from the frame-to-frame target delta."""
        t = self._pd_target
        self._pd_target_prev[:] = t              # frame start, for the substep ramp
        t[self.idx_l_arm_d] = jq[self.idx_l_arm]
        t[self.idx_r_arm_d] = jq[self.idx_r_arm]
        t[self.idx_l_grip_d] = jq[self.idx_l_grip]
        t[self.idx_r_grip_d] = jq[self.idx_r_grip]
        # Sticky velocity feedforward: qd_des = clamp(dq_des/dt) across
        # consecutive control frames, on the ARM DOFs only (the reference
        # leaves the grippers on pure -kd damping, and a jaw command is a
        # step, not a trajectory). The substep ramp in _apply_pd_feedforward
        # makes this the exact slope of the position target the servo chases.
        arm_d = self._pd_ff_arm_dofs
        c = float(self.cfg.pd_qd_des_clamp)
        self._pd_target_qd[arm_d] = np.clip(
            (t[arm_d] - self._pd_target_prev[arm_d]) / self.frame_dt, -c, c)
        self.control.joint_target_qd.assign(self._pd_target_qd)
        self.control.joint_target_q.assign(t)
        # Device mirrors of the frame's ramp endpoints, refreshed here (once per
        # frame, outside any capture) so _apply_pd_feedforward can interpolate
        # on the device; see _simulate.
        self._pd_target_prev_dev.assign(self._pd_target_prev)
        self._pd_target_dev.assign(t)

    # ------------------------------------------------------------------
    # PD feedforward (deformable_bench's full PD law, always on in PD mode)
    # ------------------------------------------------------------------
    def _init_pd_feedforward(self) -> None:
        """Wire the two feedforward channels SolverMuJoCo exposes.

        * ``control.joint_target_qd`` -> the velocity actuator installed by
          ``JointTargetMode.POSITION_VELOCITY`` (``kd*(qd_des - qd)``).
        * ``control.joint_f`` -> MuJoCo ``qfrc_applied`` (a raw joint torque),
          used for the Coriolis bias. Needs the Newton-DOF -> MuJoCo-DOF map,
          which the solver already built as ``mjc_dof_to_newton_dof``.
        """
        s = self.robot_solver
        if getattr(s, "mjw_data", None) is None or s.mjc_dof_to_newton_dof is None:
            raise RuntimeError(
                "PD feedforward needs SolverMuJoCo's MJWarp data (use_mujoco_cpu=False)")
        if int(s.mjw_data.nworld) != 1:
            raise RuntimeError("PD feedforward assumes a single world")

        m2n = s.mjc_dof_to_newton_dof.numpy()[0]
        n2m = np.full(self.model.joint_dof_count, -1, dtype=np.int32)
        for mj_i, n_i in enumerate(m2n):
            if n_i >= 0:
                n2m[int(n_i)] = mj_i
        arm_dofs = np.concatenate([self.idx_l_arm_d, self.idx_r_arm_d]).astype(np.int32)
        ff_dofs = np.concatenate([
            arm_dofs, self.idx_l_grip_d, self.idx_r_grip_d]).astype(np.int32)
        mj_dofs = n2m[ff_dofs]
        if (mj_dofs < 0).any():
            raise RuntimeError("PD feedforward: robot DOF missing from the MuJoCo model")

        self._pd_ff_arm_dofs = arm_dofs
        self._pd_ff_newton_dof = wp.array(ff_dofs, dtype=wp.int32, device=self.model.device)
        self._pd_ff_mj_dof = wp.array(mj_dofs.astype(np.int32), dtype=wp.int32,
                                      device=self.model.device)
        self._pd_target_prev = self._pd_target.copy()
        self._pd_target_qd = self.control.joint_target_qd.numpy().copy()
        self._pd_target_qd[:] = 0.0
        self.control.joint_target_qd.assign(self._pd_target_qd)
        # Device mirrors of the ramp endpoints (written per frame by
        # _sync_pd_target, read per substep by _apply_pd_feedforward).
        self._pd_target_dev = wp.array(self._pd_target, dtype=wp.float32,
                                       device=self.model.device)
        self._pd_target_prev_dev = wp.array(self._pd_target_prev, dtype=wp.float32,
                                            device=self.model.device)

    def _apply_pd_feedforward(self, alpha: float) -> None:
        """Per-substep half of the feedforward: ramp the position target and
        refresh the Coriolis torque.

        The reference runs its PD once per physics step against a per-step
        trajectory sample, so its position target is a ramp; holding this env's
        50 Hz command across all 10 substeps instead injects a sawtooth of a full
        command delta into the position error. ``alpha`` is the substep's
        fraction of the frame, matching the kinematic path's FK interpolation.

        ``qfrc_bias``/``qfrc_gravcomp`` are the values MuJoCo computed in the
        previous substep's forward pass (one ``sim_dt`` = 2 ms of lag; both are
        zero before the first step, so the first substep gets no bias term).
        """
        wp.launch(
            _lerp_dofs_kernel1,
            dim=self._pd_target_dev.shape[0],
            inputs=[self._pd_target_prev_dev, self._pd_target_dev, alpha],
            outputs=[self.control.joint_target_q],
            device=self.model.device,
        )
        d = self.robot_solver.mjw_data
        wp.launch(
            _pd_coriolis_ff_kernel,
            dim=self._pd_ff_newton_dof.shape[0],
            inputs=[d.qfrc_bias, d.qfrc_gravcomp,
                    self._pd_ff_newton_dof, self._pd_ff_mj_dof],
            outputs=[self.control.joint_f],
            device=self.model.device,
        )

    @staticmethod
    def _quat_rotate(q_xyzw, v):
        x, y, z, w = q_xyzw
        u = np.array([x, y, z])
        return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)

    def _eef_pose(self, body_idx):
        t = self.state_0.body_q.numpy()[body_idx]  # [px,py,pz, qx,qy,qz,qw]
        q_xyzw = t[3:7]
        pos = t[:3] + self._quat_rotate(q_xyzw, np.array([0.0, 0.0, self.robot.eef_offset_z]))
        return np.array([pos[0], pos[1], pos[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    def _actual_q(self):
        """Joint positions actually realized. In kinematic mode the target IS the
        pose; in PD mode the robot tracks the target, so read it back from state."""

        return self.state_0.joint_q.numpy() if self.control_mode == "pd" else self._joint_q

    @property
    def left_qpos(self):
        return self._actual_q()[self.idx_l_arm].copy()

    @property
    def right_qpos(self):
        return self._actual_q()[self.idx_r_arm].copy()

    @property
    def left_gripper_val(self):
        # open fraction in [0,1]: joint/lim (1 = open, 0 = closed). See set_robot_control.
        return float(self._actual_q()[self.idx_l_grip[0]] / self.robot.gripper_limit)

    @property
    def right_gripper_val(self):
        return float(self._actual_q()[self.idx_r_grip[0]] / self.robot.gripper_limit)

    @property
    def left_eef_pose(self):
        return self._eef_pose(self.eef_l_body)

    @property
    def right_eef_pose(self):
        return self._eef_pose(self.eef_r_body)

    def gripper_finger_positions(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """World positions of the two finger bodies of one gripper.

        ``side`` is "left" or "right". Bodies come from the child links of the
        gripper joints declared by the robot asset."""

        bodies = self._gripper_body_ids_by_side[side]
        if len(bodies) != 2:
            raise RuntimeError(
                f"gripper_finger_positions requires two gripper joints for {side}, "
                f"got {len(bodies)}")
        i, j = bodies
        bq = self.state_0.body_q.numpy()
        return bq[i][:3].copy(), bq[j][:3].copy()

    @property
    def cloth_obj_path(self):
        return self.cfg.asset

    @property
    def cloth_base_pos(self) -> np.ndarray:
        """Randomized cloth spawn position (world frame), [x, y, z].

        The bench applies cloth_init pos_offset at build time (scene_config), so
        cloth_pos_xy / start_height already carry the per-episode randomization.
        Tasks that place world targets relative to the bag (bag-strap grasp) read
        this instead of the deformable_bench randomizer internals."""
        return np.array(
            [float(self.cfg.cloth_pos_xy[0]), float(self.cfg.cloth_pos_xy[1]),
             float(self.cfg.start_height)], dtype=np.float64)

    @property
    def cloth_keypoints(self):
        """Named keypoint vertex indices from the cloth asset_meta (single source).
        e.g. {'corner_LU': 1124, ...}. Tasks should read these instead of hardcoding."""
        return {k: v["index"] if isinstance(v, dict) else v
                for k, v in self.cfg.cloth_keypoints.items()}

    @property
    def home_q_left(self):
        return np.asarray(self.robot.home_left, dtype=float)

    @property
    def home_q_right(self):
        return np.asarray(self.robot.home_right, dtype=float)

    def get_unoccluded_cloth_indices(self):
        """All cloth vertex indices. newton has no occlusion concept -> everything
        is 'visible' (stub for tasks that filter grasp candidates by visibility)."""
        return np.arange(self.model.particle_count)

    @property
    def left_base(self):
        return self.state_0.body_q.numpy()[self.base_l_body][:3].copy()

    @property
    def right_base(self):
        return self.state_0.body_q.numpy()[self.base_r_body][:3].copy()

    # ------------------------------------------------------------------
    # Free rigid bodies (cfg.rigid_bodies)
    # ------------------------------------------------------------------
    def get_rigid_body_pose(self, name: str) -> np.ndarray:
        """World pose of a free rigid body as ``[px,py,pz, qx,qy,qz,qw]``.

        Reads ``state_0.body_q`` — in PD mode that is SolverMuJoCo's integrated
        pose, in kinematic mode the eval_fk pose (the body just stays put)."""

        return self.state_0.body_q.numpy()[self.rigid_body_index[name]].copy()

    def set_rigid_body_pose(self, name: str, pos, quat_xyzw=None) -> None:
        """Teleport a free rigid body (episode init / randomization, not physics).

        Writes the free joint coords everywhere both solvers read them (model +
        both states + the env's host-side targets) and zeroes its velocity, then
        re-runs FK so ``body_q`` matches immediately."""

        qs = self._rigid_qstart[name]
        ds = self._rigid_dstart[name]
        coords = np.concatenate([
            np.asarray(pos, dtype=float).reshape(3),
            (np.asarray(quat_xyzw, dtype=float).reshape(4) if quat_xyzw is not None
             else self.get_rigid_body_pose(name)[3:7]),
        ])

        jq = self.model.joint_q.numpy()
        jq[qs:qs + 7] = coords
        self.model.joint_q.assign(jq)
        jqd = self.model.joint_qd.numpy()
        jqd[ds:ds + 6] = 0.0
        self.model.joint_qd.assign(jqd)
        for st in (self.state_0, self.state_1):
            sq = st.joint_q.numpy(); sq[qs:qs + 7] = coords; st.joint_q.assign(sq)
            sd = st.joint_qd.numpy(); sd[ds:ds + 6] = 0.0; st.joint_qd.assign(sd)
        if self.has_robot:
            # host-side copies so a later set_robot_control doesn't restore the
            # old free-joint coords (PD targets are DOF-space and robot-only, so
            # they need no update here).
            self._joint_q[qs:qs + 7] = coords
            self._joint_q_applied[qs:qs + 7] = coords
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

    def get_observation(self, include_visuals: bool = False):
        """Proprioception (+ cloth truth) as a bench ObservationMessage.

        The contract bench tasks/skills expect (``BaseTask``/``RobotSkills``).
        ``include_visuals`` (RGB/depth/pcd) lands with the Kit render bridge (P3);
        cloth ground-truth points are always available from the sim."""

        from deformablebench.sim.types import ObservationMessage

        cloth = self.get_sim_vertices()
        return ObservationMessage(
            joint_left=self.left_qpos, joint_right=self.right_qpos,
            eef_pose_left=self.left_eef_pose, eef_pose_right=self.right_eef_pose,
            gripper_left=self.left_gripper_val, gripper_right=self.right_gripper_val,
            pcd_cloth_truth=cloth,
        )

    def get_gripper_collision_controller(self, config: dict | None = None):
        """Return the optional runtime gripper/cloth collision trick controller.

        This mirrors the old benchmark API consumed by ``SimExecutor`` and teleop.
        The controller is created lazily so its previous-gripper state starts from
        the scene's current gripper values, not from a hard-coded open/closed pose.
        """

        if not self.has_robot:
            return None
        if self._gripper_collision_controller is None:
            cfg = GripperCollisionTrickConfig(**config) if config else GripperCollisionTrickConfig()
            self._gripper_collision_controller = NewtonGripperCollisionController(
                self.model,
                config=cfg,
                initial_left_gripper=self.left_gripper_val,
                initial_right_gripper=self.right_gripper_val,
            )
        return self._gripper_collision_controller

    def render(self):
        """Log the current state into the attached viewer (e.g. ViewerWeb)."""

        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Bench thin interface (consumed by SimExecutor / success / metric)
    # ------------------------------------------------------------------
    def reset(self):
        # Full reset (re-randomize, re-pose) lands in P1; P0 has no episode loop.
        raise NotImplementedError("DeformableEnv.reset is implemented in P1")

    def get_current_sim_time(self) -> float:
        return self.sim_time

    def get_sim_vertices(self) -> np.ndarray:
        """Cloth particle positions in world space, shape (N, 3)."""

        return self.state_0.particle_q.numpy()

    def get_cloth_position(self) -> np.ndarray:
        """Alias of get_sim_vertices used by the bench pipeline (SuccessChecker)."""

        return self.get_sim_vertices()

    def step_to_time(self, t: float):
        """Advance the simulation until ``sim_time`` reaches ``t``."""

        while self.sim_time < t:
            self.step()
