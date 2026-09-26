"""Scene construction: assets -> ``newton.Model``.

The runtime mesh parsing and Style3D cloth setup live in
``fabricbench.sim.style3d_asset`` so benchmark code does not import private
Newton example helpers. The actual physics still uses the local Newton fork's
``SolverStyle3D``/``SolverMuJoCo`` APIs.

P0 builds cloth only (no robot). The robot (``add_mjcf`` + ``SolverMuJoCo``) is
added in P2.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field

import numpy as np

from fabricbench.sim.style3d_asset import (
    NewtonClothParams,
    apply_contact_params,
    analyze_mesh,
    import_newton,
    load_mesh,
    project_panel_vertices,
    repair_panel_winding,
    resolve_path,
)

# Bag asset shipped with the Newton fork (relative to the newton package root).
DEFAULT_BAG_ASSET = "newton/examples/assets/style3d_probe/bag/nonwoven_small_6/nonwoven_small_6.obj"


def demo_cloth_params() -> NewtonClothParams:
    """Cloth physics matching the bag_drop demo's argparse defaults."""

    return NewtonClothParams(
        density=0.3, particle_radius=0.0015, tri_ka=500.0, tri_kd=1.0e-4,
        tri_aniso_ke=(1000.0, 1000.0, 500.0), edge_aniso_ke=(2.0e-4, 2.0e-4, 2.0e-4),
        edge_kd=1.0e-4, soft_contact_margin=0.0015, soft_contact_ke=10.0,
        soft_contact_kd=1.0e-6, soft_contact_mu=0.2, solver_iterations=5,
        linear_iterations=10, step_dt=1.0 / 120.0, style3d_self_collision=True,
        style3d_collision_radius=0.0005, style3d_collision_stiff_vf=0.01,
        style3d_collision_stiff_ee=0.005, style3d_collision_stiff_ef=0.05,
    )


def cloth_params_from_physics(physics: dict) -> NewtonClothParams:
    """Map an asset's ``asset_meta.yaml`` physics block to NewtonClothParams.

    Open newton ``SolverStyle3D`` == Style3D family, so params map 1:1:
    stretch_stiff->tri_aniso_ke (weft,warp,shear), bend_stiff->edge_aniso_ke,
    density (mass/area in panel space), friction->soft_contact_mu. Non-physical
    solver knobs keep demo defaults. Empty physics -> demo params unchanged.
    """

    base = demo_cloth_params()
    if not physics:
        return base
    ov: dict = {}
    if "density" in physics:
        ov["density"] = float(physics["density"])
    if "stretch_stiff" in physics:
        ov["tri_aniso_ke"] = tuple(float(x) for x in physics["stretch_stiff"])
    if "bend_stiff" in physics:
        ov["edge_aniso_ke"] = tuple(float(x) for x in physics["bend_stiff"])
    mu = physics.get("gripper_static_friction", physics.get("friction"))
    if mu is not None:
        ov["soft_contact_mu"] = float(mu)
    if "thickness" in physics:
        ov["particle_radius"] = float(physics["thickness"]) / 2.0
    # Solver iteration counts (non-physical accuracy knobs). Stiffer/heavier
    # cloth (the bag) needs more Newton iterations to converge per step.
    if "solver_iterations" in physics:
        ov["solver_iterations"] = int(physics["solver_iterations"])
    if "linear_iterations" in physics:
        ov["linear_iterations"] = int(physics["linear_iterations"])
    return dataclasses.replace(base, **ov)


@dataclass
class RobotConfig:
    """Parsed robot_meta.yaml (one robot = one asset dir). PIPER-compatible attr
    names so the env consumes it directly. Produced by bench RobotAsset.from_dir."""

    mjcf: str                       # absolute path to sim MJCF
    urdf: str                       # absolute path to IK URDF (matched gripper!)
    base_link: str
    ee_link: str
    left_arm: tuple[str, ...]
    right_arm: tuple[str, ...]
    left_gripper: tuple[str, ...]   # [main, follower]
    right_gripper: tuple[str, ...]
    eef_left: str
    eef_right: str
    base_left: str
    base_right: str
    left_base_pose: tuple[float, ...]
    right_base_pose: tuple[float, ...]
    gripper_limit: float = 0.035
    eef_offset_z: float = 0.1358
    arm_kp: tuple[float, ...] = (80.0, 80.0, 80.0, 40.0, 10.0, 10.0)
    arm_kd: tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 1.5, 1.5)
    gripper_kp: float = 400.0
    gripper_kd: float = 5.0
    # Max |actuator force| on each gripper DOF [N] (MuJoCo ``actfrcrange``, which
    # clamps the P+D sum). 0 keeps the model's imported limit (effectively
    # unbounded), so the force ceiling is kp x overshoot. A finite value is the
    # Isaac-style compliant finger: a full-close command stalls on the cloth at
    # the motor rating instead of driving a stiff jaw through it.
    gripper_effort: float = 0.0
    home_left: tuple[float, ...] = (0.0,) * 6
    home_right: tuple[float, ...] = (0.0,) * 6
    velocity_limits: float = 5.0
    acceleration_limits: float = 5.0


@dataclass
class RigidBodyConfig:
    """One free rigid body injected into the scene (freejoint + box collider).

    Geometry (``box_center``/``box_half``) is in the body frame; ``pos``/
    ``quat_wxyz`` place the body in the world at build time. In PD mode
    SolverMuJoCo integrates the free joint (gravity + contacts vs robot/ground);
    SolverStyle3D additionally sees the box as a one-way cloth collider.
    """

    name: str                                    # newton body key (task-facing)
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.05
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_diag: tuple[float, float, float] = (1.0e-5, 1.0e-5, 1.0e-5)
    box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_half: tuple[float, float, float] = (0.01, 0.01, 0.01)
    mu: float = 0.8
    # Optional raw MuJoCo contact params for the collider (PD mode only).
    # Default None = the ShapeConfig ke/kd conversion (ke=1e5, kd=1000 ->
    # solref timeconst 0.002, matching deformable_bench's base scene XML).
    # deformable_bench's <pair> override (-50000, -1000 + hard solimp) NaNs
    # mujoco_warp at 500 Hz, so it is NOT the default here.
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float, float, float] | None = None


@dataclass
class SceneConfig:
    """Everything needed to build and step one deformable scene."""

    asset: str = DEFAULT_BAG_ASSET
    scale: float | None = None
    device: str | None = None
    warp_cache_dir: str | None = None   # None -> warp default (~/.cache/warp), persistent
    start_height: float = 0.5
    cloth_pos_xy: tuple[float, float] = (0.0, 0.0)
    cloth_rot_z: float = 0.0   # deg; rigid Z-spin of the initial cloth placement (not panel rest)
    cloth_keypoints: dict = field(default_factory=dict)  # from asset_meta (single source)
    view_fps: float = 60.0
    view_substeps: int = 6
    robot: "RobotConfig | None" = None   # None = cloth-only
    rigid_bodies: tuple["RigidBodyConfig", ...] = ()  # free rigid objects (PD mode)
    control_mode: str = "kinematic"      # "kinematic" | "pd"
    # PD mode runs deformable_bench's full PD law
    # ``tau = kp*(q_des-q) + kd*(qd_des-qd) + qfrc_bias``:
    #   * the position target ramps across the frame's substeps instead of
    #     stepping once per frame (deformable_bench runs its PD at the physics
    #     rate, so its target is a per-step ramp, not a 50 Hz staircase);
    #   * ``qd_des`` = clamped frame-to-frame difference of the arm target
    #     ("sticky" velocity feedforward), carried by a second VELOCITY actuator;
    #   * the Coriolis/centrifugal part of ``qfrc_bias`` goes in via
    #     ``control.joint_f`` -> MuJoCo ``qfrc_applied``. Only the Coriolis part:
    #     gravity is already cancelled exactly by gravcomp, and feeding the full
    #     bias on top of that would double-compensate it.
    # Position-only servoing left the arm 17 deg behind on a fling and cost the
    # fold/flatten tasks their whip; see DeformableEnv._apply_pd_feedforward.
    pd_qd_des_clamp: float = 5.0   # [rad/s] saturation on qd_des (deformable_bench default)
    # Replay the robot substep from a CUDA graph instead of re-issuing it (PD
    # mode only). Pure throughput: SolverMuJoCo fires ~110 kernel launches per
    # substep for a 16-DOF arm pair whose kernels are 16 threads wide, and
    # mjwarp's convergence loop degrades to a per-iteration device->host readback
    # outside a graph. See DeformableEnv._robot_step.
    cuda_graph: bool = True
    # Two-way cloth->rigid coupling (PD mode): feed the cloth contact reaction
    # wrench back to free rigid bodies via SolverMuJoCo's state.body_f, so e.g.
    # a snickers bar is carried by the bag it sits in. Robot links stay one-way.
    rigid_two_way_coupling: bool = True
    # R10(1)/R13 gate 0 (2026-08-28): retired.  The 20 N default was an
    # explosion fuse from the era when the two-way channel first went live:
    # under deep penetration a ke*depth spike fed through xfrc in one tick
    # kicked the fingers.  It breaks Newton's third law one-sidedly (the
    # cloth's force is not scaled, the body's is), so a clamped finger
    # presses DEEPER than the reaction it feels.  Two measurements retire it:
    # the linear ke=500 stack never trips it (R5-A 0/5580 frames, peak 17 N)
    # while R9's material-law arms hit exactly 20.00 N on 8/8 runs, i.e.
    # every reading there was truncated.  The force band plus a smooth force
    # law removed the spike at the source.  Kept as a parameter for
    # emergencies; 0 = off.
    rigid_two_way_max_force: float = 0.0  # per-body clamp [N] on fed-back force (0 = off)
    # Two-way cloth->gripper coupling (PD mode): also feed the reaction wrench to
    # the gripper finger links, so a position-controlled finger STALLS on the
    # cloth at its true thickness instead of driving through it. Turns
    # ``grasp_close_val`` from a hand-tuned geometric gap into a grip force
    # (kp x overshoot), which is what makes the grasp thickness-agnostic.
    # Needs a meaningful ``soft_contact_ke``: at ke=10 the finger must bury a
    # full particle_radius before it feels 0.3 N, i.e. it penetrates anyway.
    gripper_two_way_coupling: bool = False
    # Particle-contact stiffness for shapes attached to gripper-joint bodies.
    # Zero inherits the global cloth contact stiffness.
    gripper_contact_ke: float = 0.0
    # E4: penalty RANGE BAND [m] on the gripper shapes (PhysX contact_offset).
    # The stock law f = ke*(r-d) reads particle_radius as both the geometric
    # probe radius and the force band, so retiring the shell to the cloth's
    # physical half-thickness collapses the normal load and mu*N with it
    # (15.4 N at r=8mm -> 0.8 N at r=0.5mm, doc GRIPPER-CONTACT 5(25)). This
    # gives the gripper its own band while the geometric stop stays wherever the
    # geometry channel puts it (particle_radius / tri_sdf_half_thickness). It
    # must be LARGER than that stop or the overlap is zero and the jaw becomes a
    # frictionless wall. 0 = off, stock behaviour, bit-identical default path.
    gripper_contact_offset: float = 0.0
    # R9: hardening compression law on the gripper band (doc GRIPPER-CONTACT 2.2
    # `p(eps) = k0*eps/(1 - eps/eps_max)`).  The stock band law is LINEAR, so the
    # only handle on the normal load is the band width -- and a wide band is what
    # rakes cloth into the jaw (R3(b): jaw stop = 1.60*offset, gathered material
    # = 169*offset + 1844 mm^2).  Hardening decouples the two: narrow band for a
    # tight stop, hardening for the newtons.  eps_max is the strain at the
    # incompressible core, as a FRACTION OF THAT SHAPE'S BAND, so d_max =
    # eps_max*band bounds penetration by construction.  Gripper shapes only (an
    # asymptote on the table's 0.5 mm band would make lying on it a wall).
    # 0 = off, stock linear law, default path bit-identical.
    gripper_contact_hardening: float = 0.0
    # T17: cap on the tangent stiffness the hardening law hands the EXPLICITLY
    # coupled finger.  f = k*d/den has slope k/den^2, up to 400x the linear k at
    # the core; the finger sees that spring integrated with forward Euler (one
    # reaction per substep, held for the whole MuJoCo step), so it is bound by
    # K*dt^2/m < 4.  This is that bound as a coefficient: the fork gets
    # k_max = gripper_contact_hardening_kcap * m_body / dt_substep^2, i.e. 1.0 =
    # a QUARTER of the explicit limit (3/4 margin).  0 = off (no cap array set,
    # no env var, default path unchanged).
    gripper_contact_hardening_kcap: float = 0.0
    # T8: true static friction on the tri-SDF channel.  The stock law
    # (`compute_projected_isotropic_friction`, shared with the vertex channel)
    # is IPC-regularised Coulomb: correct while sliding, but with NO sticking
    # state -- any load below the cone creeps at v = eps*(1-sqrt(1-rho)), and
    # eps here is 1e-2 m/s, an order above IPC's own.  Measured consequence:
    # pack's left hand drifts 19-34 mm over the 5 s B5 carry at mu_eff 0.70 and
    # loses the strap in 2/3 runs.  With a ratio > 0 the channel switches to a
    # tangential anchor (spring to the shape-local material point, capped by
    # mu*N with a return mapping), which sticks exactly below the cone.
    # kt = ratio * k_tri; pick the ratio from the pre-slip you can accept:
    # slip_max = mu * depth / ratio, so 50-100 um at mu 0.7 / depth 0.3 mm
    # means ratio ~ 2-4.  0 = off, stock law, bit-identical default path.
    sdf_anchor_kt_ratio: float = 0.0
    # R5-B: solve the prismatic finger joint coordinates INSIDE the cloth
    # solver's nonlinear loop (closed-form 1D per finger per iteration:
    # inertia + effort-clamped PD + this finger's contact penalty), and write
    # q/qd back to SolverMuJoCo each substep. Removes the one-substep force
    # lag and the "infinite-mass finger during the cloth solve" of the xfrc
    # co-sim (doc GRIPPER-CONTACT R5-B). PD mode only. The finger bodies then
    # no longer receive the xfrc reaction (their MuJoCo actuator is disabled
    # too — the balance lives in the cloth solve), so the arm no longer feels
    # the finger-mount reaction. False = off, default path bit-identical.
    gripper_joint_in_solver: bool = False
    # Symmetric rigid-mesh vertex/edge contact against cloth faces/edges.  The
    # solver API is shape-generic; the bench selects collision shapes from the
    # robot asset's declared gripper joints without body names or pairing.
    rigid_feature_contact: bool = False
    # R2-3 rigid untangling (ICM, Volino 2006) against the gripper triangles.
    # A recovery force for cloth that is ALREADY threaded through a finger --
    # the proximity channels can only push to the nearest surface, which for a
    # blade thinner than the penetration depth is the wrong side. 0/False = off,
    # nothing allocated, default path unchanged.
    rigid_untangling: bool = False
    # Multiplies the cloth's own PD diagonal, same meaning as the cloth-cloth
    # ``style3d_collision_stiff_ef``. It is the correction-per-iteration knob:
    # the unit scene launches the cloth at ~1.9 m/s at 1.0 (doc R2-3).
    rigid_untangling_stiff: float = 0.02
    # Displacement scale [m]; the per-iteration correction is 2x this. Negative
    # takes the cloth-cloth default (2 x style3d_collision_radius).
    rigid_untangling_thickness: float = -1.0
    # Max rigid triangles examined per cloth edge. A finger mesh is far finer
    # than the cloth, so a cloth edge's AABB overlaps many gripper triangles and
    # the default 32 truncates the list before the crossed one is reached
    # (measured: 905153 truncations in one episode on the r05 arm).
    rigid_untangling_capacity: int = 32
    # Post-solve position-level non-penetration pass (PBD-style projection of
    # cloth out of rigid shapes down to a residual depth of
    # ``contact_projection_slack``). Penalty forces stay the reaction channel;
    # requires two-way coupling for the jaw to stop at the cloth stack.
    contact_projection: bool = False
    contact_projection_slack: float = 2.0e-3
    contact_projection_iterations: int = 3
    # Evaluate the two-way reaction BEFORE the projection pass, i.e. on the
    # solver's converged state with its real penetration depth (ke x depth =
    # the lambda-proxy for the position constraint). Lets slack go to ~0
    # without starving the force channel; default False keeps the historical
    # order (reaction reads the post-projection residual = slack).
    contact_projection_reaction_pre: bool = False
    # S2 (XPBD position-level friction): resolve Coulomb friction inside the
    # projection loop instead of leaving it to the penalty channel. Multiplies
    # the per-shape mu; 0 keeps the pass normal-only.
    contact_projection_friction: float = 0.0
    # S1 (persistent anchor + cone return mapping): tangential stiffness as a
    # fraction of the contact normal stiffness. 0 keeps the stock viscous
    # friction, whose per-substep history reset is what causes creep.
    friction_anchor_kt_ratio: float = 0.0
    # S3 (AVBD tangential multipliers): dual-ascent step as a fraction of the
    # contact normal stiffness. Sets convergence speed only; the multiplier
    # saturates on the Coulomb cone regardless. 0 disables.
    friction_avbd_dual_k: float = 0.0
    # Feed the position projection back to the rigid side as an impulse
    # (m*dx/dt^2) instead of relying on residual penetration to produce a
    # penalty force. This is what lets contact_projection_slack go to zero.
    contact_projection_impulse: bool = False
    # Run the projection inside the cloth solver's nonlinear loop (one sweep per
    # iteration) instead of as a post-solve pass. This is the PBD arrangement:
    # elasticity and contact alternate to a self-consistent state. See §7.2e.
    contact_projection_interleaved: bool = False
    # Limit the position projection to the gripper shapes. It exists to stop the
    # jaw penetrating cloth; running it against the table as well makes the cloth
    # resist the sliding it needs to spread out when flung.
    contact_projection_gripper_only: bool = False
    # E3 triangle-level SDF contact (default OFF; default path bit-identical).
    # Constrains min_{x in tri} SDF_gripper(x) >= tri_sdf_half_thickness, so
    # detection stops depending on particle_radius covering the mesh aperture.
    tri_sdf_contact: bool = False
    tri_sdf_half_thickness: float = 5.0e-4
    tri_sdf_voxel: float = 1.0e-3
    tri_sdf_pad: float = 6.0e-3
    tri_sdf_refine_steps: int = 3
    tri_sdf_max_correction: float = 5.0e-3
    # E3(b): contact stiffness from the cloth material instead of a constant.
    # k_i = modulus * A_i / thickness (A_i = rest tributary area). 0 = off, keeps
    # the per-shape gripper_contact_ke and the bit-identical default path.
    # E3(a): carry the triangle constraint as a compliant PENALTY (force into the
    # implicit solve + equal-and-opposite on the body) instead of a position
    # projection. The projection form removes the very overlap the penalty needs,
    # so the two compete and the reaction collapses once the radius is retired.
    tri_sdf_compliant: bool = False
    # Compliant mode switches the stock per-particle penalty off on the gripper
    # shapes so the normal load is not counted twice. That also removes the only
    # gripper-cloth FRICTION channel (the stock law carries mu); set this False
    # to keep the stock contact alive as a diagnostic and separate the two.
    tri_sdf_disable_stock_penalty: bool = True
    contact_material_modulus: float = 0.0          # E_t [Pa]
    # Cloth-vs-table (ground PLANE) penalty stiffness [N/m], applied to the
    # plane shapes only. 0 = leave newton's stock value. Promoted from the T2
    # probe's PROBE_TABLE_KE env var to a real key (T10-B): the stop-surface
    # recipe needs a hard table on both tasks, and an env var is not a recipe.
    table_contact_ke: float = 0.0
    contact_material_thickness: float = 1.0e-3     # t0 [m], match the asset meta
    style3d_panel_axes: str = "xy"
    solver: str = "style3d"
    broad_phase: str = "sap"
    # Rescale each UV island so panel rest lengths match the 3D mesh. Needed for
    # generative assets whose UVs are a packed texture atlas, not a garment
    # pattern (set from asset_meta `mesh_processing.normalize_panel_islands`).
    normalize_panel_islands: bool = False
    # Path to a precomputed physics-only 2D panel (.npz from
    # tools/reparametrize_panel.py). Style3D reads UVs as the cloth's rest
    # pattern, but a generative atlas is a jigsaw of hundreds of mismatched
    # islands. When set, this replaces the panel fed to the solver while the
    # OBJ's own vt (and therefore every texture binding) is left alone.
    physics_panel: str | None = None
    cloth_params: NewtonClothParams = field(default_factory=demo_cloth_params)
    translation_preconditioner: bool = False
    default_shape_mu: float | None = None
    gripper_shape_mu: float | None = None


def _normalize_panel_units(mesh) -> None:
    """Fix a gross unit mismatch between the Style3D panel (UV) and the 3D mesh.

    Style3D uses the panel/UV coordinates as the *rest metric* for anisotropic
    stretch, so the panel must be in the same units as the 3D mesh. Some assets
    ship UVs in millimetres while the mesh is in metres (e.g. silk_35: panel
    350x350 vs mesh 0.35x0.35), a 1000x mismatch that makes the solver see a
    massively compressed rest state and explode on frame 0.

    We only correct *gross* mismatches (>30x), snapping the correction to the
    nearest power of ten, so a genuine flattened-pattern scale (O(1) larger than
    the 3D projection, as for real garments/bags) is left untouched.
    """

    pv = getattr(mesh, "panel_vertices", None)
    if pv is None:
        return
    v3 = mesh.vertices
    size3d = float(np.linalg.norm(v3[:, :2].max(0) - v3[:, :2].min(0)))
    sizep = float(np.linalg.norm(pv.max(0) - pv.min(0)))
    if size3d <= 0.0 or sizep <= 0.0:
        return
    ratio = sizep / size3d
    if ratio <= 30.0:
        return
    scale = 10.0 ** round(math.log10(ratio))
    mesh.panel_vertices = pv / scale
    if getattr(mesh, "style3d_panel_vertices", None) is not None:
        mesh.style3d_panel_vertices = mesh.style3d_panel_vertices / scale
    print(f"[builder] normalized panel units: panel/3D ratio {ratio:.1f} -> divided panel by {scale:g}", flush=True)


def _apply_physics_panel(mesh, npz_path: str) -> None:
    """Swap in a precomputed rest panel (see tools/reparametrize_panel.py).

    The npz stores the flattening against xatlas' own vertex/face indexing, so
    geometry and panel are replaced together to keep them corner-for-corner
    consistent. Rendering is unaffected: it reads the OBJ file, not this copy.
    """
    d = np.load(npz_path)
    pv, pf, f = d["panel_vertices"], d["panel_faces"], d["faces"]
    if len(f) != len(mesh.faces):
        # xatlas re-splits vertices along seams, so the face count should match
        # even though the vertex count grows; a mismatch means the npz is stale.
        raise ValueError(
            f"physics_panel {npz_path} has {len(f)} faces but the mesh has "
            f"{len(mesh.faces)} — regenerate it for this OBJ")
    mesh.panel_vertices = pv.astype(np.float64)
    mesh.panel_faces = pf.astype(np.int64)
    if getattr(mesh, "style3d_panel_vertices", None) is not None:
        uv_by_vertex = np.zeros(len(mesh.vertices), dtype=np.int64)
        uv_by_vertex[np.asarray(mesh.faces).reshape(-1)] = pf.reshape(-1)
        mesh.style3d_panel_vertices = pv[uv_by_vertex]
    print(f"[builder] physics panel <- {os.path.basename(npz_path)} "
          f"({len(pv)} panel verts)", flush=True)


def _normalize_panel_islands(mesh) -> None:
    """Rescale each UV island so panel (rest) edge lengths match the 3D mesh.

    Style3D uses the panel/UV coordinates as the anisotropic-stretch rest
    metric. Generative image-to-3D assets (EmbodiedGen) carry a *packed texture
    atlas* as UVs: per-island scale is arbitrary (normalized to fit the atlas),
    so the solver sees a wildly wrong rest state and the cloth violently
    reconfigures on frame 0 (observed 4-60 m/s kicks) and crumples mid-air.

    Fix: for every UV-connected island, scale its panel coords (about the
    island centroid) by the median of |3D edge| / |panel edge|. In-island
    conformal distortion remains, but the dominant per-island scale error is
    gone. Only the in-memory arrays change — the OBJ file and texture mapping
    are untouched.
    """

    pv = getattr(mesh, "panel_vertices", None)
    pf = getattr(mesh, "panel_faces", None)
    if pv is None or pf is None:
        return
    pv = np.asarray(pv, dtype=np.float64).copy()
    pf = np.asarray(pf, dtype=np.int64)
    v3 = np.asarray(mesh.vertices, dtype=np.float64)
    f3 = np.asarray(mesh.faces, dtype=np.int64)
    if len(pf) != len(f3):
        print("[builder] normalize_panel_islands skipped: panel/3D face count mismatch", flush=True)
        return

    # union-find over panel vertex indices -> UV islands
    parent = np.arange(len(pv))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in pf:
        r0 = find(int(tri[0]))
        for t in (int(tri[1]), int(tri[2])):
            rt = find(t)
            if rt != r0:
                parent[rt] = r0
    roots = np.fromiter((find(i) for i in range(len(pv))), dtype=np.int64, count=len(pv))

    # per-edge 3D vs panel lengths (pf[i] corresponds to f3[i] corner-for-corner)
    e3_all, ep_all, er_all = [], [], []
    for k in range(3):
        j = (k + 1) % 3
        e3_all.append(np.linalg.norm(v3[f3[:, k]] - v3[f3[:, j]], axis=1))
        ep_all.append(np.linalg.norm(pv[pf[:, k]] - pv[pf[:, j]], axis=1))
        er_all.append(roots[pf[:, k]])
    e3 = np.concatenate(e3_all)
    ep = np.concatenate(ep_all)
    er = np.concatenate(er_all)
    valid = ep > 1e-12

    n_islands = 0
    ratios = []
    for r in np.unique(er[valid]):
        m = valid & (er == r)
        s = float(np.median(e3[m] / ep[m]))
        if not np.isfinite(s) or s <= 0.0:
            continue
        idx = np.where(roots == r)[0]
        c = pv[idx].mean(axis=0)
        pv[idx] = c + (pv[idx] - c) * s
        n_islands += 1
        ratios.append(s)
    mesh.panel_vertices = pv

    # keep the per-vertex expansion (if present) in sync via the face mapping
    if getattr(mesh, "style3d_panel_vertices", None) is not None:
        uv_by_vertex = np.zeros(len(v3), dtype=np.int64)
        uv_by_vertex[f3.reshape(-1)] = pf.reshape(-1)
        mesh.style3d_panel_vertices = pv[uv_by_vertex]

    if ratios:
        print(
            f"[builder] normalized {n_islands} UV islands to 3D rest scale "
            f"(median island scale {np.median(ratios):.3g}, "
            f"range [{min(ratios):.3g}, {max(ratios):.3g}])",
            flush=True,
        )


def build_cloth_scene(cfg: SceneConfig):
    """Build a finalized cloth ``newton.Model`` and its initial ``State``.

    Returns ``(model, state_0)``.
    """

    asset_path = resolve_path(cfg.asset)
    mesh = load_mesh(asset_path)
    _normalize_panel_units(mesh)
    if cfg.physics_panel:
        _apply_physics_panel(mesh, cfg.physics_panel)
    elif cfg.normalize_panel_islands:
        _normalize_panel_islands(mesh)
    report = analyze_mesh(asset_path, mesh)
    if report.status != "ok":
        raise ValueError(f"Cannot load cloth asset {asset_path}: {report.warnings}")

    scale = float(cfg.scale) if cfg.scale is not None else float(report.recommended_scale or 1.0)

    newton, _wp = import_newton(warp_cache_dir=cfg.warp_cache_dir, device=cfg.device)
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    _add_cloth_to_builder(builder, mesh, cfg, scale)
    builder.add_ground_plane()
    model = builder.finalize()
    apply_contact_params(model, cfg.cloth_params)
    return model, model.state()


def _add_cloth_to_builder(builder, mesh, cfg: SceneConfig, scale: float) -> None:
    """Add a Style3D cloth into a shared builder."""

    import warp as wp
    from newton.solvers import style3d

    cp = cfg.cloth_params
    verts_np = mesh.vertices
    if abs(float(cfg.cloth_rot_z)) > 1e-9:
        # Rigid spin of the initial placement about the OBJ ORIGIN (0,0), matching
        # deformable_bench's flexcomp path: base_quat sets the MuJoCo cloth-body
        # quat, so the mesh rotates about the body origin (== the OBJ origin, which
        # base_pos anchors). Rotating about the vertex centroid instead swings
        # asymmetric assets whose origin sits at an edge — a 90 deg base_quat on the
        # tshirt (origin at the hem) lands it ~1 m out of reach. Symmetric silk has
        # origin == centroid, so it is unaffected either way. (rest metric / panel
        # UVs untouched: this only re-orients where the sheet lands.)
        verts_np = verts_np.copy()
        th = math.radians(float(cfg.cloth_rot_z))
        c, s = math.cos(th), math.sin(th)
        x0, y0 = verts_np[:, 0].copy(), verts_np[:, 1].copy()
        verts_np[:, 0] = c * x0 - s * y0
        verts_np[:, 1] = s * x0 + c * y0
    faces_np = mesh.faces
    panel_v = getattr(mesh, "panel_vertices", None)
    panel_f = getattr(mesh, "panel_faces", None)
    if panel_v is not None and panel_f is not None:
        faces_np, panel_f = repair_panel_winding(faces_np, panel_f, panel_v, enabled=True)
        panel_verts = [wp.vec2(float(a), float(b)) for a, b in panel_v]
        panel_indices = panel_f.astype(np.int32).reshape(-1).tolist()
    else:
        panel_verts = project_panel_vertices(wp, verts_np, cfg.style3d_panel_axes)
        panel_indices = faces_np.astype(np.int32).reshape(-1).tolist()

    vertices = [wp.vec3(float(x), float(y), float(z)) for x, y, z in verts_np]
    flat_idx = faces_np.astype(np.int32).reshape(-1).tolist()
    pos_z = float(cfg.start_height) - float(np.min(verts_np[:, 2])) * float(scale)
    # Anchor the OBJ ORIGIN at cloth_pos_xy (deformable_bench's convention:
    # world = obj*scale + spawn_pos). Symmetric sheets (silk) are modelled
    # centred, so origin == centroid and this is a no-op for them; asymmetric
    # assets (bags, tshirt: centroid 0.1-1.4 m off origin) MUST anchor by origin
    # or the whole mesh — and the grasp offsets tuned against it — shift.
    cx, cy = (float(v) for v in cfg.cloth_pos_xy)
    pos_x, pos_y = cx, cy
    style3d.add_cloth_mesh(
        builder,
        vertices=vertices,
        indices=flat_idx,
        rot=wp.quat_identity(),
        pos=wp.vec3(pos_x, pos_y, pos_z),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=float(cp.density),
        scale=float(scale),
        tri_ka=float(cp.tri_ka),
        tri_kd=float(cp.tri_kd),
        particle_radius=float(cp.particle_radius),
        panel_verts=panel_verts,
        panel_indices=panel_indices,
        tri_aniso_ke=wp.vec3(*map(float, cp.tri_aniso_ke)),
        edge_aniso_ke=wp.vec3(*map(float, cp.edge_aniso_ke)),
        edge_kd=float(cp.edge_kd),
    )


def _add_rigid_body(builder, rb: RigidBodyConfig) -> None:
    """Inject one free rigid body + box collider.

    ``add_body`` here is the fork's convenience method: link + free joint + its
    own articulation, with the body xform copied into ``joint_q`` so solvers and
    ``eval_fk`` agree on the initial pose."""

    import warp as wp
    import newton

    qw, qx, qy, qz = (float(v) for v in rb.quat_wxyz)
    ixx, iyy, izz = (float(v) for v in rb.inertia_diag)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*(float(v) for v in rb.pos)), wp.quat(qx, qy, qz, qw)),
        mass=float(rb.mass),
        com=wp.vec3(*(float(v) for v in rb.com)),
        inertia=wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
        label=rb.name,
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=float(rb.mu))
    # Raw MuJoCo solref/solimp (stiff pinchable contact). Only meaningful in PD
    # mode, where SolverMuJoCo.register_custom_attributes has registered them.
    shape_attrs = {}
    if rb.solref is not None and "mujoco:solref" in builder.custom_attributes:
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

        shape_attrs["mujoco:solref"] = wp.vec2(*(float(v) for v in rb.solref))
        shape_attrs["mujoco:solref_mode"] = SOLREF_MODE_RAW
        if rb.solimp is not None:
            shape_attrs["mujoco:geom_solimp"] = tuple(float(v) for v in rb.solimp)
        # higher priority -> MuJoCo uses THIS geom's solref/solimp/friction for
        # the whole contact pair instead of solmix-averaging with the finger's.
        shape_attrs["mujoco:geom_priority"] = 1
    builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(*(float(v) for v in rb.box_center)), wp.quat_identity()),
        hx=float(rb.box_half[0]), hy=float(rb.box_half[1]), hz=float(rb.box_half[2]),
        cfg=shape_cfg,
        label=f"{rb.name}_geom",
        custom_attributes=shape_attrs or None,
    )


def build_robot_cloth_scene(cfg: SceneConfig):
    """Build robot (add_mjcf) + cloth in ONE shared builder. Returns (model, state)."""

    import newton

    import_newton(warp_cache_dir=cfg.warp_cache_dir, device=cfg.device)

    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    if cfg.control_mode == "pd":
        # PD mode runs the robot under SolverMuJoCo; its per-DOF target gain/mode
        # attributes (joint_target_ke/kd/mode) must exist on the builder. The env
        # fills the actual gains by name after finalize (robot.py).
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(cfg.robot.mjcf, floating=False, enable_self_collisions=False)
    for rb in cfg.rigid_bodies:
        _add_rigid_body(builder, rb)

    asset_path = resolve_path(cfg.asset)
    mesh = load_mesh(asset_path)
    _normalize_panel_units(mesh)
    if cfg.physics_panel:
        _apply_physics_panel(mesh, cfg.physics_panel)
    elif cfg.normalize_panel_islands:
        _normalize_panel_islands(mesh)
    report = analyze_mesh(asset_path, mesh)
    scale = float(cfg.scale) if cfg.scale is not None else float(report.recommended_scale or 1.0)
    _add_cloth_to_builder(builder, mesh, cfg, scale)

    builder.add_ground_plane()
    model = builder.finalize()
    apply_contact_params(model, cfg.cloth_params)
    return model, model.state()
