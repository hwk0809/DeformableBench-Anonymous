"""Task config (YAML) -> runnable env + planner + skills.

A task YAML (``config/task/<name>.yaml``) declares the asset, robot, planner,
control mode and success criteria; this module resolves it into the concrete
objects the runner drives, slimmed to what the open benchmark needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fabricbench.bench.common.asset import DeformableAsset, RigidAsset, load_robot
from fabricbench.paths import asset_dir
from fabricbench.sim.builder import SceneConfig


def _config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "task"


def _randomization_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "randomization"


def _contact_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "contact"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins), in place."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _mjcf_with_table_height(mjcf_path: str, table_dz: float) -> str:
    """Write a copy of ``mjcf_path`` with the robot base raised by ``-table_dz``.

    Table-height DR (see ``randomization.TableHeightRandomizer``): the table is
    static scene geometry, so a table that drops by ``table_dz`` is simulated by
    moving the ``robot_placement_frame`` body the other way. Mirrors
    deformable_bench's ``_apply_table_height_rand``, except that builder takes an
    MJCF *path* (``builder.add_mjcf``) rather than a parsed tree, so the patched
    XML is written out instead of mutated in memory.

    The copy lands NEXT TO the original: mesh ``file=`` attributes are relative to
    the MJCF's own directory, so any other location would break them. Names are
    keyed by dz in micrometers, so episodes sharing a dz share one file and a
    rerun overwrites rather than accumulates. ``_runtime__*`` is gitignored.
    """
    import xml.etree.ElementTree as ET

    src = Path(mjcf_path)
    out = src.with_name(f"_runtime__{src.stem}__dz{round(table_dz * 1e6):+d}um{src.suffix}")
    tree = ET.parse(src)
    node = tree.getroot().find(".//body[@name='robot_placement_frame']")
    if node is None:
        raise ValueError(
            f"table_height randomization needs a body named 'robot_placement_frame' "
            f"in {src} — none found")
    pos = [float(v) for v in node.get("pos", "0 0 0").split()]
    pos[2] -= table_dz          # table down => base up
    node.set("pos", " ".join(f"{v:.5f}" for v in pos))
    tree.write(out)
    return str(out)


@dataclass
class TaskSpec:
    name: str
    cloth: str                      # asset name under assets/, e.g. "cloth/silk_35"
    robot: str = "piper"
    planner: str = "toppra"         # "toppra" | "min_jerk"
    control_mode: str = "kinematic"  # "kinematic" | "pd"
    settle_frames: int = 90
    rigid_objects: list = field(default_factory=list)  # [{asset, pos?, quat?}, ...]
    cloth_init: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)   # per-task cloth_params overrides
    contact_materials: dict = field(default_factory=dict)  # per-task shape material overrides
    sim: dict = field(default_factory=dict)        # frequency: physics_fps/substeps/control_fps
    task_params: dict = field(default_factory=dict)
    collision_trick: dict = field(default_factory=dict)
    planner_override: dict = field(default_factory=dict)
    # Per-task PD gain overrides on top of robot_meta (PD mode only). Allowed
    # keys: arm_kp, arm_kd, gripper_kp, gripper_kd, gripper_effort.
    pd_gains: dict = field(default_factory=dict)
    # Replay the PD robot substep from a CUDA graph. Throughput only; see
    # SceneConfig.cuda_graph.
    cuda_graph: bool = True
    translation_preconditioner: bool | None = None
    randomization: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # Frequencies (single source of truth; run_task AND collect_data both read these).
    # Defaults = deformable_bench's 500 Hz physics / 25 Hz record: env.step() runs at
    # physics_fps (50) with `substeps` (10) substeps -> 500 Hz effective physics; the
    # executor records/commands every physics_fps/control_fps (= 50/25 = 2) frames =
    # clean 25 Hz. control_fps MUST divide physics_fps.
    @property
    def physics_fps(self) -> int:
        return int(self.sim.get("physics_fps", 50))

    @property
    def substeps(self) -> int:
        return int(self.sim.get("substeps", 10))

    @property
    def control_fps(self) -> int:
        return int(self.sim.get("control_fps", 25))

    @classmethod
    def from_name(cls, name: str, randomization: str | None = None) -> "TaskSpec":
        path = _config_dir() / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"task config not found: {path}")
        spec = cls.from_yaml(path)
        if randomization:
            # Orthogonal difficulty preset (RoboTwin-style): config/randomization/<name>.yaml
            # holds ONLY a randomization fragment (texture/light/...), deep-merged over the
            # task's own randomization block. Same preset serves collection AND eval.
            import yaml as _yaml
            rpath = _randomization_dir() / f"{randomization}.yaml"
            if not rpath.exists():
                raise FileNotFoundError(f"randomization preset not found: {rpath}")
            with open(rpath) as f:
                preset = _yaml.safe_load(f) or {}
            _deep_merge(spec.randomization, preset)
            spec.raw["randomization"] = spec.randomization
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        with open(path) as f:
            d = yaml.safe_load(f)
        # `base: <task name>` -- inherit config/task/<name>.yaml and override it.
        # Used by the `*_contact.yaml` switch variants: the ONLY difference from
        # the production task must be the contact keys, so they inherit instead of
        # copying 100 lines that would silently drift. NO `base:` key => nothing is
        # read (research variants stay frozen full copies on purpose).
        seen = [str(path)]
        base_name = d.get("base")
        while (parent := d.pop("base", None)):
            ppath = _config_dir() / f"{parent}.yaml"
            if not ppath.exists():
                raise FileNotFoundError(f"base task config not found: {ppath}")
            if str(ppath) in seen:
                raise ValueError(f"base: cycle {seen + [str(ppath)]}")
            seen.append(str(ppath))
            with open(ppath) as f:
                d = _deep_merge(yaml.safe_load(f) or {}, d)   # child wins
            print(f"[spec] base={parent} ({ppath})", flush=True)
        if base_name:
            d["base"] = base_name   # stamped back into spec.raw -> run metadata
        # Contact preset, same shape as the --randomization preset in from_name():
        # config/contact/<name>.yaml holds a contact-stack fragment, deep-merged
        # over the WHOLE task dict (preset wins) so one yaml key swaps a whole
        # recipe. NOTE the merge target is `d`, not the built TaskSpec: tasks read
        # their parameters from spec.raw (run_task passes cfg=spec.raw), so a
        # preset that only touched the dataclass fields would be a no-op for
        # task_params. The resolved name is stamped back into `d` -> spec.raw ->
        # run metadata.
        # NO `contact_profile:` key => nothing is read, nothing is merged, not a
        # single new statement runs: the task takes the stock path bit-for-bit.
        profile = d.get("contact_profile")
        if profile:
            cpath = _contact_dir() / f"{profile}.yaml"
            if not cpath.exists():
                raise FileNotFoundError(f"contact profile not found: {cpath}")
            with open(cpath) as f:
                _deep_merge(d, yaml.safe_load(f) or {})
            d["contact_profile"] = profile
            print(f"[spec] contact_profile={profile} ({cpath})", flush=True)
        return cls(
            name=d["name"],
            cloth=d["cloth"],
            robot=d.get("robot", "piper"),
            planner=d.get("planner", "toppra"),
            control_mode=d.get("control_mode", "kinematic"),
            settle_frames=int(d.get("settle_frames", 90)),
            rigid_objects=d.get("rigid_objects", []) or [],
            cloth_init=d.get("cloth_init", {}) or {},
            physics=d.get("physics", {}) or {},
            contact_materials=d.get("contact_materials", {}) or {},
            sim=d.get("sim", {}) or {},
            task_params=d.get("task_params", {}) or {},
            collision_trick=d.get("collision_trick", {}) or {},
            planner_override=d.get("planner_override", {}) or {},
            pd_gains=d.get("pd_gains", {}) or {},
            cuda_graph=bool(d.get("cuda_graph", True)),
            translation_preconditioner=d.get("translation_preconditioner"),
            randomization=d.get("randomization", {}) or {},
            eval=d.get("eval", {}) or {},
            raw=d,
        )

    def robot_config(self, randomizer=None):
        """Load the robot from its asset dir (robot_meta.yaml). Only piper for now.

        With a ``randomizer`` whose table-height axis is on, the episode's table dz
        is applied here — to the MJCF (sim) and to both ``*_base_pose`` (IK) in one
        place, so the two can't drift apart.
        """
        cfg = load_robot(str(asset_dir("robot/" + self.robot)))
        import dataclasses

        table_dz = float(getattr(randomizer, "table_height_offset", 0.0) or 0.0)
        if table_dz:
            def _raise(pose):
                p = list(pose)
                p[2] -= table_dz    # table down => base up (see TableHeightRandomizer)
                return tuple(p)
            cfg = dataclasses.replace(
                cfg,
                mjcf=_mjcf_with_table_height(cfg.mjcf, table_dz),
                left_base_pose=_raise(cfg.left_base_pose),
                right_base_pose=_raise(cfg.right_base_pose),
            )
        if self.pd_gains:
            allowed = {"arm_kp", "arm_kd", "gripper_kp", "gripper_kd", "gripper_effort"}
            unknown = set(self.pd_gains) - allowed
            if unknown:
                raise ValueError(f"pd_gains: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
            def _gain(v):
                return tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v)
            cfg = dataclasses.replace(
                cfg, **{k: _gain(v) for k, v in self.pd_gains.items()})
        return cfg

    def rigid_assets(self) -> list[RigidAsset]:
        """Load the RigidAssets declared in ``rigid_objects:`` (render + physics)."""
        return [RigidAsset.from_dir(str(asset_dir(ro["asset"]))) for ro in self.rigid_objects]

    def _rigid_body_configs(self) -> tuple:
        out = []
        for ro, asset in zip(self.rigid_objects, self.rigid_assets()):
            out.append(asset.rigid_body_config(
                pos=ro.get("pos"), quat_wxyz=ro.get("quat")))
        return tuple(out)

    # ------------------------------------------------------------------
    def scene_config(self, device: str | None = None, randomizer=None) -> SceneConfig:
        from fabricbench.sim.builder import cloth_params_from_physics

        # `cloth` is a repo asset name (assets/cloth/<x>) or, for generated assets
        # kept outside the repo, an absolute/existing dir path.
        import os as _os
        cloth_dir = self.cloth if _os.path.isdir(self.cloth) else str(asset_dir(self.cloth))
        asset = DeformableAsset.from_dir(cloth_dir)
        # Cloth init: drop from base_pos and settle. xy = placement, z = drop height.
        # With a randomizer, base_pos comes from randomization.cloth_init (+pos/rot
        # offset); else the task's top-level cloth_init (back-compat).
        default_base = self.cloth_init.get("base_pos", [0.0, 0.0, 0.5])
        if randomizer is not None:
            base_pos, rot_z = randomizer.sample_cloth_placement(default_base)
        else:
            base_pos, rot_z = default_base, 0.0
        # Physics from the asset's asset_meta.yaml; yaml `physics:` can still override.
        cloth_params = cloth_params_from_physics(asset.physics)
        if self.physics:
            import dataclasses
            cloth_params = dataclasses.replace(cloth_params, **self.physics)
        contact_materials = self.contact_materials
        # T10-B `contact_profile: stopface`. The three stack numbers that are
        # allowed to differ between the two tasks are the ones the ASSET fixes,
        # so read them from the asset's own meta rather than letting each task
        # yaml carry its own copy (that drift is what made flatten and pack run
        # different recipes for three weeks; docs/GRIPPER-CONTACT.md T10-B).
        #   t0   (meta physics.thickness)      -> stiffness law k = E_t*A/t0
        #   E_t  (meta physics.contact_modulus)
        # The COLLISION thickness is not one of them: the cloth keeps the task's
        # production particle_radius r, and the gripper shell half-thickness is
        # set to that same r (h = r), so the contact stack never edits the cloth
        # side (2026-09-18; r=t0/2 moved the cloth-table rest height and pushed
        # grasped corners, data/eval_out/TUP3G/TUP3G_GRASP.md).
        # Set by the preset, never by a task yaml. Absent => not a single line
        # below runs and the stack is read from the yaml exactly as before.
        if contact_materials.get("stack_from_asset"):
            import dataclasses
            try:
                t0 = float(asset.physics["thickness"])
            except KeyError as exc:
                raise ValueError(
                    f"stack_from_asset needs physics.thickness in "
                    f"{cloth_dir}/asset_meta.yaml (missing {exc})") from exc
            # T17: contact_modulus is a MATERIAL number and most cloth metas do
            # not carry one yet.  Thickness has no default (it is the geometry
            # of the stack and getting it wrong silently changes the stop), but
            # E_t does: 20 kPa is the transverse modulus the blue tshirt meta
            # carries and the value every stopface recipe has run with, so an
            # asset without the key lands on the recipe's own number instead of
            # refusing to start.
            if "contact_modulus" in asset.physics:
                e_t = float(asset.physics["contact_modulus"])
            else:
                e_t = 20000.0
                print("[spec] stack_from_asset: contact_modulus absent -> "
                      "default 20000 Pa", flush=True)
            # Legacy switch (default False since 2026-09-18): True restores the old
            # recipe that overwrote the cloth's particle_radius with t0/2.
            if bool(contact_materials.get("stack_particle_radius_from_asset", False)):
                cloth_params = dataclasses.replace(
                    cloth_params, particle_radius=0.5 * t0)
            stack_values = {"contact_material_modulus": e_t}
            # Shell half-thickness h = the cloth's particle_radius (h = r). False
            # keeps whatever tri_sdf_half_thickness the yaml carries (experiments).
            if bool(contact_materials.get("stack_half_thickness_from_asset", True)):
                stack_values["tri_sdf_half_thickness"] = float(cloth_params.particle_radius)
            contact_materials = {**contact_materials, **stack_values}
            print(f"[spec] stack_from_asset: t0={t0 * 1e3:.2f}mm "
                  f"h={float(contact_materials.get('tri_sdf_half_thickness', 5.0e-4)) * 1e3:.3f}mm "
                  f"r={cloth_params.particle_radius * 1e3:.3f}mm "
                  f"E_t={e_t:g}Pa", flush=True)
        return SceneConfig(
            asset=asset.obj_path,
            scale=float(asset.scale),
            device=device,
            robot=self.robot_config(randomizer),
            rigid_bodies=self._rigid_body_configs(),
            control_mode=self.control_mode,
            cuda_graph=bool(self.cuda_graph),
            start_height=float(base_pos[2]),
            cloth_pos_xy=(float(base_pos[0]), float(base_pos[1])),
            cloth_rot_z=float(rot_z),
            cloth_keypoints=asset.keypoints,   # single source: asset_meta.yaml
            normalize_panel_islands=bool(
                asset.mesh_processing.get("normalize_panel_islands", False)),
            physics_panel=(_os.path.join(cloth_dir, _pp)
                           if (_pp := asset.mesh_processing.get("physics_panel")) else None),
            cloth_params=cloth_params,
            translation_preconditioner=(
                bool(self.translation_preconditioner)
                if self.translation_preconditioner is not None else False
            ),
            default_shape_mu=(
                float(contact_materials["default_shape_mu"])
                if "default_shape_mu" in contact_materials else None
            ),
            gripper_shape_mu=(
                float(contact_materials["gripper_shape_mu"])
                if "gripper_shape_mu" in contact_materials else None
            ),
            gripper_two_way_coupling=bool(
                contact_materials.get("gripper_two_way_coupling", False)
            ),
            # Per-body clamp [N] on the cloth reaction fed back to rigid bodies.
            # The 20 N default silently caps grip force once gripper_kp is raised
            # (measured: P3c pins at exactly 20.00 N for 40-60 frames per episode,
            # vs 46 N on the Isaac reference), so high-kp stacks must raise it.
            rigid_two_way_max_force=float(
                contact_materials.get("rigid_two_way_max_force", 0.0)   # R10(1)/R13 gate 0: clamp retired
            ),
            gripper_contact_ke=float(
                contact_materials.get("gripper_contact_ke", 0.0)
            ),
            gripper_contact_offset=float(
                contact_materials.get("gripper_contact_offset", 0.0)
            ),
            gripper_contact_hardening=float(
                contact_materials.get("gripper_contact_hardening", 0.0)
            ),
            gripper_contact_hardening_kcap=float(
                contact_materials.get("gripper_contact_hardening_kcap", 0.0)
            ),
            gripper_joint_in_solver=bool(
                contact_materials.get("gripper_joint_in_solver", False)
            ),
            rigid_feature_contact=bool(
                contact_materials.get("rigid_feature_contact", False)
            ),
            rigid_untangling=bool(
                contact_materials.get("rigid_untangling", False)
            ),
            rigid_untangling_stiff=float(
                contact_materials.get("rigid_untangling_stiff", 0.02)
            ),
            rigid_untangling_thickness=float(
                contact_materials.get("rigid_untangling_thickness", -1.0)
            ),
            rigid_untangling_capacity=int(
                contact_materials.get("rigid_untangling_capacity", 32)
            ),
            contact_projection=bool(
                contact_materials.get("contact_projection", False)
            ),
            contact_projection_slack=float(
                contact_materials.get("contact_projection_slack", 2.0e-3)
            ),
            contact_projection_iterations=int(
                contact_materials.get("contact_projection_iterations", 3)
            ),
            contact_projection_friction=float(
                contact_materials.get("contact_projection_friction", 0.0)
            ),
            friction_anchor_kt_ratio=float(
                contact_materials.get("friction_anchor_kt_ratio", 0.0)
            ),
            friction_avbd_dual_k=float(
                contact_materials.get("friction_avbd_dual_k", 0.0)
            ),
            contact_projection_reaction_pre=bool(
                contact_materials.get("contact_projection_reaction_pre", False)
            ),
            contact_projection_impulse=bool(
                contact_materials.get("contact_projection_impulse", False)
            ),
            contact_projection_gripper_only=bool(
                contact_materials.get("contact_projection_gripper_only", False)
            ),
            contact_projection_interleaved=bool(
                contact_materials.get("contact_projection_interleaved", False)
            ),
            tri_sdf_contact=bool(
                contact_materials.get("tri_sdf_contact", False)
            ),
            sdf_anchor_kt_ratio=float(
                contact_materials.get("sdf_anchor_kt_ratio", 0.0)
            ),
            tri_sdf_half_thickness=float(
                contact_materials.get("tri_sdf_half_thickness", 5.0e-4)
            ),
            tri_sdf_voxel=float(
                contact_materials.get("tri_sdf_voxel", 1.0e-3)
            ),
            tri_sdf_pad=float(
                contact_materials.get("tri_sdf_pad", 6.0e-3)
            ),
            tri_sdf_refine_steps=int(
                contact_materials.get("tri_sdf_refine_steps", 3)
            ),
            tri_sdf_max_correction=float(
                contact_materials.get("tri_sdf_max_correction", 5.0e-3)
            ),
            tri_sdf_compliant=bool(
                contact_materials.get("tri_sdf_compliant", False)
            ),
            tri_sdf_disable_stock_penalty=bool(
                contact_materials.get("tri_sdf_disable_stock_penalty", True)
            ),
            contact_material_modulus=float(
                contact_materials.get("contact_material_modulus", 0.0)
            ),
            table_contact_ke=float(
                contact_materials.get("table_contact_ke", 0.0)
            ),
            contact_material_thickness=float(
                contact_materials.get(
                    "contact_material_thickness",
                    (asset.physics or {}).get("thickness", 1.0e-3),
                )
            ),
            view_fps=float(self.physics_fps),   # env.step() rate (sim: block)
            view_substeps=int(self.substeps),   # substeps/frame -> physics_fps*substeps Hz
        )

    def build_planner(self, randomizer=None):
        """Construct a PlannerManager(dual_arm) from the robot_meta config.

        Pass the SAME randomizer used for ``scene_config`` whenever table-height DR
        is on: the IK base_pose must carry the identical dz as the sim MJCF.
        """
        from fabricbench.bench.robot.planning.planner_manager import PlannerManager

        r = self.robot_config(randomizer)
        vel = float(self.planner_override.get("velocity_limits", r.velocity_limits))
        acc = float(self.planner_override.get("acceleration_limits", r.acceleration_limits))

        def arm(joints, gripper, base_pose) -> dict:
            return {
                "urdf_path": r.urdf, "base_link": r.base_link, "ee_link": r.ee_link,
                "base_pose": list(base_pose), "joint_names": list(joints),
                "gripper_joints": list(gripper),
                "velocity_limits": vel, "acceleration_limits": acc,
            }

        cfg = {"mode": "dual_urdf",
               "left": arm(r.left_arm, r.left_gripper, r.left_base_pose),
               "right": arm(r.right_arm, r.right_gripper, r.right_base_pose)}
        return PlannerManager(robot_type="dual_arm", planner_type=self.planner, dual_arm_config=cfg)

    @property
    def success_check_cfg(self) -> dict[str, Any]:
        return self.eval.get("success_check", {"type": "mean_z", "threshold": 0.1})
