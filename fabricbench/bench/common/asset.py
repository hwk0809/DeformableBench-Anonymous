"""
Asset loader — geometry + metadata, simulator-agnostic.

An Asset bundles everything known about a sim asset *before* any simulator
is involved: mesh geometry, vertex count, default pose, and rendering metadata.
Two flavors:

  - DeformableAsset: cloth/bag/soft-body. Carries physics parameters, keypoints,
    mesh processing config, and per-piece physics overrides.
  - RigidAsset: free-floating or static rigid object. Carries the optional
    `mujoco_body` name used by the renderer to track per-frame xforms.

Schema lives in `<asset_dir>/asset_meta.yaml` with `kind: deformable | rigid`.
Legacy `cloth_config.yaml` is accepted during the migration window and is
treated as `kind: deformable`.

Usage:
    asset = Asset.from_dir("assets/cloth/silk_35")              # dispatches by kind
    asset = DeformableAsset.from_dir("assets/cloth/silk_35")    # asserts kind=deformable
    asset = RigidAsset.from_dir("assets/object/snickers")       # asserts kind=rigid
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from loguru import logger

project_root = Path(__file__).resolve().parent.parent

KIND_DEFORMABLE = "deformable"
KIND_RIGID = "rigid"
_LEGACY_CONFIG_NAME = "cloth_config.yaml"
_META_NAME = "asset_meta.yaml"

# Slot where cloth assets land in the runtime scene wrapper. Used to derive
# `usd_prim_path` when the asset_meta.yaml omits it.
_CLOTH_SCENE_SLOT = "/World/Cloth"


def safe_prim_name(name: str) -> str:
    """Sanitize an arbitrary string into a valid USD prim name.

    USD prim names allow [A-Za-z0-9_] and must not start with a digit.
    Dots, dashes, and other punctuation get collapsed to underscores.
    """
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "material"
    if cleaned[0].isdigit():
        return f"material_{cleaned}"
    return cleaned


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(project_root / path)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _coerce_numeric(v):
    """Recursively coerce yaml-loaded str like '40000e-3' / '91e-3' into float.

    Reason: pyyaml (yaml 1.1) requires a decimal point in the mantissa for
    scientific-notation floats, so '40000e-3' is parsed as str, not 40.0.
    """
    if isinstance(v, dict):
        return {k: _coerce_numeric(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        out = [_coerce_numeric(x) for x in v]
        return type(v)(out) if isinstance(v, tuple) else out
    if isinstance(v, str):
        s = v.strip().rstrip(",")
        try:
            return float(s)
        except ValueError:
            return v
    return v


def _load_obj_preserve_vertex_order(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ while preserving the original vertex indices used by configs.

    trimesh may reorder or expand vertices for some textured OBJs, which breaks
    semantic keypoint indices stored in asset_meta.yaml.
    """
    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                vertices.append((float(xs), float(ys), float(zs)))
            elif line.startswith("f "):
                toks = line.split()[1:]
                face = [int(tok.split("/")[0]) - 1 for tok in toks]
                if len(face) < 3:
                    continue
                if len(face) == 3:
                    faces.append(tuple(face))
                else:
                    for i in range(1, len(face) - 1):
                        faces.append((face[0], face[i], face[i + 1]))

    if not vertices or not faces:
        raise ValueError(f"Failed to parse OBJ geometry from {obj_path}")

    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
    )


def _load_meta(asset_dir: str, overrides: Optional[dict]) -> dict:
    """Find asset_meta.yaml (preferred) or cloth_config.yaml (legacy fallback).

    Legacy cloth_config.yaml is treated as kind=deformable; this fallback is a
    short-lived migration aid and will be removed once all assets are migrated.
    """
    asset_dir = _resolve(asset_dir)
    cfg = None
    used_path = None
    for fname in (_META_NAME, _LEGACY_CONFIG_NAME):
        cfg_path = os.path.join(asset_dir, fname)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            used_path = cfg_path
            if fname == _LEGACY_CONFIG_NAME and "kind" not in cfg:
                cfg["kind"] = KIND_DEFORMABLE
            break
    if cfg is None:
        raise FileNotFoundError(
            f"No {_META_NAME} or {_LEGACY_CONFIG_NAME} in {asset_dir}"
        )
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg = _coerce_numeric(cfg)
    cfg.setdefault("_source_path", used_path)
    return cfg


def _build_common_kwargs(asset_dir: str, cfg: dict, kind: str) -> dict:
    asset_dir = _resolve(asset_dir)
    obj_file = cfg.get("obj")
    if not obj_file:
        raise ValueError(f"'obj' field missing in {asset_dir}")
    obj_path = os.path.join(asset_dir, obj_file)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    usd_file = cfg.get("usd")
    usd_path = None
    if usd_file:
        usd_path = (
            usd_file if os.path.isabs(usd_file)
            else os.path.normpath(os.path.join(asset_dir, usd_file))
        )
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD render asset not found: {usd_path}")

    vertices, faces = _load_obj_preserve_vertex_order(obj_path)

    return dict(
        asset_dir=asset_dir,
        name=cfg.get("name", Path(obj_path).stem),
        kind=kind,
        obj_path=obj_path,
        vertices=vertices,
        faces=faces,
        num_vertices=len(vertices),
        num_faces=len(faces),
        default_pos=np.array(cfg.get("default_pos", [0.55, 0.0, 0.01]), dtype=np.float32),
        default_quat=np.array(cfg.get("default_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
        scale=float(cfg.get("scale", 1.0)),
        usd_path=usd_path,
        usd_prim_path=cfg.get("usd_prim_path"),
        raw_config=cfg,
    )


@dataclass
class Asset:
    """Base for any sim asset loaded from <dir>/asset_meta.yaml."""

    asset_dir: str
    name: str
    kind: str                       # KIND_DEFORMABLE | KIND_RIGID
    obj_path: str
    vertices: np.ndarray            # (N, 3) float32
    faces: np.ndarray               # (M, 3) int32
    num_vertices: int
    num_faces: int
    default_pos: np.ndarray         # [x, y, z]
    default_quat: np.ndarray        # [qw, qx, qy, qz]
    scale: float = 1.0
    usd_path: Optional[str] = None
    usd_prim_path: Optional[str] = None
    raw_config: dict = field(default_factory=dict)

    @classmethod
    def from_dir(cls, asset_dir: str, overrides: Optional[dict] = None) -> "Asset":
        """Load by dispatching on the `kind` field in the yaml.

        If called on a subclass (DeformableAsset.from_dir / RigidAsset.from_dir),
        the loaded asset must match that subclass or a TypeError is raised.
        """
        cfg = _load_meta(asset_dir, overrides)
        kind = cfg.get("kind", KIND_DEFORMABLE)
        if kind == KIND_DEFORMABLE:
            instance = DeformableAsset._build(asset_dir, cfg)
        elif kind == KIND_RIGID:
            instance = RigidAsset._build(asset_dir, cfg)
        else:
            raise ValueError(f"Unknown asset kind {kind!r} in {asset_dir}")

        if cls is not Asset and not isinstance(instance, cls):
            raise TypeError(
                f"{cls.__name__}.from_dir({asset_dir!r}) loaded a "
                f"{type(instance).__name__} (yaml kind={kind!r}); "
                f"use Asset.from_dir or the matching subclass."
            )
        return instance


@dataclass
class DeformableAsset(Asset):
    """Cloth / bag / soft-body. Carries physics + simulator-specific metadata."""

    backend: str = "direct_obj"
    physics: dict = field(default_factory=dict)
    keypoints: dict = field(default_factory=dict)
    mesh_processing: dict = field(default_factory=dict)
    piece_physics: dict = field(default_factory=dict)

    @classmethod
    def _build(cls, asset_dir: str, cfg: dict) -> "DeformableAsset":
        common = _build_common_kwargs(asset_dir, cfg, kind=KIND_DEFORMABLE)
        # Derive runtime cloth mesh path if yaml omits it. The renderer drives
        # cloth verts into this prim each frame (see omniverse_kit_render_v7.py
        # `_USD_CLOTH_MESH_PATH`); leaving it None silently falls back to the
        # silk default and the cloth never animates.
        if common["usd_prim_path"] is None:
            obj_stem = Path(common["obj_path"]).stem
            container = safe_prim_name(common["name"])
            mesh = safe_prim_name(f"{obj_stem}_obj")
            common["usd_prim_path"] = f"{_CLOTH_SCENE_SLOT}/{container}/{mesh}"

        asset = cls(
            **common,
            backend=cfg.get("backend", "direct_obj"),
            physics=cfg.get("physics", {}),
            keypoints=cfg.get("keypoints", {}),
            mesh_processing=cfg.get("mesh_processing", {}),
            piece_physics=cfg.get("piece_physics", {}),
        )
        logger.info(
            f"[DeformableAsset] loaded '{asset.name}': "
            f"{asset.num_vertices} verts, {asset.num_faces} faces, "
            f"prim={asset.usd_prim_path}"
        )
        return asset


@dataclass
class RigidAsset(Asset):
    """Free-floating or static rigid object.

    ``collision_obj`` (an axis-aligned box proxy in the body frame, metres) plus
    the ``physics`` block (mass/com/inertia_diag/friction) make the asset
    simulatable: :meth:`rigid_body_config` turns it into the sim-layer
    ``RigidBodyConfig`` the builder injects as a freejoint body.
    """

    mujoco_body: Optional[str] = None  # None = static (no per-frame xform tracking)
    collision_obj_path: Optional[str] = None
    physics: dict = field(default_factory=dict)

    @classmethod
    def _build(cls, asset_dir: str, cfg: dict) -> "RigidAsset":
        common = _build_common_kwargs(asset_dir, cfg, kind=KIND_RIGID)
        collision_obj = cfg.get("collision_obj")
        collision_path = (
            os.path.join(_resolve(asset_dir), collision_obj) if collision_obj else None
        )
        if collision_path and not os.path.exists(collision_path):
            raise FileNotFoundError(f"collision OBJ not found: {collision_path}")
        asset = cls(
            **common,
            mujoco_body=cfg.get("mujoco_body"),
            collision_obj_path=collision_path,
            physics=cfg.get("physics", {}) or {},
        )
        logger.info(
            f"[RigidAsset] loaded '{asset.name}': "
            f"{asset.num_vertices} verts, {asset.num_faces} faces, "
            f"scale={asset.scale}, mujoco_body={asset.mujoco_body!r}"
        )
        return asset

    @property
    def body_name(self) -> str:
        """Scene body key for this object (asset name unless mujoco_body is set)."""
        return self.mujoco_body or self.name

    def collision_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(center, half_extents) of the collision proxy's AABB, body frame [m]."""
        if self.collision_obj_path is None:
            raise ValueError(f"rigid asset {self.name!r} has no collision_obj")
        verts, _ = _load_obj_preserve_vertex_order(self.collision_obj_path)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        return (lo + hi) / 2.0, (hi - lo) / 2.0

    def rigid_body_config(self, pos=None, quat_wxyz=None):
        """Build the sim ``RigidBodyConfig`` (bench->sim direction, like load_robot)."""
        from fabricbench.sim.builder import RigidBodyConfig

        center, half = self.collision_box()
        phys = self.physics
        return RigidBodyConfig(
            name=self.body_name,
            pos=tuple(float(v) for v in (pos if pos is not None else self.default_pos)),
            quat_wxyz=tuple(float(v) for v in (quat_wxyz if quat_wxyz is not None
                                               else self.default_quat)),
            mass=float(phys.get("mass", 0.05)),
            com=tuple(float(v) for v in phys.get("com", (0.0, 0.0, 0.0))),
            inertia_diag=tuple(float(v) for v in phys.get("inertia_diag", (1e-5, 1e-5, 1e-5))),
            box_center=tuple(float(v) for v in center),
            box_half=tuple(float(v) for v in half),
            mu=float(phys.get("friction", 0.8)),
            solref=(tuple(float(v) for v in phys["solref"]) if "solref" in phys else None),
            solimp=(tuple(float(v) for v in phys["solimp"]) if "solimp" in phys else None),
        )


def load_robot(robot_dir: str):
    """Load ``robot_meta.yaml`` from a robot asset dir -> sim ``RobotConfig``.

    The robot asset framework, parallel to ``DeformableAsset`` for cloth. Resolves
    mjcf/urdf to absolute paths. Defined here (bench) and returns a sim type, so
    sim never depends on bench.
    """
    import os
    import yaml
    from fabricbench.sim.builder import RobotConfig

    with open(os.path.join(robot_dir, "robot_meta.yaml")) as f:
        m = yaml.safe_load(f)
    arms, g, c = m["arms"], m["gripper"], m["control"]
    home = m.get("home", {})
    pl = m.get("planner", {})
    return RobotConfig(
        mjcf=os.path.join(robot_dir, m["sim"]["mjcf"]),
        urdf=os.path.join(robot_dir, m["ik"]["urdf"]),
        base_link=m["ik"]["base_link"], ee_link=m["ik"]["ee_link"],
        left_arm=tuple(arms["left"]["joints"]), right_arm=tuple(arms["right"]["joints"]),
        left_gripper=tuple(arms["left"]["gripper"]), right_gripper=tuple(arms["right"]["gripper"]),
        eef_left=arms["left"]["eef_body"], eef_right=arms["right"]["eef_body"],
        base_left=arms["left"]["base_body"], base_right=arms["right"]["base_body"],
        left_base_pose=tuple(arms["left"]["base_pose"]), right_base_pose=tuple(arms["right"]["base_pose"]),
        gripper_limit=float(g["limit"]), eef_offset_z=float(g["eef_offset_z"]),
        arm_kp=tuple(c["arm_kp"]), arm_kd=tuple(c["arm_kd"]),
        gripper_kp=float(c["gripper_kp"]), gripper_kd=float(c["gripper_kd"]),
        home_left=tuple(home.get("left", (0.0,) * 6)), home_right=tuple(home.get("right", (0.0,) * 6)),
        velocity_limits=float(pl.get("velocity_limits", 5.0)),
        acceleration_limits=float(pl.get("acceleration_limits", 5.0)),
    )
