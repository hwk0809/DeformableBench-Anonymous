"""USD scene constants + newton geometry extraction for rendering / render_state.

Shared by the Kit ShmBridge (R4 render) and the collector's save_render_state.
The robot link basenames here match BOTH our newton MJCF body labels and the
original MuJoCo body names, so the USD-link mapping is identical and the
Kit-side stays untouched (only the geometry source changes MuJoCo -> newton).
"""

from __future__ import annotations

import numpy as np

# newton body basename -> USD link prim name (== the MuJoCo->USD mapping).
# 18 links (base + link1..8 per arm); the gripper_base bodies have no USD prim.
MUJOCO_BODY_TO_USD_LINK: dict[str, str] = {
    "base_link_left": "left_base_link",
    "link1": "left_link1", "link2": "left_link2", "link3": "left_link3",
    "link4": "left_link4", "link5": "left_link5", "link6": "left_link6",
    "link7": "left_link7", "link8": "left_link8",
    "base_link_right": "right_base_link",
    "link1_arm2": "right_link1", "link2_arm2": "right_link2", "link3_arm2": "right_link3",
    "link4_arm2": "right_link4", "link5_arm2": "right_link5", "link6_arm2": "right_link6",
    "link7_arm2": "right_link7", "link8_arm2": "right_link8",
}
LINK_BODY_NAMES: list[str] = list(MUJOCO_BODY_TO_USD_LINK.keys())
LINK_USD_NAMES: list[str] = list(MUJOCO_BODY_TO_USD_LINK.values())

USD_ROBOT_ROOT = "/World/dualarm_piper"
USD_CLOTH_MESH_PATH = "/World/Cloth/silk_35_aligned/silk_35_obj"
USD_PHYSICS_SCENE = "/physicsScene"


def robot_link_poses(env) -> np.ndarray:
    """(18, 7) link world poses [x,y,z, qw,qx,qy,qz] in USD link order.

    newton body_q is [x,y,z, qx,qy,qz,qw] (xyzw); we reorder the quat to wxyz to
    match the USD/render convention the Kit/USD renderer expects."""
    from fabricbench.sim.env import resolve_body_index

    body_q = env.state_0.body_q.numpy()
    out = np.zeros((len(LINK_BODY_NAMES), 7), dtype=np.float64)
    for i, name in enumerate(LINK_BODY_NAMES):
        t = body_q[resolve_body_index(env.model, name)]
        out[i, :3] = t[:3]
        out[i, 3] = t[6]          # qw
        out[i, 4:7] = t[3:6]      # qx,qy,qz
    return out


def gripper_values(env) -> np.ndarray:
    """(2,) [left, right] gripper open fractions (repo convention 1=open)."""
    return np.array([env.left_gripper_val, env.right_gripper_val], dtype=np.float32)


def compose_runtime_scene(base_scene_usd_path: str, overlays: list[dict],
                          output_path: str | None = None) -> str:
    """Overlay asset USDs onto a base scene, returning a runtime USDA path.

    Each overlay = {"prim": path,
    "usda": asset_usd}. Without this the cloth prim in the base scene is a gray
    placeholder; overlaying the cloth asset USD binds its real material (e.g.
    BlueSilk) so the render matches the reference. Output defaults to
    ``_runtime__<scene>__<assets>.usda`` next to the base scene (gitignored).
    """
    import os
    from pathlib import Path

    from pxr import Sdf, Usd, UsdGeom

    if not overlays:
        raise ValueError("compose_runtime_scene: empty overlays")
    base_scene_usd_path = os.path.abspath(base_scene_usd_path)
    overlays = [{**o, "usda": os.path.abspath(o["usda"])} for o in overlays]
    if output_path is None:
        cache_dir = os.path.dirname(base_scene_usd_path)
        scene_stem = Path(base_scene_usd_path).stem
        asset_stems = "_".join(Path(o["usda"]).stem for o in overlays)
        output_path = os.path.join(cache_dir, f"_runtime__{scene_stem}__{asset_stems}.usda")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_layer = Sdf.Layer.FindOrOpen(base_scene_usd_path)
    if base_layer is None:
        raise RuntimeError(f"cannot open base scene: {base_scene_usd_path}")
    new_layer = Sdf.Layer.CreateNew(output_path)
    new_layer.TransferContent(base_layer)

    stage = None
    for overlay in overlays:
        prim_path, usda_path = overlay["prim"], overlay["usda"]
        prim_spec = new_layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            if stage is None:
                stage = Usd.Stage.Open(new_layer)
            UsdGeom.Xform.Define(stage, prim_path)
            prim_spec = new_layer.GetPrimAtPath(prim_path)
        prim_spec.referenceList.prependedItems = [Sdf.Reference(usda_path)]
        prim_spec.referenceList.deletedItems = []
        prim_spec.referenceList.appendedItems = []
        prim_spec.referenceList.addedItems = []
        prim_spec.referenceList.orderedItems = []
        if prim_path == "/World/Cloth":  # drop stale cloth children so bindings don't dangle
            cloth_stem = Path(usda_path).stem
            stale = [n for n in list(prim_spec.nameChildren.keys())
                     if (n.startswith("silk_") or n.startswith("blue_tshirt") or n.startswith("Tshirt"))
                     and n != cloth_stem]
            for n in stale:
                del prim_spec.nameChildren[n]
    new_layer.Save()
    return output_path
