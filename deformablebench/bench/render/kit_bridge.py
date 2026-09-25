"""ShmKitBridge — canonical Kit (Omniverse) renderer via shared memory + ZMQ.

Reuses the sim-side bridge (``_shm_fabric.ShmFabricBridge``:
SHM layout, ZMQ handshake, world->USD-local transforms, image SHM) untouched, and
swaps ONLY the geometry source from MuJoCo to newton:

- cloth vertices  <- env.get_sim_vertices()      (newton particle_q, world)
- 18 robot links  <- scene.robot_link_poses(env) (newton body_q, wxyz)
- gripper         <- env.left/right_gripper_val

Done via a tiny adapter that quacks like the MuJoCo env the parent expects, plus a
subclass that skips the mujoco body-id resolution. The Kit-side app
(tools/omniverse/omniverse_kit_render_v7.py) is geometry-agnostic and runs as-is.

Kit runs in its OWN process (its bundled python); we launch it as a subprocess and
talk over ZMQ. ``config/render/kit_render.yaml`` holds all Kit settings/paths.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import types
from pathlib import Path

import numpy as np
import yaml

from deformablebench.bench.render import scene
from deformablebench.bench.render._shm_fabric import ShmFabricBridge
from deformablebench.bench.render.bridge import RenderBridge

_REPO = Path(__file__).resolve().parents[4]   # .../deformablebench
_DEFAULT_CFG = str(Path(__file__).resolve().parent.parent / "config" / "render" / "kit_render.yaml")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(_REPO / p)


class _NewtonShmBridge(ShmFabricBridge):
    """ShmFabricBridge with mujoco body-id resolution skipped: link poses are fed
    directly via the adapter's data.xpos/xquat indexed 0..num_links-1; tracked
    rigid objects (if any) occupy the rows after the links."""

    def _resolve_body_ids(self, model):  # noqa: D401  (override)
        self._mj_body_ids = list(range(self.num_links))
        self._tracked_rigid_body_ids = [self.num_links + i for i in range(self.num_rigid_objects)]


class _NewtonEnvAdapter:
    """Quacks like the MuJoCo env that ShmFabricBridge.write_frame reads.

    ``rigid_bodies`` (env rigid body names) are appended after the robot links in
    data.xpos/xquat, matching _NewtonShmBridge's tracked-rigid row indices."""

    def __init__(self, env, rigid_bodies: list[str] | None = None):
        self.env = env
        self.model = None
        self._rigid_bodies = list(rigid_bodies or [])
        self.data = types.SimpleNamespace(time=0.0, xpos=None, xquat=None)

    def refresh(self) -> None:
        self._cloth = self.env.get_sim_vertices().astype(np.float64)
        lp = scene.robot_link_poses(self.env)        # (18,7) [x,y,z, qw,qx,qy,qz]
        if self._rigid_bodies:
            rp = np.zeros((len(self._rigid_bodies), 7))
            for i, name in enumerate(self._rigid_bodies):
                pose = self.env.get_rigid_body_pose(name)   # [x,y,z, qx,qy,qz,qw]
                rp[i, :3] = pose[:3]
                rp[i, 3:] = pose[[6, 3, 4, 5]]              # -> wxyz
            lp = np.concatenate([lp, rp], axis=0)
        self.data.time = float(self.env.get_current_sim_time())
        self.data.xpos = lp[:, :3]
        self.data.xquat = lp[:, 3:]                   # wxyz (what write_frame expects)
        self._grip = (float(self.env.left_gripper_val), float(self.env.right_gripper_val))

    def get_cloth_position(self):
        return self._cloth

    def get_gripper_value(self):
        return self._grip


class ShmKitBridge(RenderBridge):
    def __init__(self, width: int | None = None, height: int | None = None,
                 config_path: str = _DEFAULT_CFG,
                 attach: dict | None = None, cloth_usd: str | None = None,
                 cloth_mesh_path: str | None = None,
                 rigid_objects: list[dict] | None = None,
                 show_window: bool = False):
        """``attach`` (worker mode): {shm_name, zmq_endpoint, w2l, robot_root_offset,
        num_verts} from an orchestrator that already launched Kit + did INIT — the
        worker attaches to that Kit instead of launching its own (Kit starts once
        for the whole batch). ``None`` = standalone owner (launches its own Kit).

        ``rigid_objects``: [{"body": env body name, "usda": asset USD path,
        "prim": scene prim path}] — each USD is overlaid at its prim and the
        prim's xform tracks the sim body every frame (see rigid_render_objects)."""
        with open(_abs(config_path)) as f:
            self.cfg = yaml.safe_load(f)
        self.image_width = int(width or self.cfg.get("width", 1280))
        self.image_height = int(height or self.cfg.get("height", 720))
        self.camera_names = list(self.cfg["cameras"])
        self._attach = attach
        self._rigid_objects = list(rigid_objects or [])
        self._show_window = bool(show_window)
        self._cloth_usd = cloth_usd   # overlay onto /World/Cloth (binds real material)
        # per-asset prim the Kit side drives cloth verts into; default = silk scene slot
        self._cloth_mesh_path = cloth_mesh_path or scene.USD_CLOTH_MESH_PATH
        self._bridge: _NewtonShmBridge | None = None
        self._adapter: _NewtonEnvAdapter | None = None
        self._kit: subprocess.Popen | None = None
        self._frame = 0
        self._episode_id = 0

    # ------------------------------------------------------------------
    def _build_kit_command(self, zmq_endpoint: str) -> list[str]:
        c = self.cfg
        script = _abs(c["kit_render_script"])
        exec_args = [
            script, "--fabric-mode", "--zmq-endpoint", zmq_endpoint,
            "--width", str(self.image_width), "--height", str(self.image_height),
            "--renderer", c["renderer"], "--spp", str(c["spp"]),
            "--frame-updates", str(c.get("kit_frame_updates", 1)),
            "--camera-updates", str(c.get("kit_camera_updates", 6)),
            "--capture-timeout", str(c.get("kit_capture_timeout", 30.0)),
            "--capture-retries", str(c.get("kit_capture_retries", 2)),
            "--capture-retry-updates", str(c.get("kit_capture_retry_updates", 10)),
            "--quit-on-complete",
        ]
        if c.get("optix_denoiser"):
            exec_args.append("--optix-denoiser")
        exec_args += ["--cameras", *self.camera_names]

        kit_app = _abs(c["kit_app"])
        if not os.path.exists(kit_app):
            raise FileNotFoundError(f"Kit app not found: {kit_app}")
        cmd = [kit_app]
        if not self._show_window:
            cmd.append("--no-window")
        cmd += ["--/app/useFabricSceneDelegate=false",
                "--/app/settings/fabricDefaultStageFrameHistoryCount=3"]
        # Pin the Kit (Vulkan) renderer to one GPU on a shared/contended box.
        # Use Kit's own renderer selector, NOT CUDA_VISIBLE_DEVICES (which hides
        # GPUs from Kit's CUDA<->Vulkan interop and deadlocks it -> no frames).
        # This keeps every GPU visible to CUDA and only steers rendering.
        _active_gpu = os.environ.get("KIT_ACTIVE_GPU")
        if _active_gpu is None and c.get("active_gpu") is not None:
            _active_gpu = str(c["active_gpu"])
        if _active_gpu is not None and _active_gpu != "":
            cmd += ["--/renderer/multiGpu/enabled=false",
                    f"--/renderer/activeGpu={_active_gpu}"]
        if not c.get("texture_streaming", True):
            cmd += ["--/rtx/resourcemanager/enableTextureStreaming=false",
                    "--/rtx-transient/resourcemanager/enableTextureStreaming=false"]
        cmd += ["--exec", shlex.join(exec_args)]
        return cmd

    # ------------------------------------------------------------------
    @classmethod
    def launch_owner(cls, num_verts: int, width: int | None = None, height: int | None = None,
                     config_path: str = _DEFAULT_CFG,
                     cloth_usd: str | None = None, cloth_mesh_path: str | None = None,
                     rigid_objects: list[dict] | None = None,
                     show_window: bool = False):
        """Orchestrator: launch Kit ONCE + INIT; hold the returned bridge (keeps Kit +
        SHM alive) and pass the handoff dict to each worker (ShmKitBridge(attach=...))."""
        from loguru import logger
        self = cls(width=width, height=height, config_path=config_path,
                   cloth_usd=cloth_usd, cloth_mesh_path=cloth_mesh_path,
                   rigid_objects=rigid_objects, show_window=show_window)
        self._bridge = _NewtonShmBridge(
            num_cloth_verts=int(num_verts), num_links=scene_num_links(),
            num_rigid_objects=len(self._rigid_objects),
            image_shm=True, num_cameras=len(self.camera_names),
            image_width=self.image_width, image_height=self.image_height,
        )
        self._set_tracked_rigids()
        ep = self._bridge.zmq_endpoint
        cmd = self._build_kit_command(ep)
        logger.info(f"[ShmKitBridge] orchestrator launching Kit ONCE (endpoint {ep})")
        self._kit = subprocess.Popen(cmd, cwd=str(_REPO), start_new_session=True)
        self._bridge.connect(ep)
        self._init_scene()
        w2l = self._bridge._w2l_mat
        handoff = {
            "shm_name": self._bridge._shm_name, "zmq_endpoint": ep,
            "w2l": w2l.tolist() if w2l is not None else None,
            "robot_root_offset": self._bridge._robot_root_offset.tolist(),
            "num_verts": int(num_verts),
        }
        logger.success("[ShmKitBridge] Kit INIT complete (owner); workers will attach")
        return self, handoff

    def _rigid_body_names(self) -> list[str]:
        return [ro["body"] for ro in self._rigid_objects]

    def _set_tracked_rigids(self) -> None:
        """Register rigid body->prim tracking on the SHM bridge (before INIT so
        the prim list rides the INIT message; workers call it post-attach)."""
        if self._rigid_objects:
            self._bridge.set_tracked_rigid_objects(
                [ro["body"] for ro in self._rigid_objects],
                [ro["prim"] for ro in self._rigid_objects],
            )

    def _init_scene(self) -> None:
        scene_usd = _abs(self.cfg["base_scene_usd"])
        overlays = []
        if self._cloth_usd:
            # overlay the cloth asset USD onto /World/Cloth so its real material
            # (BlueSilk etc.) binds — base-scene cloth is a gray placeholder.
            overlays.append({"prim": "/World/Cloth", "usda": _abs(self._cloth_usd)})
        # rigid objects: overlay each asset USD at its tracked prim
        overlays += [{"prim": ro["prim"], "usda": _abs(ro["usda"])} for ro in self._rigid_objects]
        if overlays:
            scene_usd = scene.compose_runtime_scene(scene_usd, overlays)
        self._bridge.init_scene(
            scene_usd_path=scene_usd,
            cameras=self.camera_names,
            renderer=self.cfg["renderer"], spp=int(self.cfg["spp"]),
            width=self.image_width, height=self.image_height,
            frame_updates=int(self.cfg.get("kit_frame_updates", 1)),
            camera_updates=int(self.cfg.get("kit_camera_updates", 6)),
            optix_denoiser=bool(self.cfg.get("optix_denoiser", False)),
            cloth_normals=bool(self.cfg.get("cloth_normals", False)),
            cloth_subdivision=str(self.cfg.get("cloth_subdivision", "none")),
            cloth_subdivision_level=int(self.cfg.get("cloth_subdivision_level", 0)),
            cloth_mesh_path=self._cloth_mesh_path,
        )

    def reset(self, env) -> None:
        from loguru import logger

        if self._bridge is None:
            n_verts = int(env.model.particle_count)
            if self._attach is not None:
                # WORKER: attach to the orchestrator's already-running Kit + SHM.
                # rigid_objects must match the owner's (same task spec) so the
                # SHM layout sizes agree.
                a = self._attach
                self._bridge = _NewtonShmBridge(
                    num_cloth_verts=int(a["num_verts"]), num_links=scene_num_links(),
                    num_rigid_objects=len(self._rigid_objects),
                    attach=True, shm_name=a["shm_name"], image_shm=True,
                    num_cameras=len(self.camera_names),
                    image_width=self.image_width, image_height=self.image_height,
                )
                self._set_tracked_rigids()
                self._bridge.connect(a["zmq_endpoint"])
                self._bridge.set_transforms(a.get("w2l"), a.get("robot_root_offset", [0, 0, 0]))
                self._bridge._image_shm_cameras = list(self.camera_names)
                logger.info("[ShmKitBridge] attached to orchestrator Kit")
            else:
                # OWNER: launch our own Kit + INIT (standalone / single-episode).
                self._bridge = _NewtonShmBridge(
                    num_cloth_verts=n_verts, num_links=scene_num_links(),
                    num_rigid_objects=len(self._rigid_objects),
                    image_shm=True, num_cameras=len(self.camera_names),
                    image_width=self.image_width, image_height=self.image_height,
                )
                self._set_tracked_rigids()
                ep = self._bridge.zmq_endpoint
                cmd = self._build_kit_command(ep)
                logger.info(f"[ShmKitBridge] launching Kit (endpoint {ep})")
                self._kit = subprocess.Popen(cmd, cwd=str(_REPO), start_new_session=True)
                self._bridge.connect(ep)
                self._init_scene()
                logger.success("[ShmKitBridge] Kit INIT complete")
            self._adapter = _NewtonEnvAdapter(env, self._rigid_body_names())
        else:
            self._adapter = _NewtonEnvAdapter(env, self._rigid_body_names())
            self._bridge.reset_velocity_state()

    def start_episode(self, episode_id: int, output_dir: str | None = None,
                      lighting: dict | None = None, texture: dict | None = None) -> None:
        """Send EPISODE_START to Kit so it applies per-episode visual DR (texture
        variant + lighting) before this episode's frames. ``lighting``/``texture``
        are the ``Randomizer.get_isaac_{lighting,texture}_params()`` dicts (empty =
        no change). No-op if there's nothing to apply (keeps the clean path fast)."""
        self._episode_id = int(episode_id)
        if self._bridge is None or (not lighting and not texture):
            return
        self._bridge.start_episode(int(episode_id), output_dir,
                                   lighting=lighting or None, texture=texture or None)

    def render(self, env) -> dict[str, np.ndarray]:
        self._adapter.refresh()
        reply = self._bridge.write_frame(self._adapter, self._frame, episode_id=self._episode_id)
        self._frame += 1
        if reply.get("image_shm") and self._bridge.image_shm_available:
            buf_idx = int(reply.get("buffer_idx", self._frame % 2))
            cams = self._bridge.image_shm_cameras or self.camera_names
            return self._bridge.read_images(buf_idx, cams)
        return {}

    def close(self) -> None:
        from loguru import logger
        if self._bridge is not None:
            try:
                self._bridge.close()   # sends SHUTDOWN to Kit + unlinks SHM
            except Exception as e:
                logger.warning(f"[ShmKitBridge] close: {e}")
        if self._kit is not None:
            try:
                self._kit.terminate()
                self._kit.wait(timeout=10)
            except Exception:
                self._kit.kill()


def scene_num_links() -> int:
    return len(scene.LINK_BODY_NAMES)


def rigid_render_objects(spec) -> list[dict]:
    """TaskSpec ``rigid_objects:`` -> ShmKitBridge ``rigid_objects`` dicts.

    One {"body", "usda", "prim"} per declared rigid object that ships a USD;
    the prim name is derived from the asset name (e.g. /World/Snickers)."""
    from deformablebench.bench.common.asset import safe_prim_name

    out = []
    for asset in spec.rigid_assets():
        if not asset.usd_path:
            continue
        out.append({
            "body": asset.body_name,
            "usda": asset.usd_path,
            "prim": f"/World/{safe_prim_name(asset.name).capitalize()}",
        })
    return out
