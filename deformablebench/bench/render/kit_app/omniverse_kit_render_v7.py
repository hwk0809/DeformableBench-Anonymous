#!/usr/bin/env python3
"""
Omniverse Kit 渲染器 v7: v6 全功能 + import zmq 修复 (Kit 嵌入式 python 的 sys.path 不含 pip site-packages)。

v6 的功能保留: 基于 v5，新增 Image SHM 零拷贝图像传输。

相比 v5 的关键改进:
  - 渲染后的 RGB 图像直接写入共享内存 (Image SHM)，跳过 PNG 编解码 + 磁盘 I/O
  - 每帧节省 ~90-180ms (3 相机 × 1280×720)
  - 完全向后兼容: 若 Image SHM 不可用，自动回退到 v5 的 PNG 文件路径

v5 的功能保留:
  - Render Product + Annotator 离屏渲染
  - viewport fallback (v4 compat)

运行方式:
    kit_app --exec "tools/omniverse/omniverse_kit_render_v7.py --fabric-mode --zmq-endpoint ipc:///tmp/..."
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from urllib.request import pathname2url


# ===========================================================================
# Camera path mapping (same as v3/v4)
# ===========================================================================

CAMERA_PATH_MAP = {
    "realsense_cam": "/World/realsense_d435i_camera",
    "realsense_d435i": "/World/realsense_d435i_camera",
    "static_cam": "/World/StaticCamera",
    "vis_cam": "/World/VisCamera",
    "front_oblique_cam": "/World/FrontObliqueCamera",
    "left_hand_cam": "/World/dualarm_piper/left_link6/LeftHandCamera",
    "right_hand_cam": "/World/dualarm_piper/right_link6/RightHandCamera",
}

# USD prim paths
_USD_ROBOT_ROOT = "/World/dualarm_piper"
_USD_CLOTH_MESH_PATH = "/World/Cloth/silk_35_aligned/silk_35_obj"
_USD_PARTICLE_SYSTEM = "/World/Cloth/particleSystem"
_USD_PHYSICS_SCENE = "/physicsScene"
_USD_TABLE_PATH = "/World/table"

# Link mapping (same order as render_shm_bridge.py)
_MUJOCO_BODY_TO_USD_LINK = {
    "base_link_left": "left_base_link",
    "link1": "left_link1",
    "link2": "left_link2",
    "link3": "left_link3",
    "link4": "left_link4",
    "link5": "left_link5",
    "link6": "left_link6",
    "link7": "left_link7",
    "link8": "left_link8",
    "base_link_right": "right_base_link",
    "link1_arm2": "right_link1",
    "link2_arm2": "right_link2",
    "link3_arm2": "right_link3",
    "link4_arm2": "right_link4",
    "link5_arm2": "right_link5",
    "link6_arm2": "right_link6",
    "link7_arm2": "right_link7",
    "link8_arm2": "right_link8",
}

LINK_USD_NAMES = list(_MUJOCO_BODY_TO_USD_LINK.values())

# SHM header
HEADER_SIZE = 64
HEADER_FORMAT = "<QIIIIBB"
HEADER_PACKED_SIZE = struct.calcsize(HEADER_FORMAT)

# ZMQ message types
MSG_INIT = "INIT"
MSG_INIT_REPLY = "INIT_REPLY"
MSG_TICK = "TICK"
MSG_FRAME_DONE = "FRAME_DONE"
MSG_EPISODE_START = "EPISODE_START"
MSG_EPISODE_START_REPLY = "EPISODE_START_REPLY"
MSG_SHUTDOWN = "SHUTDOWN"
MSG_SHUTDOWN_REPLY = "SHUTDOWN_REPLY"


# ===========================================================================
# Argument parsing
# ===========================================================================

def _extract_user_argv(argv: list[str]) -> list[str]:
    filtered_args = [arg for arg in argv if not arg.startswith("--/")]
    if filtered_args:
        return filtered_args
    env_args = os.environ.get("OMNIVERSE_KIT_RENDER_ARGS", "").strip()
    if env_args:
        return shlex.split(env_args)
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omniverse Kit Renderer v6 (SHM+Fabric, Image SHM zero-copy)")

    # Fabric mode args
    parser.add_argument("--fabric-mode", action="store_true", help="Enable SHM+Fabric receiver mode")
    parser.add_argument("--zmq-endpoint", type=str, default=None, help="ZMQ IPC endpoint for fabric mode")

    # Common render args (shared with v3/v4)
    parser.add_argument("--usd-dir", required=False, help="Directory with episode_xxxx dirs (v3 batch mode)")
    parser.add_argument("--cameras", nargs="+", default=["static_cam"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--renderer", type=str, default="PathTracing", choices=["PathTracing", "RayTracedLighting"])
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--optix-denoiser", action="store_true", default=False,
                        help="Enable OptiX AI denoiser (PathTracing only)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frame-updates", type=int, default=6)
    parser.add_argument("--camera-updates", type=int, default=6,
                        help="(v5: ignored in render_product mode, only used in viewport fallback)")
    parser.add_argument("--load-updates", type=int, default=30)
    parser.add_argument("--capture-timeout", type=float, default=30.0)
    parser.add_argument("--capture-retries", type=int, default=2)
    parser.add_argument("--capture-retry-updates", type=int, default=10)
    parser.add_argument("--frame-padding", type=int, default=6)
    parser.add_argument("--output-subdir", default="kit_render")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-on-capture-timeout", action="store_true")
    parser.add_argument("--fail-on-missing-camera", action="store_true")
    parser.add_argument("--quit-on-complete", action="store_true", default=False)

    # v3 batch/stream args
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--list-cameras-only", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--stream-mode", action="store_true")
    parser.add_argument("--stream-queue-dir", type=str, default=None)
    parser.add_argument("--stream-poll-interval", type=float, default=0.01)

    return parser.parse_args(argv)


# ===========================================================================
# SHM Reader (Kit side) — identical to v4
# ===========================================================================

class ShmReader:
    """Read-only access to the SHM double-buffer."""

    def __init__(self, shm_name: str, num_verts: int, num_links: int, num_rigid_objects: int = 0):
        import numpy as np
        self._np = np
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # Orchestrator owns the segment; Kit must not unlink it on exit/crash.
        from multiprocessing.resource_tracker import unregister
        unregister(self._shm._name, "shared_memory")
        self.num_verts = num_verts
        self.num_links = num_links
        self.num_rigid_objects = num_rigid_objects

        cloth_bytes = num_verts * 3 * 4
        cloth_vel_bytes = num_verts * 3 * 4
        cloth_normal_bytes = num_verts * 3 * 4
        link_bytes = num_links * 7 * 8
        gripper_bytes = 2 * 4
        rigid_bytes = num_rigid_objects * 7 * 8
        buffer_size = (cloth_bytes + cloth_vel_bytes + cloth_normal_bytes
                       + link_bytes + gripper_bytes + rigid_bytes)

        self._cloth_views = []
        self._cloth_vel_views = []
        self._cloth_normal_views = []
        self._link_views = []
        self._rigid_views = []
        for i in range(2):
            buf_start = HEADER_SIZE + i * buffer_size
            cloth_end = buf_start + cloth_bytes
            vel_end = cloth_end + cloth_vel_bytes
            normal_end = vel_end + cloth_normal_bytes
            link_end = normal_end + link_bytes
            gripper_end = link_end + gripper_bytes
            self._cloth_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=self._shm.buf, offset=buf_start)
            )
            self._cloth_vel_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=self._shm.buf, offset=cloth_end)
            )
            self._cloth_normal_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=self._shm.buf, offset=vel_end)
            )
            self._link_views.append(
                np.ndarray((num_links, 7), dtype=np.float64, buffer=self._shm.buf, offset=normal_end)
            )
            self._rigid_views.append(
                np.ndarray((num_rigid_objects, 7), dtype=np.float64,
                           buffer=self._shm.buf, offset=gripper_end)
                if num_rigid_objects > 0
                else np.zeros((0, 7), dtype=np.float64)
            )

    def get_cloth_vertices(self, buf_idx: int):
        return self._cloth_views[buf_idx]

    def get_cloth_velocities(self, buf_idx: int):
        return self._cloth_vel_views[buf_idx]

    def get_cloth_normals(self, buf_idx: int):
        return self._cloth_normal_views[buf_idx]

    def get_link_poses(self, buf_idx: int):
        return self._link_views[buf_idx]

    def get_rigid_object_poses(self, buf_idx: int):
        return self._rigid_views[buf_idx]

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass


# ===========================================================================
# Image SHM Writer (Kit side) — writes rendered RGB images to shared memory
# ===========================================================================

# Image SHM constants (must match common/render_shm_bridge.py)
IMAGE_SHM_HEADER_SIZE = 64
IMAGE_SHM_HEADER_FORMAT = "<QIIIIIBB"  # 30 bytes packed
IMAGE_SHM_HEADER_PACKED_SIZE = struct.calcsize(IMAGE_SHM_HEADER_FORMAT)


class ImageShmWriter:
    """Kit-side writer for rendered images into shared memory.

    The SHM segment is created by the orchestrator (ShmFabricBridge) and
    attached here by name. Layout:
        [header 64B] [buffer0: cam0_rgb | cam1_rgb | ...] [buffer1: ...]
    """

    def __init__(self):
        self._shm = None
        self._image_views = None  # [buf_idx][cam_idx] → numpy view
        self.num_cameras = 0
        self.width = 0
        self.height = 0
        self.channels = 3

    def attach(self, shm_name: str, num_cameras: int, width: int, height: int):
        """Open an existing image SHM segment and create numpy views."""
        import numpy as np

        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # Don't let resource_tracker unlink — orchestrator owns this SHM
        from multiprocessing.resource_tracker import unregister
        unregister(self._shm._name, "shared_memory")

        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        image_size = width * height * self.channels
        per_buffer = num_cameras * image_size

        self._image_views = []
        for b in range(2):
            buf_start = IMAGE_SHM_HEADER_SIZE + b * per_buffer
            cam_views = []
            for c in range(num_cameras):
                offset = buf_start + c * image_size
                view = np.ndarray(
                    (height, width, self.channels),
                    dtype=np.uint8,
                    buffer=self._shm.buf,
                    offset=offset,
                )
                cam_views.append(view)
            self._image_views.append(cam_views)

        print(f"[Kit][v7] ImageShmWriter attached: '{shm_name}', "
              f"{num_cameras} cams, {width}x{height}")

    def write_image(self, buf_idx: int, cam_idx: int, rgb: "np.ndarray"):
        """Write an RGB image (H,W,3) uint8 into the SHM buffer."""
        self._image_views[buf_idx][cam_idx][:] = rgb

    def write_header(self, write_seq: int, buffer_index: int, valid_mask: int):
        packed = struct.pack(
            IMAGE_SHM_HEADER_FORMAT,
            write_seq,
            self.num_cameras,
            self.width,
            self.height,
            self.channels,
            self.width * self.height * self.channels,
            buffer_index,
            valid_mask,
        )
        self._shm.buf[:IMAGE_SHM_HEADER_PACKED_SIZE] = packed

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    @property
    def available(self) -> bool:
        return self._shm is not None and self._image_views is not None


# ===========================================================================
# Kit Runtime (from v4, unchanged)
# ===========================================================================

class KitRuntime:
    def __init__(self):
        import carb.settings
        import omni.kit.app
        import omni.kit.viewport.utility as vp_utils
        import omni.timeline
        import omni.usd

        self._viewport_utils = vp_utils
        self.settings = carb.settings.get_settings()
        self.app = omni.kit.app.get_app()
        self.usd_context = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()

    @property
    def viewport_utils(self):
        return self._viewport_utils

    def get_viewport(self):
        viewport = self.viewport_utils.get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport found")
        return viewport

    def update(self, count: int = 1):
        for _ in range(max(1, count)):
            self.app.update()

    def open_stage(self, usd_path):
        stage_uri = f"file://{pathname2url(str(usd_path))}"
        print(f"[Kit] Opening stage: {stage_uri}")
        self.usd_context.open_stage(stage_uri)

    def open_stage_path_fallback(self, usd_path):
        self.usd_context.open_stage(str(usd_path))

    def get_stage_debug_info(self) -> dict:
        """v3 compatibility stub: probes the active stage path. Return root identifier
        + real path so v3's path-match check succeeds without forcing the fallback."""
        try:
            stage = self.usd_context.get_stage()
            if stage is None:
                return {}
            root = stage.GetRootLayer()
            return {
                "root_identifier": getattr(root, "identifier", "") or "",
                "root_real_path": getattr(root, "realPath", "") or "",
            }
        except Exception:
            return {}

    def wait_for_prims(self, prim_paths, timeout_sec: float = 30.0, min_updates: int = 1) -> list:
        """v3 compatibility stub: poll until any of the requested prim paths exist."""
        if not prim_paths:
            return []
        deadline = time.time() + timeout_sec
        update_count = 0
        while time.time() < deadline:
            self.update(1)
            update_count += 1
            stage = self.usd_context.get_stage()
            if stage is None or update_count < max(1, min_updates):
                continue
            ready = [p for p in prim_paths if stage.GetPrimAtPath(p).IsValid()]
            if ready:
                return ready
        return []

    def set_current_time(self, time_sec: float):
        """v3 compatibility stub."""
        self.timeline.set_current_time(float(time_sec))

    def set_end_time(self, time_sec: float):
        """v3 compatibility stub."""
        self.timeline.set_end_time(float(time_sec))

    def wait_for_stage_ready(self, timeout_sec: float = 60.0, min_updates: int = 1):
        deadline = time.time() + timeout_sec
        update_count = 0
        while time.time() < deadline:
            self.update(1)
            update_count += 1
            stage = self.usd_context.get_stage()
            if stage is None:
                continue
            if update_count < max(1, min_updates):
                continue
            is_loading = None
            if hasattr(self.usd_context, "is_loading"):
                try:
                    is_loading = bool(self.usd_context.is_loading())
                except Exception:
                    pass
            if is_loading is False:
                return
            if is_loading is None and update_count >= max(10, min_updates):
                return
        raise TimeoutError("Timed out waiting for stage")

    def close_stage(self):
        if self.usd_context.get_stage() is not None:
            self.usd_context.close_stage()

    def set_renderer(self, renderer: str, spp: int, optix_denoiser: bool = False):
        print(f"[Kit] set_renderer called: renderer={renderer}, spp={spp}, optix_denoiser={optix_denoiser}")
        self.settings.set_bool("/app/asyncRendering", False)
        self.settings.set_bool("/app/renderer/waitIdle", True)
        self.settings.set_bool("/app/hydraEngine/waitIdle", True)
        try:
            self.settings.set("/rtx/pathtracing/spp", int(spp))
        except Exception as e:
            print(f"[Kit] WARN: set spp via .set() failed: {e}")
        try:
            self.settings.set_int("/rtx/pathtracing/spp", int(spp))
        except Exception as e:
            print(f"[Kit] WARN: set spp via .set_int() failed: {e}")
        total = int(spp) * 256
        try:
            self.settings.set_int("/rtx/pathtracing/totalSpp", total)
        except Exception as e:
            print(f"[Kit] WARN: set totalSpp failed: {e}")
        render_mode = "pt" if renderer == "PathTracing" else "rtx"
        for key in ("/persistent/rtx/modes/active", "/rtx/rendermode", "/renderer/multiGpu/currentMode"):
            try:
                self.settings.set(key, render_mode)
            except Exception as e:
                print(f"[Kit] WARN: set {key} failed: {e}")
        if renderer == "PathTracing":
            try:
                self.settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", bool(optix_denoiser))
            except Exception as e:
                print(f"[Kit] WARN: set optixDenoiser failed: {e}")
        try:
            actual_spp = self.settings.get("/rtx/pathtracing/spp")
            actual_total = self.settings.get("/rtx/pathtracing/totalSpp")
            actual_mode = self.settings.get("/rtx/rendermode")
            actual_denoiser = self.settings.get("/rtx/pathtracing/optixDenoiser/enabled")
            print(f"[Kit] VERIFY: spp={actual_spp}, totalSpp={actual_total}, rendermode={actual_mode}, optixDenoiser={actual_denoiser}")
        except Exception as e:
            print(f"[Kit] WARN: readback failed: {e}")

    def stop_timeline(self):
        self.timeline.stop()

    def clear_selection(self):
        selection = getattr(self.usd_context, "get_selection", None)
        if callable(selection):
            try:
                selection().clear_selected_prim_paths()
            except Exception:
                pass

    def configure_clean_capture(self, viewport) -> list[str]:
        applied = []
        self.clear_selection()
        setting_values = {
            "/app/viewport/grid/enabled": False,
            "/app/viewport/axes/enabled": False,
            "/app/viewport/gizmo/enabled": False,
            "/app/viewport/manipulator/enabled": False,
            "/app/viewport/selectionOutline/enabled": False,
            "/app/viewport/outline/enabled": False,
            "/app/viewport/hud/enabled": False,
            "/app/viewport/cameraPreview/enabled": False,
            "/app/viewport/show/lights": False,
            "/persistent/app/viewport/show/lights": False,
            "/app/viewport/show/cameras": False,
            "/persistent/app/viewport/show/cameras": False,
            "/persistent/app/viewport/grid/enabled": False,
            "/persistent/app/viewport/axes/enabled": False,
            "/persistent/app/viewport/gizmo/enabled": False,
            "/persistent/app/viewport/manipulator/enabled": False,
            "/persistent/app/viewport/selectionOutline/enabled": False,
            "/persistent/app/viewport/outline/enabled": False,
            "/persistent/app/viewport/hud/enabled": False,
            "/persistent/app/viewport/cameraPreview/enabled": False,
        }
        for key, value in setting_values.items():
            try:
                self.settings.set_bool(key, bool(value))
                applied.append(key)
            except Exception:
                pass
        return applied

    def post_quit(self):
        self.app.post_quit()


# ===========================================================================
# Render Product Capture (v5 新增 — 离屏渲染替代 viewport 切相机)
# ===========================================================================

class RenderProductCapture:
    """Off-screen render products with per-camera annotators.

    All cameras render in parallel via render products.
    Uses orchestrator.step_async() to drive both rendering AND
    SyntheticData data capture (LdrColorSDhost → annotator).
    Then reads annotator.get_data() for each camera.
    """

    def __init__(self):
        self._render_products = {}   # cam_name → render product handle
        self._annotators = {}        # cam_name → rgb annotator
        self._rep = None
        self._runtime = None

    def setup(self, camera_paths: dict[str, str], width: int, height: int,
              runtime: "KitRuntime" = None):
        """Create render products + RGB annotators, warm up pipeline."""
        import asyncio
        import carb.settings
        import omni.replicator.core as rep

        self._rep = rep
        self._runtime = runtime

        settings = carb.settings.get_settings()
        settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
        if settings.get("/omni/replicator/captureOnPlay"):
            settings.set_bool("/omni/replicator/captureOnPlay", False)

        # Create render products (all cameras render in parallel)
        for cam_name, cam_path in camera_paths.items():
            rp = rep.create.render_product(cam_path, (width, height), force_new=True)
            self._render_products[cam_name] = rp
            print(f"[Kit][v7] Render product created: {cam_name} → {cam_path} ({width}x{height})")

        # Let Hydra initialize render product delegates
        if runtime is not None:
            print("[Kit][v7] Waiting for Hydra to initialize render products...")
            runtime.update(10)

        # Create and attach one annotator per camera
        for cam_name, rp in self._render_products.items():
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([rp])
            self._annotators[cam_name] = annotator

        # Warm-up: orchestrator step to initialize full SyntheticData pipeline
        # Requires FSD disabled (useFabricSceneDelegate=false) for Replicator < 1.11.32
        try:
            async def _warmup():
                await rep.orchestrator.step_async(
                    rt_subframes=4, delta_time=0.0, pause_timeline=True
                )
            task = asyncio.ensure_future(_warmup())
            if runtime is not None:
                while not task.done():
                    runtime.update(1)
            if task.done() and task.exception() is not None:
                raise task.exception()
            print("[Kit][v7] Orchestrator warm-up complete")
        except Exception as e:
            print(f"[Kit][v7] WARN: Orchestrator warm-up failed: {e}")

        print(f"[Kit][v7] {len(self._render_products)} off-screen render product(s) ready")

    def render_and_capture(self, frame_idx: int, output_dir: str,
                           frame_padding: int, frame_updates: int) -> list[str]:
        """app.update() renders all render products in parallel, then read annotators."""
        import numpy as np

        # app.update() renders all render products simultaneously (no orchestrator needed
        # when FSD is disabled — SyntheticData can read render vars directly)
        self._runtime.update(frame_updates)

        # Read annotator data and save PNGs
        png_paths = []
        for cam_name, annotator in self._annotators.items():
            data = annotator.get_data()
            if data is None:
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: annotator None for {cam_name} frame {frame_idx}")
                continue

            if isinstance(data, dict):
                data = data.get("data", data)

            if not isinstance(data, np.ndarray):
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: type {type(data)} for {cam_name} frame {frame_idx}")
                continue

            if data.ndim != 3 or data.shape[2] not in (3, 4):
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: shape {data.shape} for {cam_name} frame {frame_idx}")
                continue

            rgb = data[:, :, :3] if data.shape[2] == 4 else data

            output_name = cam_name if not cam_name.startswith("/") else cam_name.strip("/").replace("/", "_")
            out_dir = Path(output_dir) / output_name / "rgb"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"rgb_{frame_idx:0{frame_padding}d}.png"

            try:
                try:
                    import cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                except ImportError:
                    from PIL import Image
                    Image.fromarray(rgb).save(str(out_path))
                png_paths.append(str(out_path))
            except Exception as e:
                print(f"[Kit][v7] Failed to save {out_path}: {e}")

        if frame_idx == 0:
            print(f"[Kit][v7] Frame {frame_idx}: captured {len(png_paths)} PNGs (fallback)")
        return png_paths

    def render_and_capture_shm(self, frame_idx: int, buf_idx: int,
                               image_writer: "ImageShmWriter",
                               frame_updates: int) -> int:
        """Render all cameras and write RGB data directly to image SHM.

        Returns a valid_mask bitmask (bit i = camera i written successfully).

        Diagnostic env vars (all opt-in, default = current behavior):
          RENDER_SYNC_MODE=probe            : per-TICK log hash at N=frame_updates,
                                              +1, +2, ... +RENDER_SYNC_DIAG_EXTRA_UPDATES.
                                              Only active for frame <= RENDER_SYNC_DIAG_MAX_FRAME.
          RENDER_SYNC_MODE=orchestrator     : replace runtime.update(N) with
                                              rep.orchestrator.step_async(..., wait) — the
                                              strict-sync path used in Kit warmup.
          RENDER_SYNC_TIMING=1              : log ms spent on render loop per TICK.
        """
        import numpy as np
        import os as _os
        import hashlib as _hashlib
        import time as _time

        mode = _os.environ.get("RENDER_SYNC_MODE", "")
        want_timing = _os.environ.get("RENDER_SYNC_TIMING", "") == "1"
        diag_max_frame = int(_os.environ.get("RENDER_SYNC_DIAG_MAX_FRAME", "3"))
        diag_extra = int(_os.environ.get("RENDER_SYNC_DIAG_EXTRA_UPDATES", "6"))
        orch_subframes = int(_os.environ.get("RENDER_SYNC_ORCH_SUBFRAMES", "4"))
        # Default-ON: use orchestrator flush at frame_idx==0 of each episode to
        # drain the pipeline and avoid returning the previous episode's last frame
        # (or Kit INIT residue for the very first episode).
        ep_flush_enabled = _os.environ.get("RENDER_EP_START_FLUSH", "1") == "1"

        t_render_start = _time.perf_counter()

        # Episode-boundary flush: only on frame 0, only when not already in a
        # special diagnostic mode. Uses the same primitive that Kit warmup does.
        do_ep_flush = (
            frame_idx == 0
            and ep_flush_enabled
            and mode not in ("probe", "orchestrator")
        )

        if do_ep_flush:
            # Two strategies for frame-0 flush:
            #   RENDER_EP_FLUSH_STRATEGY=orchestrator : rep.orchestrator.step_async
            #     (strict + tends to leave pathtracing in high-SPP accumulation
            #     mode which slows subsequent frames ~2x)
            #   RENDER_EP_FLUSH_STRATEGY=updates (default): just do N extra
            #     runtime.update(1) calls. N controlled by RENDER_EP_FLUSH_UPDATES.
            flush_strategy = _os.environ.get("RENDER_EP_FLUSH_STRATEGY", "updates")
            flush_updates_n = int(_os.environ.get("RENDER_EP_FLUSH_UPDATES", "10"))
            if flush_strategy == "orchestrator":
                import asyncio as _asyncio
                async def _sync_step():
                    await self._rep.orchestrator.step_async(
                        rt_subframes=orch_subframes, delta_time=0.0, pause_timeline=True,
                    )
                try:
                    _task = _asyncio.ensure_future(_sync_step())
                    while not _task.done():
                        self._runtime.update(1)
                    if _task.exception() is not None:
                        raise _task.exception()
                    if want_timing:
                        print(f"[Kit][v7] frame=0 ep-start orchestrator flush OK "
                              f"({(_time.perf_counter() - t_render_start) * 1000:.1f}ms)")
                except Exception as _e:
                    print(f"[Kit][v7] ep-start orchestrator failed: {_e}; fallback to runtime.update")
                    self._runtime.update(frame_updates)
            else:
                # Simple: just do N updates to let pipeline drain with the current
                # USD (no orchestrator side-effects on pathtracing accumulation).
                self._runtime.update(flush_updates_n)
                if want_timing:
                    print(f"[Kit][v7] frame=0 ep-start updates({flush_updates_n}) flush OK "
                          f"({(_time.perf_counter() - t_render_start) * 1000:.1f}ms)")
        elif mode == "probe" and frame_idx > diag_max_frame:
            # probe only for first few frames; rest fall through to default
            self._runtime.update(frame_updates)
        elif mode == "probe" and frame_idx <= diag_max_frame:
            first_cam = next(iter(self._annotators), None)
            if first_cam is not None:
                hashes = []
                t_probe_0 = _time.perf_counter()
                self._runtime.update(frame_updates)
                _d = self._annotators[first_cam].get_data()
                if isinstance(_d, dict):
                    _d = _d.get("data", _d)
                if isinstance(_d, np.ndarray) and _d.ndim == 3:
                    _rgb = _d[:, :, :3] if _d.shape[2] == 4 else _d
                    h = _hashlib.sha1(np.ascontiguousarray(_rgb).tobytes()).hexdigest()[:12]
                    hashes.append((frame_updates, h,
                                   float(_rgb.astype(np.float32).mean()),
                                   (_time.perf_counter() - t_probe_0) * 1000))
                for extra in range(1, diag_extra + 1):
                    t_probe = _time.perf_counter()
                    self._runtime.update(1)
                    _d = self._annotators[first_cam].get_data()
                    if isinstance(_d, dict):
                        _d = _d.get("data", _d)
                    if isinstance(_d, np.ndarray) and _d.ndim == 3:
                        _rgb = _d[:, :, :3] if _d.shape[2] == 4 else _d
                        h = _hashlib.sha1(np.ascontiguousarray(_rgb).tobytes()).hexdigest()[:12]
                        hashes.append((frame_updates + extra, h,
                                       float(_rgb.astype(np.float32).mean()),
                                       (_time.perf_counter() - t_probe) * 1000))
                print(f"[Kit][v7][DIAG probe] frame={frame_idx} cam={first_cam}:")
                for n, h, m, dt in hashes:
                    print(f"    N={n:2d}  hash={h}  mean={m:.2f}  extra_dt={dt:6.1f}ms")
        elif mode == "orchestrator":
            # Strict sync: orchestrator step_async with pause_timeline; block main thread
            # by polling runtime.update(1) until the awaitable completes.
            import asyncio as _asyncio
            async def _sync_step():
                await self._rep.orchestrator.step_async(
                    rt_subframes=orch_subframes, delta_time=0.0, pause_timeline=True,
                )
            try:
                _task = _asyncio.ensure_future(_sync_step())
                while not _task.done():
                    self._runtime.update(1)
                if _task.exception() is not None:
                    raise _task.exception()
            except Exception as _e:
                print(f"[Kit][v7] orchestrator step failed: {_e}; falling back to runtime.update")
                self._runtime.update(frame_updates)
        else:
            self._runtime.update(frame_updates)

        if want_timing:
            _dt_ms = (_time.perf_counter() - t_render_start) * 1000
            print(f"[Kit][v7][TIMING] mode={mode or 'default'} frame={frame_idx} render_loop_ms={_dt_ms:.1f}")

        valid_mask = 0
        cam_idx = 0
        for cam_name, annotator in self._annotators.items():
            data = annotator.get_data()
            if data is None:
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: annotator None for {cam_name} frame {frame_idx}")
                cam_idx += 1
                continue

            if isinstance(data, dict):
                data = data.get("data", data)

            if not isinstance(data, np.ndarray):
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: type {type(data)} for {cam_name} frame {frame_idx}")
                cam_idx += 1
                continue

            if data.ndim != 3 or data.shape[2] not in (3, 4):
                if frame_idx == 0:
                    print(f"[Kit][v7] WARN: shape {data.shape} for {cam_name} frame {frame_idx}")
                cam_idx += 1
                continue

            rgb = data[:, :, :3] if data.shape[2] == 4 else data
            image_writer.write_image(buf_idx, cam_idx, rgb)
            valid_mask |= (1 << cam_idx)
            cam_idx += 1

        if frame_idx == 0:
            n_valid = bin(valid_mask).count("1")
            print(f"[Kit][v7] Frame {frame_idx}: wrote {n_valid} images to SHM")

        return valid_mask

    def cleanup(self):
        """Destroy all render products and detach annotators."""
        for cam_name, annotator in self._annotators.items():
            try:
                annotator.detach()
            except Exception:
                pass
        for cam_name, rp in self._render_products.items():
            try:
                rp.destroy()
            except Exception:
                pass
        self._render_products.clear()
        self._annotators.clear()




# ===========================================================================
# Viewport capture fallback (from v4, used when replicator is unavailable)
# ===========================================================================

class ViewportCaptureBackend:
    def __init__(self, runtime: KitRuntime, viewport):
        import inspect
        self._inspect = inspect
        self.runtime = runtime
        self.viewport = viewport
        self.capture_mode = None
        self.capture_callable = None
        self._resolve_backend()

    def _resolve_backend(self):
        capture_fn = getattr(self.runtime.viewport_utils, "capture_viewport_to_file", None)
        if callable(capture_fn):
            self.capture_mode = "utility.capture_viewport_to_file"
            self.capture_callable = capture_fn
            return
        schedule_fn = getattr(self.viewport, "schedule_capture", None)
        if callable(schedule_fn):
            self.capture_mode = "viewport.schedule_capture"
            self.capture_callable = schedule_fn
            return
        raise RuntimeError("No viewport capture backend found")

    def capture_to_file(self, output_path: Path, timeout: float, retries: int = 0, retry_updates: int = 10):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries + 1):
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink(missing_ok=True)
            if self.capture_mode == "utility.capture_viewport_to_file":
                try:
                    capture_result = self.capture_callable(self.viewport, str(output_path))
                except TypeError:
                    capture_result = self.capture_callable(str(output_path))
            else:
                capture_result = self.capture_callable(str(output_path))
            try:
                self._wait_for_file(output_path, capture_result, timeout)
                return
            except TimeoutError:
                if attempt >= retries:
                    raise
                output_path.unlink(missing_ok=True)
                self.runtime.update(max(1, retry_updates))

    def _wait_for_file(self, output_path: Path, capture_result, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.runtime.update(1)
            if output_path.exists() and output_path.stat().st_size > 0:
                return
            if capture_result is not None:
                try:
                    if hasattr(capture_result, "done") and capture_result.done():
                        if hasattr(capture_result, "result"):
                            capture_result.result()
                    elif (hasattr(capture_result, "wait_for_result")
                          and not self._inspect.iscoroutinefunction(capture_result.wait_for_result)):
                        capture_result.wait_for_result()
                except Exception:
                    pass
        raise TimeoutError(f"Capture timeout: {output_path}")


# ===========================================================================
# Viewport helpers (for fallback mode)
# ===========================================================================

def _set_viewport_camera(viewport, camera_path: str):
    if hasattr(viewport, "set_active_camera"):
        viewport.set_active_camera(camera_path)
    elif hasattr(viewport, "camera_path"):
        viewport.camera_path = camera_path
    else:
        raise RuntimeError("Cannot set viewport camera")


def _set_viewport_resolution(viewport, width: int, height: int):
    resolution = (int(width), int(height))
    if hasattr(viewport, "set_texture_resolution"):
        viewport.set_texture_resolution(resolution)
    elif hasattr(viewport, "set_resolution"):
        viewport.set_resolution(resolution)
    elif hasattr(viewport, "resolution"):
        viewport.resolution = resolution


def _ensure_front_oblique_camera(stage) -> str | None:
    """Create a paper/debug camera above +X, looking down toward -X."""
    try:
        from pxr import Gf, UsdGeom

        camera_path = CAMERA_PATH_MAP["front_oblique_cam"]
        prim = stage.GetPrimAtPath(camera_path)
        if prim.IsValid() and prim.GetTypeName() == "Camera":
            return camera_path

        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(11.4)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(11.785)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera.CreateProjectionAttr("perspective")

        xformable = UsdGeom.Xformable(camera)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        translate_op.Set(Gf.Vec3d(1.0, 0.0, 1.1))
        # Equivalent to Kit UI Orient XYZ = (0, 34, 90) degrees.
        orient_op.Set(Gf.Quatf(0.6762096, Gf.Vec3f(0.206738, 0.206738, 0.6762096)))
        print(f"[Kit][v7] Created front_oblique_cam at {camera_path}: translate=(1, 0, 1.1), orient=(0, 34, 90)")
        return camera_path
    except Exception as exc:
        print(f"[Kit][v7] WARN: failed to create front_oblique_cam: {exc}")
        return None


def _resolve_camera_path(stage, camera_name: str) -> str | None:
    if camera_name.startswith("/"):
        prim = stage.GetPrimAtPath(camera_name)
        if prim.IsValid() and prim.GetTypeName() == "Camera":
            return camera_name
        return None
    mapped_path = CAMERA_PATH_MAP.get(camera_name)
    if mapped_path is not None:
        prim = stage.GetPrimAtPath(mapped_path)
        if prim.IsValid() and prim.GetTypeName() == "Camera":
            return mapped_path
        if camera_name == "front_oblique_cam":
            return _ensure_front_oblique_camera(stage)
    # Fallback: traverse stage
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera" and prim.GetName() == camera_name:
            return str(prim.GetPath())
    return None


# ===========================================================================
# FabricReceiver: Kit-side SHM reader + Fabric/USD updater + render loop
# ===========================================================================

class FabricReceiver:
    """Receives frame data from SHM, updates Fabric/USD prims, renders via
    off-screen render products (v5) or viewport fallback (v4 compat)."""

    def __init__(self, runtime: KitRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self._shm_reader: ShmReader | None = None
        self._use_fabric = False
        self._fabric_stage = None
        # USD prim handles (populated in _handle_init)
        self._cloth_points_attr = None
        self._cloth_extent_attr = None
        self._cloth_velocities_attr = None
        self._cloth_normals_enabled = False
        self._link_ops: list[tuple] = []  # [(translate_op, orient_op), ...]
        self._first_tick = True

        # Camera info
        self._camera_paths: dict[str, str] = {}  # name → path

        # v5: off-screen render product capture (primary)
        self._rp_capture: RenderProductCapture | None = None
        # v4 fallback: viewport capture
        self._viewport = None
        self._capture_backend: ViewportCaptureBackend | None = None

        self._capture_mode: str = "uninitialized"  # "render_product" or "viewport"

        # v6: image SHM writer (zero-copy image transfer)
        self._image_writer: ImageShmWriter | None = None
        self._image_shm_cameras: list[str] = []  # camera order in SHM slots

        # Output
        self._output_dir: str = ""
        self._episode_id: int = 0

    def run(self, zmq_endpoint: str):
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(zmq_endpoint)
        print(f"[Kit][v7] ZMQ REP bound to {zmq_endpoint}")

        try:
            while True:
                msg = sock.recv_json()
                msg_type = msg.get("type", "")

                if msg_type == MSG_INIT:
                    reply = self._handle_init(msg)
                    sock.send_json(reply)

                elif msg_type == MSG_TICK:
                    reply = self._handle_tick(msg)
                    sock.send_json(reply)

                elif msg_type == MSG_EPISODE_START:
                    reply = self._handle_episode_start(msg)
                    sock.send_json(reply)

                elif msg_type == MSG_SHUTDOWN:
                    sock.send_json({"type": MSG_SHUTDOWN_REPLY, "status": "ok"})
                    break

                else:
                    sock.send_json({"status": "error", "error": f"Unknown message type: {msg_type}"})

                # Keep Kit alive
                self.runtime.update(1)

        finally:
            if self._image_writer is not None:
                self._image_writer.close()
            if self._rp_capture is not None:
                self._rp_capture.cleanup()
            if self._shm_reader is not None:
                self._shm_reader.close()
            sock.close()
            ctx.term()
            print("[Kit][v7] Shutdown complete")

    def _handle_init(self, msg: dict) -> dict:
        """Open scene once, extract transforms, setup render products."""
        import numpy as np
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        self._cloth_normals_enabled = msg.get("cloth_normals", False)
        self._cloth_subdivision = msg.get("cloth_subdivision", "none")
        self._cloth_subdivision_level = msg.get("cloth_subdivision_level", 0)

        t_start = time.perf_counter()

        # 1. Attach to SHM
        shm_name = msg["shm_name"]
        num_verts = msg["num_verts"]
        num_links = msg["num_links"]
        num_rigid_objects = int(msg.get("num_rigid_objects", 0))
        rigid_usd_prims = list(msg.get("rigid_usd_prims", []))
        self._shm_reader = ShmReader(shm_name, num_verts, num_links, num_rigid_objects)
        print(
            f"[Kit][v7] Attached to SHM '{shm_name}': {num_verts} verts, "
            f"{num_links} links, {num_rigid_objects} rigid"
        )
        # Cache rigid prim list for xform-op resolution (after stage is open).
        self._rigid_usd_prims = rigid_usd_prims
        self._rigid_ops: list[tuple] = []

        # 2. Open scene template (only once!)
        scene_usd_path = msg["scene_usd_path"]
        self.runtime.open_stage(Path(scene_usd_path))
        self.runtime.wait_for_stage_ready(timeout_sec=60.0, min_updates=msg.get("load_updates", 30))

        stage = self.runtime.usd_context.get_stage()
        if stage is None:
            self.runtime.open_stage_path_fallback(Path(scene_usd_path))
            self.runtime.wait_for_stage_ready(timeout_sec=60.0, min_updates=msg.get("load_updates", 30))
            stage = self.runtime.usd_context.get_stage()

        if stage is None:
            return {"type": MSG_INIT_REPLY, "status": "error", "error": "Failed to open stage"}

        # 3. Disable physics
        self._disable_physics(stage, UsdPhysics)

        # 4. Extract cloth world-to-local transform
        cloth_mesh_path = msg.get("cloth_mesh_path", _USD_CLOTH_MESH_PATH)
        self._cloth_mesh_path = cloth_mesh_path  # store for later use
        cloth_prim = stage.GetPrimAtPath(cloth_mesh_path)
        w2l_matrix = None
        if cloth_prim.IsValid():
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            local_to_world = xform_cache.GetLocalToWorldTransform(cloth_prim)
            world_to_local = local_to_world.GetInverse()
            w2l_matrix = []
            for row in range(4):
                for col in range(4):
                    w2l_matrix.append(float(world_to_local[row][col]))
            print(f"[Kit][v7] Cloth w2l matrix extracted")
        else:
            print(f"[Kit][v7] WARN: Cloth mesh not found at {cloth_mesh_path}")

        # 5. Extract robot root offset
        robot_root_offset = [0.0, 0.0, 0.0]
        robot_root_prim = stage.GetPrimAtPath(_USD_ROBOT_ROOT)
        if robot_root_prim.IsValid():
            xformable = UsdGeom.Xformable(robot_root_prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    value = op.Get()
                    robot_root_offset = [float(value[0]), float(value[1]), float(value[2])]
                    break
        print(f"[Kit][v7] Robot root offset: {robot_root_offset}")

        # 6. Try Fabric (usdrt) for direct GPU access
        try:
            import usdrt
            fabric_stage = usdrt.Usd.Stage.Attach(self.runtime.usd_context.get_stage_id())
            if fabric_stage is not None:
                self._fabric_stage = fabric_stage
                self._use_fabric = True
                print("[Kit][v7] usdrt Fabric stage attached (fast path)")
            else:
                print("[Kit][v7] usdrt stage attach returned None, falling back to USD API")
        except (ImportError, AttributeError, Exception) as e:
            print(f"[Kit][v7] usdrt not available ({e}), falling back to USD API")

        # 7. Cache cloth mesh points + velocities attribute
        if cloth_prim.IsValid():
            cloth_mesh = UsdGeom.Mesh(cloth_prim)
            self._cloth_points_attr = cloth_mesh.GetPointsAttr()
            self._cloth_extent_attr = cloth_mesh.GetExtentAttr()
            point_based = UsdGeom.PointBased(cloth_prim)
            self._cloth_velocities_attr = point_based.CreateVelocitiesAttr()
            if self._cloth_normals_enabled:
                cloth_mesh.CreateDoubleSidedAttr(True)

                # Subdivision Surface — smooth shading without explicit normals.
                # Explicit per-vertex normals cause artifacts at cloth folds
                # (shared vertex averages opposing face normals → dark spots).
                # Subdivision smooths the geometry; the renderer derives correct
                # normals from the smooth surface automatically.
                subdiv = self._cloth_subdivision
                if subdiv and subdiv != "none":
                    cloth_mesh.GetSubdivisionSchemeAttr().Set(subdiv)
                    cloth_mesh.GetInterpolateBoundaryAttr().Set(UsdGeom.Tokens.edgeAndCorner)
                    if self._cloth_subdivision_level > 0:
                        cloth_prim.CreateAttribute(
                            "refinementLevel", Sdf.ValueTypeNames.Int
                        ).Set(self._cloth_subdivision_level)
                    print(f"[Kit][v7] Cloth: doubleSided=ON, subdivision={subdiv}, "
                          f"level={self._cloth_subdivision_level}")
                else:
                    print("[Kit][v7] Cloth: doubleSided=ON, subdivision=none (flat shading)")
            else:
                print("[Kit][v7] Cloth: cloth_normals=OFF")
            try:
                existing_pts = self._cloth_points_attr.Get()
                if existing_pts is not None:
                    print(f"[Kit][v7] Cloth mesh: {len(existing_pts)} vertices, "
                          f"extent={self._cloth_extent_attr.Get()}")
                else:
                    print("[Kit][v7] WARN: Cloth points attr returned None")
            except Exception as e:
                print(f"[Kit][v7] Could not read initial points: {e}")

        # 8. Cache robot link xformOps
        self._link_ops = []
        for usd_link_name in LINK_USD_NAMES:
            usd_path = f"{_USD_ROBOT_ROOT}/{usd_link_name}"
            prim = stage.GetPrimAtPath(usd_path)
            if not prim.IsValid():
                print(f"[Kit][v7] WARN: Link prim not found: {usd_path}")
                self._link_ops.append((None, None))
                continue

            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            orient_op = None
            for op in ops:
                if op.GetOpName() == "xformOp:translate":
                    translate_op = op
                elif op.GetOpName() == "xformOp:orient":
                    orient_op = op

            if translate_op is None:
                translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            if orient_op is None:
                orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

            self._link_ops.append((translate_op, orient_op))
        print(f"[Kit][v7] Cached {len([x for x in self._link_ops if x[0] is not None])} link xformOps")

        # 8b. Cache xform ops for tracked rigid prims (e.g. /World/Snickers).
        # Created lazily — if the prim was added by compose_runtime_scene the
        # Xformable has no ops yet, so we add translate + orient.
        self._rigid_ops = []
        for prim_path in self._rigid_usd_prims:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"[Kit][v7] WARN: rigid prim not found: {prim_path}")
                self._rigid_ops.append((None, None))
                continue
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            translate_op = next(
                (op for op in ops if op.GetOpName() == "xformOp:translate"), None
            )
            orient_op = next(
                (op for op in ops if op.GetOpName() == "xformOp:orient"), None
            )
            if translate_op is None:
                translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            if orient_op is None:
                orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            self._rigid_ops.append((translate_op, orient_op))
        if self._rigid_usd_prims:
            print(
                f"[Kit][v7] Cached {len([x for x in self._rigid_ops if x[0] is not None])} "
                f"rigid xformOps"
            )

        # 9. Setup renderer
        renderer = msg.get("renderer", "PathTracing")
        spp = msg.get("spp", 64)
        optix_denoiser = msg.get("optix_denoiser", False)
        print(f"[Kit][v7] INIT msg render params: renderer={renderer}, spp={spp}, optix_denoiser={optix_denoiser}")
        self.runtime.set_renderer(renderer, spp, optix_denoiser=optix_denoiser)
        self.runtime.stop_timeline()

        # 10. Resolve cameras
        cameras = msg.get("cameras", ["static_cam"])
        self._camera_paths = {}
        for cam_name in cameras:
            cam_path = _resolve_camera_path(stage, cam_name)
            if cam_path:
                self._camera_paths[cam_name] = cam_path
                print(f"[Kit][v7] Camera: {cam_name} → {cam_path}")
            else:
                print(f"[Kit][v7] WARN: Camera '{cam_name}' not found")

        # 11. Setup capture — try render products first, fallback to viewport
        width = msg.get("width", 1280)
        height = msg.get("height", 720)

        # Always configure clean capture for global UI settings
        try:
            viewport = self.runtime.get_viewport()
            self.runtime.configure_clean_capture(viewport)
        except Exception:
            pass

        self._setup_capture(width, height, msg)

        # Store params for tick
        self._frame_updates = msg.get("frame_updates", 6)
        self._camera_updates = msg.get("camera_updates", self.args.camera_updates)
        self._capture_timeout = msg.get("capture_timeout", 30.0)
        self._capture_retries = msg.get("capture_retries", 2)
        self._capture_retry_updates = msg.get("capture_retry_updates", 10)
        self._frame_padding = 6

        # 12. Attach to image SHM (v6: zero-copy image transfer)
        image_shm_name = msg.get("image_shm_name")
        image_shm_ready = False
        if image_shm_name and self._capture_mode == "render_product":
            try:
                self._image_writer = ImageShmWriter()
                self._image_writer.attach(
                    image_shm_name,
                    num_cameras=len(self._camera_paths),
                    width=width,
                    height=height,
                )
                self._image_shm_cameras = list(self._camera_paths.keys())
                image_shm_ready = True
                print(f"[Kit][v7] Image SHM ready, cameras={self._image_shm_cameras}")
            except Exception as e:
                print(f"[Kit][v7] WARN: Image SHM attach failed ({e}), falling back to PNG")
                self._image_writer = None
        elif image_shm_name:
            print(f"[Kit][v7] Image SHM skipped: capture_mode={self._capture_mode} (requires render_product)")

        # Initial render to warm up GPU
        self.runtime.update(self._frame_updates)

        t_end = time.perf_counter()
        print(f"[Kit][v7] INIT complete in {t_end - t_start:.2f}s (capture_mode={self._capture_mode}, "
              f"image_shm={image_shm_ready})")

        return {
            "type": MSG_INIT_REPLY,
            "status": "ok",
            "w2l_matrix": w2l_matrix,
            "robot_root_offset": robot_root_offset,
            "use_fabric": self._use_fabric,
            "capture_mode": self._capture_mode,
            "cameras_resolved": {k: v for k, v in self._camera_paths.items()},
            "image_shm_ready": image_shm_ready,
            "image_shm_cameras": self._image_shm_cameras if image_shm_ready else [],
            "init_time_sec": t_end - t_start,
        }

    def _setup_capture(self, width: int, height: int, msg: dict):
        """Try render products first, verify they work, fall back to viewport capture."""
        try:
            self._rp_capture = RenderProductCapture()
            self._rp_capture.setup(self._camera_paths, width, height, runtime=self.runtime)
            self._capture_mode = "render_product"
            print("[Kit][v7] Using off-screen render products (viewport stays free)")
        except (ImportError, Exception) as e:
            print(f"[Kit][v7][WARN] Render products unavailable ({e}), falling back to viewport capture (v4 compat)")
            self._rp_capture = None
            self._capture_mode = "viewport"
            self._viewport = self.runtime.get_viewport()
            _set_viewport_resolution(self._viewport, width, height)
            self.runtime.configure_clean_capture(self._viewport)
            self._capture_backend = ViewportCaptureBackend(self.runtime, self._viewport)
            print(f"[Kit][v7][WARN] Viewport fallback ready: {width}x{height}, "
                  f"capture={self._capture_backend.capture_mode}")

    def _handle_tick(self, msg: dict) -> dict:
        """Read SHM, update prims, render, capture."""
        from pxr import Gf, Vt
        import numpy as np

        t_start = time.perf_counter()

        buf_idx = msg["buffer_idx"]
        frame_idx = msg["frame_idx"]
        seq = msg["seq"]

        # 1. Read cloth data from SHM (zero-copy numpy view)
        cloth_verts = self._shm_reader.get_cloth_vertices(buf_idx)  # (N, 3) float32
        cloth_vels = self._shm_reader.get_cloth_velocities(buf_idx)  # (N, 3) float32
        # Normals are in SHM but NOT applied to the mesh — saved to NPZ only
        link_poses = self._shm_reader.get_link_poses(buf_idx)  # (18, 7) float64

        t_read = time.perf_counter()

        # 2. Update cloth mesh (always USD API for reliable Hydra notification)
        if self._cloth_points_attr is not None:
            self._update_cloth_usd(cloth_verts, cloth_vels, Vt)
            self._update_cloth_extent(cloth_verts, Vt)

            if self._first_tick:
                self._first_tick = False
                print(f"[Kit][v7] FIRST TICK: cloth shape={cloth_verts.shape}, "
                      f"min={cloth_verts.min(axis=0)}, max={cloth_verts.max(axis=0)}, "
                      f"first_vert={cloth_verts[0]}")

        t_cloth = time.perf_counter()

        # 3. Update robot link transforms
        for i, (translate_op, orient_op) in enumerate(self._link_ops):
            if translate_op is None:
                continue
            pose = link_poses[i]
            translate_op.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
            orient_op.Set(Gf.Quatd(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])))

        # 3b. Update tracked rigid prim transforms (snickers etc.)
        if self._rigid_ops:
            rigid_poses = self._shm_reader.get_rigid_object_poses(buf_idx)
            for i, (translate_op, orient_op) in enumerate(self._rigid_ops):
                if translate_op is None:
                    continue
                pose = rigid_poses[i]
                translate_op.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
                orient_op.Set(Gf.Quatd(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])))

        t_links = time.perf_counter()

        # 4. Render + Capture (branched by capture mode)
        # v6: image SHM path (zero-copy, no PNG I/O)
        if self._image_writer is not None and self._image_writer.available and self._rp_capture is not None:
            valid_mask = self._handle_tick_image_shm(frame_idx, buf_idx, seq)
            t_capture = time.perf_counter()
            return {
                "type": MSG_FRAME_DONE,
                "seq": seq,
                "status": "ok",
                "image_shm": True,
                "buffer_idx": buf_idx,
                "valid_mask": valid_mask,
                "timing": {
                    "shm_read_ms": (t_read - t_start) * 1000,
                    "cloth_update_ms": (t_cloth - t_read) * 1000,
                    "link_update_ms": (t_links - t_cloth) * 1000,
                    "render_capture_ms": (t_capture - t_links) * 1000,
                    "total_ms": (t_capture - t_start) * 1000,
                },
            }

        # v5 fallback: PNG file path
        if self._capture_mode == "render_product":
            png_paths = self._handle_tick_render_product(frame_idx)
        else:
            # v4 fallback: viewport camera switching
            png_paths = self._handle_tick_viewport_fallback(frame_idx)

        t_capture = time.perf_counter()

        return {
            "type": MSG_FRAME_DONE,
            "seq": seq,
            "status": "ok",
            "png_paths": png_paths,
            "timing": {
                "shm_read_ms": (t_read - t_start) * 1000,
                "cloth_update_ms": (t_cloth - t_read) * 1000,
                "link_update_ms": (t_links - t_cloth) * 1000,
                "render_capture_ms": (t_capture - t_links) * 1000,
                "total_ms": (t_capture - t_start) * 1000,
            },
        }

    def _handle_tick_image_shm(self, frame_idx: int, buf_idx: int, seq: int) -> int:
        """v6 path: render all cameras and write RGB directly to image SHM."""
        try:
            valid_mask = self._rp_capture.render_and_capture_shm(
                frame_idx=frame_idx,
                buf_idx=buf_idx,
                image_writer=self._image_writer,
                frame_updates=self._frame_updates,
            )
            self._image_writer.write_header(seq, buf_idx, valid_mask)
        except Exception as e:
            print(f"[Kit][v7] Image SHM capture failed for frame {frame_idx}: {e}")
            valid_mask = 0
        return valid_mask

    def _handle_tick_render_product(self, frame_idx: int) -> list[str]:
        """v5 fallback: app.update() renders all render products simultaneously,
        then read annotator data and save PNGs. No viewport switching."""
        try:
            png_paths = self._rp_capture.render_and_capture(
                frame_idx=frame_idx,
                output_dir=self._output_dir,
                frame_padding=self._frame_padding,
                frame_updates=self._frame_updates,
            )
        except Exception as e:
            print(f"[Kit][v7] Render product capture failed for frame {frame_idx}: {e}")
            png_paths = []
        return png_paths

    def _handle_tick_viewport_fallback(self, frame_idx: int) -> list[str]:
        """v4 fallback: switch viewport camera for each capture."""
        self.runtime.clear_selection()
        self.runtime.update(self._frame_updates)

        png_paths = []
        for cam_name, cam_path in self._camera_paths.items():
            _set_viewport_camera(self._viewport, cam_path)
            self.runtime.configure_clean_capture(self._viewport)
            self.runtime.update(self._camera_updates)

            output_name = cam_name if not cam_name.startswith("/") else cam_name.strip("/").replace("/", "_")
            output_dir = Path(self._output_dir) / output_name / "rgb"
            output_path = output_dir / f"rgb_{frame_idx:0{self._frame_padding}d}.png"

            try:
                self._capture_backend.capture_to_file(
                    output_path,
                    timeout=self._capture_timeout,
                    retries=self._capture_retries,
                    retry_updates=self._capture_retry_updates,
                )
                png_paths.append(str(output_path))
            except (TimeoutError, RuntimeError) as e:
                print(f"[Kit][v7] Viewport capture failed for {cam_name} frame {frame_idx}: {e}")
                output_path.unlink(missing_ok=True)
        return png_paths

    def _update_cloth_usd(self, cloth_verts, cloth_vels, Vt):
        """Update cloth mesh points + velocities via USD API.

        Smooth shading is handled by Subdivision Surface (set in _handle_init),
        not by explicit per-vertex normals.  Normals data is still collected
        through SHM and saved to NPZ for offline use.
        """
        from pxr import Usd

        if hasattr(Vt.Vec3fArray, 'FromNumpy'):
            points = Vt.Vec3fArray.FromNumpy(cloth_verts)
        else:
            points = Vt.Vec3fArray([(float(r[0]), float(r[1]), float(r[2])) for r in cloth_verts])
        self._cloth_points_attr.Set(points, Usd.TimeCode.Default())

        if self._cloth_velocities_attr is not None and cloth_vels is not None:
            if hasattr(Vt.Vec3fArray, 'FromNumpy'):
                velocities = Vt.Vec3fArray.FromNumpy(cloth_vels)
            else:
                velocities = Vt.Vec3fArray([(float(r[0]), float(r[1]), float(r[2])) for r in cloth_vels])
            self._cloth_velocities_attr.Set(velocities, Usd.TimeCode.Default())

    def _update_cloth_extent(self, cloth_verts, Vt):
        """Update mesh extent (AABB) to prevent Hydra frustum culling."""
        from pxr import Gf, Usd
        if self._cloth_extent_attr is None:
            return
        min_pt = cloth_verts.min(axis=0)
        max_pt = cloth_verts.max(axis=0)
        pad = 0.02
        extent = Vt.Vec3fArray([
            Gf.Vec3f(float(min_pt[0]) - pad, float(min_pt[1]) - pad, float(min_pt[2]) - pad),
            Gf.Vec3f(float(max_pt[0]) + pad, float(max_pt[1]) + pad, float(max_pt[2]) + pad),
        ])
        self._cloth_extent_attr.Set(extent, Usd.TimeCode.Default())

    def _handle_episode_start(self, msg: dict) -> dict:
        self._episode_id = msg.get("episode_id", 0)
        self._output_dir = msg.get("output_dir", "")
        print(f"[Kit][v7] Episode {self._episode_id} started, output: {self._output_dir}")

        # Apply per-episode lighting randomization (if provided)
        lighting = msg.get("lighting")
        if lighting:
            self._apply_lighting(lighting)

        # Apply per-episode texture randomization (if provided)
        texture = msg.get("texture")
        if texture:
            self._apply_textures(texture)

        return {"type": MSG_EPISODE_START_REPLY, "status": "ok"}

    def _apply_textures(self, texture: dict):
        """Apply per-episode texture parameters from TextureRandomizer to USD scene.

        Args:
            texture: {"table_material_path": str, "table_variant_set": str,
                      "table_variant_name": str}
        """
        stage = self.runtime.usd_context.get_stage()
        if stage is None:
            print("[Kit][v7] WARNING: No stage for texture update")
            return
        material_path = texture.get("table_material_path")
        variant_set_name = texture.get("table_variant_set", "Look")
        variant_name = texture.get("table_variant_name")
        if not material_path or not variant_name:
            return
        prim = stage.GetPrimAtPath(material_path)
        if not prim.IsValid():
            print(f"[Kit][v7] WARNING: material prim not found: {material_path}")
            return
        vsets = prim.GetVariantSets()
        if not vsets.HasVariantSet(variant_set_name):
            print(f"[Kit][v7] WARNING: variant set '{variant_set_name}' not on {material_path}")
            return
        vset = vsets.GetVariantSet(variant_set_name)
        names = set(vset.GetVariantNames())
        # Legacy fallback: old render_state metadata recorded `texture_NNNN`; the
        # unified USDA uses split-prefixed names. If the requested name isn't
        # present, try `texture_seen_NNNN` then `texture_unseen_NNNN`.
        candidate = variant_name
        if candidate not in names and variant_name.startswith("texture_") and "_seen_" not in variant_name and "_unseen_" not in variant_name:
            tail = variant_name[len("texture_"):]
            for guess in (f"texture_seen_{tail}", f"texture_unseen_{tail}"):
                if guess in names:
                    candidate = guess
                    print(f"[Kit][v7] texture variant '{variant_name}' not found, falling back to '{guess}'")
                    break
        ok = vset.SetVariantSelection(candidate)
        print(f"[Kit][v7] Applied texture: {variant_set_name}={candidate} ok={ok}")

    def _apply_lighting(self, lighting: dict):
        """Apply lighting parameters from LightRandomizer to USD scene.

        Args:
            lighting: {"rectlight_exposure_offset": float, "rectlight_color": [r,g,b],
                       "dome_intensity": float, "dome_color": [r,g,b]}
        """
        from pxr import Gf, Usd

        stage = self.runtime.usd_context.get_stage()
        if stage is None:
            print("[Kit][v7] WARNING: No stage for lighting update")
            return

        # Cache baseline values on first call so subsequent episodes don't accumulate.
        if not hasattr(self, "_lighting_baseline"):
            self._lighting_baseline = {}
            for rect_path in ["/RectLight", "/RectLight_01"]:
                prim = stage.GetPrimAtPath(rect_path)
                if not prim.IsValid():
                    continue
                exp_attr = prim.GetAttribute("inputs:exposure")
                base_exp = float(exp_attr.Get()) if (exp_attr and exp_attr.IsValid() and exp_attr.Get() is not None) else 0.0
                self._lighting_baseline[rect_path] = {"exposure": base_exp}
            dome_prim = stage.GetPrimAtPath("/World/DomeLight")
            if dome_prim.IsValid():
                int_attr = dome_prim.GetAttribute("inputs:intensity")
                base_int = float(int_attr.Get()) if (int_attr and int_attr.IsValid() and int_attr.Get() is not None) else 0.0
                self._lighting_baseline["/World/DomeLight"] = {"intensity": base_int}

        # RectLight exposure (absolute = baseline + offset) + color
        exposure_offset = lighting.get("rectlight_exposure_offset", 0.0)
        color_rgb = lighting.get("rectlight_color")
        for rect_path in ["/RectLight", "/RectLight_01"]:
            prim = stage.GetPrimAtPath(rect_path)
            if not prim.IsValid():
                print(f"[Kit][v7] DEBUG: RectLight prim invalid at {rect_path}")
                continue
            exposure_attr = prim.GetAttribute("inputs:exposure")
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if exposure_attr and exposure_attr.IsValid():
                base_exposure = self._lighting_baseline.get(rect_path, {}).get("exposure", 0.0)
                target = float(base_exposure + exposure_offset)
                exposure_attr.Set(target)
                got = exposure_attr.Get()
                print(f"[Kit][v7] DEBUG: {rect_path} exposure base={base_exposure} target={target} read_back={got}")
            if intensity_attr and intensity_attr.IsValid():
                # Also scale intensity directly as a fallback in case PathTracer ignores live exposure changes.
                base_intensity = self._lighting_baseline.setdefault(rect_path, {}).get("intensity")
                if base_intensity is None:
                    base_intensity = float(intensity_attr.Get() or 0.0)
                    self._lighting_baseline[rect_path]["intensity"] = base_intensity
                intensity_attr.Set(float(base_intensity * (2.0 ** exposure_offset)))
                print(f"[Kit][v7] DEBUG: {rect_path} intensity base={base_intensity} target={base_intensity * (2.0 ** exposure_offset):.3f}")
            if color_rgb:
                color_attr = prim.GetAttribute("inputs:color")
                if color_attr and color_attr.IsValid():
                    color_attr.Set(Gf.Vec3f(*color_rgb))

        # DomeLight intensity (absolute) + color
        dome_intensity = lighting.get("dome_intensity")
        dome_color = lighting.get("dome_color")
        dome_prim = stage.GetPrimAtPath("/World/DomeLight")
        if dome_prim.IsValid():
            if dome_intensity is not None:
                intensity_attr = dome_prim.GetAttribute("inputs:intensity")
                if intensity_attr and intensity_attr.IsValid():
                    intensity_attr.Set(float(dome_intensity))
            if dome_color:
                color_attr = dome_prim.GetAttribute("inputs:color")
                if color_attr and color_attr.IsValid():
                    color_attr.Set(Gf.Vec3f(*dome_color))

        print(f"[Kit][v7] Applied lighting: exposure_offset={exposure_offset:+.2f}, "
              f"dome_intensity={dome_intensity}")

    def _disable_physics(self, stage, UsdPhysics):
        """Disable physics to prevent simulation overriding animation."""
        from pxr import Usd, UsdGeom

        physics_prim = stage.GetPrimAtPath(_USD_PHYSICS_SCENE)
        if physics_prim.IsValid():
            physics_prim.SetActive(False)
            print("[Kit][v7] Deactivated physics scene")

        particle_prim = stage.GetPrimAtPath(_USD_PARTICLE_SYSTEM)
        if particle_prim.IsValid():
            particle_prim.SetActive(False)
            print("[Kit][v7] Disabled particle system")

        _cloth_path = getattr(self, '_cloth_mesh_path', _USD_CLOTH_MESH_PATH)
        cloth_prim = stage.GetPrimAtPath(_cloth_path)
        if cloth_prim.IsValid():
            UsdGeom.Imageable(cloth_prim).MakeVisible()
            print(f"[Kit][v7] Forced cloth mesh visible at {_cloth_path}")
        cloth_parent = stage.GetPrimAtPath("/World/Cloth")
        if cloth_parent.IsValid():
            UsdGeom.Imageable(cloth_parent).MakeVisible()

        root_joint_prim = stage.GetPrimAtPath(f"{_USD_ROBOT_ROOT}/root_joint")
        if root_joint_prim.IsValid():
            try:
                root_joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print("[Kit][v7] Removed ArticulationRootAPI")
            except Exception:
                pass

        table_prim = stage.GetPrimAtPath(_USD_TABLE_PATH)
        if table_prim.IsValid():
            for prim in Usd.PrimRange(table_prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                    except Exception:
                        pass
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    try:
                        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                    except Exception:
                        pass
            print("[Kit][v7] Disabled table physics")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args(_extract_user_argv(sys.argv[1:]))

    runtime = KitRuntime()
    runtime.set_renderer(args.renderer, args.spp, optix_denoiser=args.optix_denoiser)
    runtime.update(5)

    if args.fabric_mode:
        zmq_endpoint = args.zmq_endpoint
        if not zmq_endpoint:
            raise SystemExit("[Kit] --fabric-mode requires --zmq-endpoint")

        try:
            import zmq  # noqa: F401
        except ImportError:
            # Kit's embedded Python (kernel/py) omits pip site-packages from sys.path.
            # Locate them via carb: kernel/py/carb/ -> kit_root/ -> python/lib/pyX.Y/site-packages/
            # (fix ported from deformable_bench omniverse_kit_render_v7.py)
            import pathlib as _pathlib
            import sys as _sys

            try:
                import carb as _carb

                _kit_root = _pathlib.Path(_carb.__file__).parent.parent.parent.parent
                _pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
                _sp = _kit_root / "python" / "lib" / _pyver / "site-packages"
                if _sp.is_dir() and str(_sp) not in _sys.path:
                    _sys.path.insert(0, str(_sp))
                import zmq  # noqa: F401
            except ImportError:
                raise SystemExit("[Kit] pyzmq is required for fabric mode. Install: pip install pyzmq")

        receiver = FabricReceiver(runtime, args)
        receiver.run(zmq_endpoint)

    else:
        # Fallback to v3 batch/stream mode
        if not args.usd_dir:
            raise SystemExit("[Kit] Non-fabric mode requires --usd-dir")

        try:
            v3_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, v3_dir)
            from omniverse_kit_render_v3 import render_episode, run_stream_server, _list_episode_dirs
            usd_dir = Path(args.usd_dir).expanduser().resolve()

            if args.stream_mode:
                run_stream_server(runtime, args, usd_dir)
            else:
                episode_dirs = _list_episode_dirs(usd_dir, args.episodes)
                for ep_dir in episode_dirs:
                    render_episode(runtime, args, ep_dir)
        except ImportError as e:
            raise SystemExit(f"[Kit] Failed to import v3 functions: {e}")

    if args.quit_on_complete:
        runtime.post_quit()


# Kit --exec entry point
try:
    main()
except SystemExit as e:
    print(f"[Kit] Exit: {e}")
except Exception as e:
    print(f"[Kit] Fatal error: {e}")
    import traceback
    traceback.print_exc()
