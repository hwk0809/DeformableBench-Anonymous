"""
Shared Memory Bridge for MuJoCo → Omniverse Kit online rendering.

This module provides:
1. SHM protocol: double-buffered shared memory layout for cloth vertices + robot link poses
2. ShmFabricBridge: MuJoCo-side writer that sends frame data via SHM + ZMQ signaling

Usage:
    bridge = ShmFabricBridge(num_cloth_verts=14065, num_links=18)
    bridge.connect(zmq_endpoint)
    bridge.init_scene(scene_usd_path, cameras=["static_cam"], ...)
    for frame_idx in range(num_frames):
        reply = bridge.write_frame(env, frame_idx, episode_id=0)
    bridge.close()

Requires: pyzmq (pip install pyzmq)
"""

from __future__ import annotations

import json
import os
import struct
import time
from multiprocessing import shared_memory

import numpy as np

try:
    import zmq
except ImportError:
    raise ImportError(
        "pyzmq is required for SHM+Fabric rendering. "
        "Install with: conda run -n deformable pip install pyzmq"
    )

from loguru import logger


# ===========================================================================
# Constants (mirrored from scene_animation_injector.py)
# ===========================================================================

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

_USD_ROBOT_ROOT = "/World/dualarm_piper"
_USD_CLOTH_MESH_PATH = "/World/Cloth/silk_35_aligned/silk_35_obj"
_USD_PARTICLE_SYSTEM = "/World/Cloth/particleSystem"
_USD_PHYSICS_SCENE = "/physicsScene"
_USD_TABLE_PATH = "/World/table"
_MUJOCO_TO_USD_OFFSET = np.array([0.0, 0.0, 0.0])

# Link body names in deterministic order (Python 3.7+ dict insertion order)
LINK_BODY_NAMES = list(_MUJOCO_BODY_TO_USD_LINK.keys())
LINK_USD_NAMES = list(_MUJOCO_BODY_TO_USD_LINK.values())
NUM_LINKS = len(LINK_BODY_NAMES)  # 18

# ===========================================================================
# SHM Protocol
# ===========================================================================

# Header layout (64 bytes, packed)
#   write_seq: uint64 (8)
#   num_cloth_verts: uint32 (4)
#   num_links: uint32 (4)
#   frame_index: uint32 (4)
#   episode_id: uint32 (4)
#   shutdown_flag: uint8 (1)
#   buffer_index: uint8 (1)
#   _pad: 38 bytes
HEADER_SIZE = 64
HEADER_FORMAT = "<QIIIIBB"  # 26 bytes packed, rest is padding
HEADER_PACKED_SIZE = struct.calcsize(HEADER_FORMAT)  # 26

# Per-buffer layout (double-buffered):
#   cloth_vertices:    float32[num_verts][3]            positions
#   cloth_velocities:  float32[num_verts][3]            finite-diff velocity (motion vectors)
#   cloth_normals:     float32[num_verts][3]            per-vertex normals (saved to NPZ)
#   link_poses:        float64[num_links][7]            [x, y, z, qw, qx, qy, qz]
#   gripper_values:    float32[2]
#   rigid_object_poses: float64[num_rigid_objects][7]   [x, y, z, qw, qx, qy, qz] in MuJoCo world frame
#                                                       (renderer applies this to /World/<prim> xform ops)


# ===========================================================================
# Image SHM Protocol (for rendered image transfer, separate from cloth SHM)
# ===========================================================================

# Header layout (64 bytes, packed)
#   write_seq:    uint64  (8)  frame sequence number
#   num_cameras:  uint32  (4)  number of cameras
#   width:        uint32  (4)  image width
#   height:       uint32  (4)  image height
#   channels:     uint32  (4)  channels per pixel (always 3)
#   image_size:   uint32  (4)  single image bytes = W*H*C
#   buffer_index: uint8   (1)  active double-buffer slot (0 or 1)
#   valid_mask:   uint8   (1)  bitmask: bit i = camera i has valid data
#   _pad:         34 bytes
IMAGE_SHM_HEADER_SIZE = 64
IMAGE_SHM_HEADER_FORMAT = "<QIIIIIBB"  # 30 bytes packed, rest is padding
IMAGE_SHM_HEADER_PACKED_SIZE = struct.calcsize(IMAGE_SHM_HEADER_FORMAT)  # 30


def compute_image_shm_size(num_cameras: int, width: int, height: int, channels: int = 3) -> int:
    """Total SHM size for double-buffered image transfer."""
    image_size = width * height * channels
    per_buffer = num_cameras * image_size
    return IMAGE_SHM_HEADER_SIZE + 2 * per_buffer


class ImageShmLayout:
    """Provides numpy views into image shared memory with double-buffering.

    Layout: [header 64B] [buffer0: cam0_rgb | cam1_rgb | ...] [buffer1: ...]
    Each cam image is (height, width, channels) uint8 contiguous in C order.
    """

    def __init__(self, shm_buf, num_cameras: int, width: int, height: int, channels: int = 3):
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.channels = channels
        self.image_size = width * height * channels
        self._buf = shm_buf
        self._per_buffer = num_cameras * self.image_size

        # Create numpy views: _image_views[buf_idx][cam_idx]
        self._image_views: list[list[np.ndarray]] = []
        for b in range(2):
            buf_start = IMAGE_SHM_HEADER_SIZE + b * self._per_buffer
            cam_views = []
            for c in range(num_cameras):
                offset = buf_start + c * self.image_size
                view = np.ndarray(
                    (height, width, channels),
                    dtype=np.uint8,
                    buffer=shm_buf,
                    offset=offset,
                )
                cam_views.append(view)
            self._image_views.append(cam_views)

    def get_image(self, buf_idx: int, cam_idx: int) -> np.ndarray:
        """Return a zero-copy numpy view of camera image in the given buffer slot."""
        return self._image_views[buf_idx][cam_idx]

    def write_header(self, write_seq: int, buffer_index: int, valid_mask: int):
        packed = struct.pack(
            IMAGE_SHM_HEADER_FORMAT,
            write_seq,
            self.num_cameras,
            self.width,
            self.height,
            self.channels,
            self.image_size,
            buffer_index,
            valid_mask,
        )
        self._buf[:IMAGE_SHM_HEADER_PACKED_SIZE] = packed

    def read_header(self) -> dict:
        packed = bytes(self._buf[:IMAGE_SHM_HEADER_PACKED_SIZE])
        vals = struct.unpack(IMAGE_SHM_HEADER_FORMAT, packed)
        return {
            "write_seq": vals[0],
            "num_cameras": vals[1],
            "width": vals[2],
            "height": vals[3],
            "channels": vals[4],
            "image_size": vals[5],
            "buffer_index": vals[6],
            "valid_mask": vals[7],
        }


# ===========================================================================
# Cloth SHM Protocol
# ===========================================================================

def compute_buffer_size(num_verts: int, num_links: int = NUM_LINKS, num_rigid_objects: int = 0) -> int:
    cloth_bytes = num_verts * 3 * 4  # float32
    cloth_vel_bytes = num_verts * 3 * 4  # float32 (velocities)
    cloth_normal_bytes = num_verts * 3 * 4  # float32 (normals)
    link_bytes = num_links * 7 * 8  # float64
    gripper_bytes = 2 * 4  # float32
    rigid_bytes = num_rigid_objects * 7 * 8  # float64
    return (cloth_bytes + cloth_vel_bytes + cloth_normal_bytes
            + link_bytes + gripper_bytes + rigid_bytes)


def compute_total_shm_size(num_verts: int, num_links: int = NUM_LINKS, num_rigid_objects: int = 0) -> int:
    return HEADER_SIZE + 2 * compute_buffer_size(num_verts, num_links, num_rigid_objects)


class ShmLayout:
    """Provides numpy views into a shared memory buffer with double-buffering."""

    def __init__(self, shm_buf, num_verts: int, num_links: int = NUM_LINKS, num_rigid_objects: int = 0):
        self.num_verts = num_verts
        self.num_links = num_links
        self.num_rigid_objects = num_rigid_objects
        self._buf = shm_buf
        self._buffer_size = compute_buffer_size(num_verts, num_links, num_rigid_objects)

        cloth_bytes = num_verts * 3 * 4
        cloth_vel_bytes = num_verts * 3 * 4
        cloth_normal_bytes = num_verts * 3 * 4
        link_bytes = num_links * 7 * 8
        gripper_bytes = 2 * 4
        rigid_bytes = num_rigid_objects * 7 * 8

        self._cloth_views = []
        self._cloth_vel_views = []
        self._cloth_normal_views = []
        self._link_views = []
        self._gripper_views = []
        self._rigid_views = []

        for i in range(2):
            buf_start = HEADER_SIZE + i * self._buffer_size
            cloth_end = buf_start + cloth_bytes
            vel_end = cloth_end + cloth_vel_bytes
            normal_end = vel_end + cloth_normal_bytes
            link_end = normal_end + link_bytes
            gripper_end = link_end + gripper_bytes
            rigid_end = gripper_end + rigid_bytes

            self._cloth_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=shm_buf, offset=buf_start)
            )
            self._cloth_vel_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=shm_buf, offset=cloth_end)
            )
            self._cloth_normal_views.append(
                np.ndarray((num_verts, 3), dtype=np.float32, buffer=shm_buf, offset=vel_end)
            )
            self._link_views.append(
                np.ndarray((num_links, 7), dtype=np.float64, buffer=shm_buf, offset=normal_end)
            )
            self._gripper_views.append(
                np.ndarray((2,), dtype=np.float32, buffer=shm_buf, offset=link_end)
            )
            # Empty rigid views are still valid numpy zero-row arrays — cleaner than
            # branching every getter on num_rigid_objects.
            self._rigid_views.append(
                np.ndarray((num_rigid_objects, 7), dtype=np.float64, buffer=shm_buf, offset=gripper_end)
                if num_rigid_objects > 0
                else np.zeros((0, 7), dtype=np.float64)
            )

    def get_cloth_vertices(self, buf_idx: int) -> np.ndarray:
        return self._cloth_views[buf_idx]

    def get_cloth_velocities(self, buf_idx: int) -> np.ndarray:
        return self._cloth_vel_views[buf_idx]

    def get_cloth_normals(self, buf_idx: int) -> np.ndarray:
        return self._cloth_normal_views[buf_idx]

    def get_link_poses(self, buf_idx: int) -> np.ndarray:
        return self._link_views[buf_idx]

    def get_gripper_values(self, buf_idx: int) -> np.ndarray:
        return self._gripper_views[buf_idx]

    def get_rigid_object_poses(self, buf_idx: int) -> np.ndarray:
        """(num_rigid_objects, 7) float64 — [x,y,z,qw,qx,qy,qz] per tracked body."""
        return self._rigid_views[buf_idx]

    def write_header(self, write_seq: int, frame_index: int, episode_id: int,
                     shutdown: bool, buffer_index: int):
        packed = struct.pack(
            HEADER_FORMAT,
            write_seq,
            self.num_verts,
            self.num_links,
            frame_index,
            episode_id,
            1 if shutdown else 0,
            buffer_index,
        )
        self._buf[:HEADER_PACKED_SIZE] = packed

    def read_header(self) -> dict:
        packed = bytes(self._buf[:HEADER_PACKED_SIZE])
        vals = struct.unpack(HEADER_FORMAT, packed)
        return {
            "write_seq": vals[0],
            "num_cloth_verts": vals[1],
            "num_links": vals[2],
            "frame_index": vals[3],
            "episode_id": vals[4],
            "shutdown_flag": vals[5],
            "buffer_index": vals[6],
        }


# ===========================================================================
# Coordinate transform
# ===========================================================================

def _batch_world_to_local(
    cloth_pos_world: np.ndarray,
    w2l_mat: np.ndarray | None,
) -> np.ndarray:
    """Vectorized MuJoCo world → USD local transform.

    Gf.Matrix4d row-vector convention: p_local = p_world @ R + T
    where R = mat[:3, :3], T = mat[3, :3]
    """
    pos = cloth_pos_world.astype(np.float64) + _MUJOCO_TO_USD_OFFSET
    if w2l_mat is None:
        return pos.astype(np.float32)
    R = w2l_mat[:3, :3]
    T = w2l_mat[3, :3]
    return (pos @ R + T).astype(np.float32)


def _batch_normals_to_local(
    normals_world: np.ndarray,
    w2l_mat: np.ndarray | None,
) -> np.ndarray:
    """Rotate normals from MuJoCo world space to USD local space.

    Normals are direction vectors — apply rotation only (no translation).
    """
    if w2l_mat is None:
        return normals_world.astype(np.float32)
    R = w2l_mat[:3, :3]
    rotated = normals_world.astype(np.float64) @ R
    # Re-normalize after rotation (handles any numerical drift)
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (rotated / norms).astype(np.float32)


def _compute_reference_normals(
    positions: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute area-weighted per-vertex normals from mesh geometry.

    Used as a reference direction for flip detection — NOT for rendering.
    """
    v0 = positions[triangles[:, 0]]
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # area-weighted
    vert_n = np.zeros((positions.shape[0], 3), dtype=np.float64)
    np.add.at(vert_n, triangles[:, 0], face_n)
    np.add.at(vert_n, triangles[:, 1], face_n)
    np.add.at(vert_n, triangles[:, 2], face_n)
    norms = np.linalg.norm(vert_n, axis=1, keepdims=True)
    return (vert_n / np.maximum(norms, 1e-8)).astype(np.float32)


def _sanitize_normals(
    normals: np.ndarray,
    prev_normals: np.ndarray | None,
    ref_normals: np.ndarray | None = None,
    frame_idx: int = -1,
) -> np.ndarray:
    """Full normal sanitization: filter → flip fix → temporal fallback → re-normalize.

    Steps:
      1. Replace NaN / Inf → 0
      2. Detect zero-length normals
      3. Flip detection: if normal · ref_normal < 0, flip it  (wrong direction → shadow)
      4. Temporal fallback: bad verts → prev frame; first frame → [0,0,1]
      5. Force re-normalize to unit length

    Returns (N, 3) float32 unit normals, guaranteed finite and non-zero.
    """
    n_verts = normals.shape[0]
    result = normals.copy()

    # --- Step 1: filter NaN / Inf ---
    nan_mask = ~np.isfinite(result)
    n_nan = int(nan_mask.any(axis=1).sum())
    if n_nan > 0:
        result[nan_mask] = 0.0

    # --- Step 2: detect zero-length normals ---
    lengths = np.linalg.norm(result, axis=1)
    zero_mask = lengths < 1e-6
    n_zero = int(zero_mask.sum())

    # --- Step 3: flip detection using geometric reference ---
    n_flipped = 0
    if ref_normals is not None:
        # Only check non-zero normals (zero ones will be fixed in step 4)
        valid_mask = ~zero_mask
        dot = np.sum(result[valid_mask] * ref_normals[valid_mask], axis=1)
        flipped_local = dot < 0
        n_flipped = int(flipped_local.sum())
        if n_flipped > 0:
            valid_indices = np.where(valid_mask)[0]
            flip_indices = valid_indices[flipped_local]
            result[flip_indices] *= -1

    # --- Step 4: temporal fallback for zero-length ---
    n_fallback_prev = 0
    n_fallback_default = 0
    if n_zero > 0:
        if prev_normals is not None:
            result[zero_mask] = prev_normals[zero_mask]
            still_bad = np.linalg.norm(result[zero_mask], axis=1) < 1e-6
            n_fallback_prev = n_zero - int(still_bad.sum())
            if np.any(still_bad):
                bad_indices = np.where(zero_mask)[0][still_bad]
                result[bad_indices] = [0.0, 0.0, 1.0]
                n_fallback_default = int(still_bad.sum())
        else:
            result[zero_mask] = [0.0, 0.0, 1.0]
            n_fallback_default = n_zero

    # --- Step 5: force re-normalize entire array ---
    lengths = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(lengths, 1e-8)

    # --- Logging ---
    # 只在真正异常时打: NaN 是严重 bug, 大量坏法线说明仿真有问题.
    # 偶发 1-10 个 flipped 在布料褶皱处很常见, 自动修好即可, 不打扰用户.
    n_bad_total = n_nan + n_zero + n_flipped
    if n_nan > 0 or n_bad_total > 10:
        logger.warning(
            f"[NormalSanitize] frame={frame_idx}: "
            f"{n_bad_total}/{n_verts} bad normals "
            f"(nan={n_nan}, zero={n_zero}, flipped={n_flipped}) → "
            f"fixed: flip={n_flipped}, prev_frame={n_fallback_prev}, "
            f"default_up={n_fallback_default}"
        )

    return result.astype(np.float32)


# ===========================================================================
# ZMQ Message Types
# ===========================================================================

MSG_INIT = "INIT"
MSG_INIT_REPLY = "INIT_REPLY"
MSG_TICK = "TICK"
MSG_FRAME_DONE = "FRAME_DONE"
MSG_EPISODE_START = "EPISODE_START"
MSG_EPISODE_START_REPLY = "EPISODE_START_REPLY"
MSG_SHUTDOWN = "SHUTDOWN"
MSG_SHUTDOWN_REPLY = "SHUTDOWN_REPLY"


def _zmq_send(socket, msg: dict):
    socket.send_json(msg)


def _zmq_recv(socket, timeout_ms: int | None = None) -> dict:
    if timeout_ms is not None:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))
        if socket not in events:
            raise TimeoutError(f"ZMQ recv timed out after {timeout_ms}ms")
    return socket.recv_json()


# ===========================================================================
# ShmFabricBridge (MuJoCo side)
# ===========================================================================

class ShmFabricBridge:
    """MuJoCo-side bridge: writes frame data to SHM and signals Kit via ZMQ."""

    def __init__(
        self,
        num_cloth_verts: int,
        num_links: int = NUM_LINKS,
        num_rigid_objects: int = 0,
        shm_name: str | None = None,
        attach: bool = False,
        image_shm: bool = False,
        num_cameras: int = 0,
        image_width: int = 1280,
        image_height: int = 720,
    ):
        """
        Args:
            num_rigid_objects: Number of free rigid bodies the env will track and
                publish per-frame xforms for (e.g. snickers). Owner and worker
                must agree; pass 0 for legacy cloth-only flow.
            shm_name: Custom SHM name. Default: ``deformable_bench_{pid}``.
            attach: If True, open an existing SHM (worker mode) instead of
                creating a new one (owner/orchestrator mode).
            image_shm: If True, create/attach a second SHM segment for rendered
                images (eliminates PNG file I/O).
            num_cameras: Number of cameras (required when image_shm=True).
            image_width: Image width in pixels.
            image_height: Image height in pixels.
        """
        self.num_verts = num_cloth_verts
        self.num_links = num_links
        self.num_rigid_objects = num_rigid_objects
        self._write_seq = 0
        self._is_owner = not attach

        # Shared memory
        self._shm_name = shm_name or f"deformable_bench_{os.getpid()}"
        total_size = compute_total_shm_size(num_cloth_verts, num_links, num_rigid_objects)
        if attach:
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=False
            )
            # Prevent resource_tracker from unlinking SHM when this
            # (attacher) process exits — only the owner should unlink.
            from multiprocessing.resource_tracker import unregister
            unregister(self._shm._name, "shared_memory")
            logger.info(f"[ShmBridge] Attached to SHM '{self._shm_name}'")
        else:
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=True, size=total_size
            )
            logger.info(f"[ShmBridge] Created SHM '{self._shm_name}', size={total_size} bytes")
        self._layout = ShmLayout(self._shm.buf, num_cloth_verts, num_links, num_rigid_objects)

        # Rigid object tracking (set via set_tracked_rigid_objects)
        self._tracked_rigid_body_names: list[str] = []
        self._tracked_rigid_body_ids: list[int] | None = None
        self._tracked_rigid_usd_prims: list[str] = []

        # Image SHM (optional, for zero-copy rendered image transfer)
        self._image_shm: shared_memory.SharedMemory | None = None
        self._image_layout: ImageShmLayout | None = None
        self._image_shm_ready = False  # confirmed by Kit in INIT_REPLY
        self._image_shm_cameras: list[str] = []  # camera order in SHM slots

        if image_shm and num_cameras > 0:
            img_shm_name = f"{self._shm_name}_img"
            img_size = compute_image_shm_size(num_cameras, image_width, image_height)
            try:
                if attach:
                    self._image_shm = shared_memory.SharedMemory(
                        name=img_shm_name, create=False
                    )
                    from multiprocessing.resource_tracker import unregister
                    unregister(self._image_shm._name, "shared_memory")
                    logger.info(f"[ShmBridge] Attached to image SHM '{img_shm_name}'")
                else:
                    self._image_shm = shared_memory.SharedMemory(
                        name=img_shm_name, create=True, size=img_size
                    )
                    logger.info(
                        f"[ShmBridge] Created image SHM '{img_shm_name}', "
                        f"size={img_size} bytes ({num_cameras} cams, {image_width}x{image_height})"
                    )
                self._image_layout = ImageShmLayout(
                    self._image_shm.buf, num_cameras, image_width, image_height
                )
            except Exception as e:
                logger.warning(f"[ShmBridge] Image SHM failed, falling back to PNG I/O: {e}")
                self._image_shm = None
                self._image_layout = None

        # ZMQ
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._connected = False

        # Transforms (populated from INIT_REPLY or set_transforms)
        self._w2l_mat: np.ndarray | None = None
        self._robot_root_offset: np.ndarray = np.zeros(3, dtype=np.float64)

        # MuJoCo body IDs (populated on first write_frame)
        self._mj_body_ids: list[int] | None = None

        # Cloth normals: collect & sanitize for NPZ saving (off by default)
        self.enable_cloth_normals: bool = False
        self._cloth_triangles: np.ndarray | None = None  # cached topology for flip detection

        # Per-frame state (reset between episodes)
        self._prev_cloth_local: np.ndarray | None = None
        self._prev_frame_time: float | None = None
        self._prev_normals_local: np.ndarray | None = None

    @property
    def zmq_endpoint(self) -> str:
        return f"ipc:///tmp/{self._shm_name}.ipc"

    @property
    def image_shm_available(self) -> bool:
        """True if image SHM layout is initialized (created or attached).

        Per-frame Kit confirmation is checked separately via reply.get("image_shm").
        """
        return self._image_layout is not None

    @property
    def image_shm_name(self) -> str | None:
        """Name of the image SHM segment, or None if not created."""
        return self._image_shm._name if self._image_shm is not None else None

    @property
    def image_shm_cameras(self) -> list[str]:
        """Camera order confirmed by Kit for image SHM slots."""
        return list(self._image_shm_cameras)

    def read_images(self, buf_idx: int, camera_names: list[str]) -> dict[str, np.ndarray]:
        """Read rendered images from image SHM.

        Args:
            buf_idx: Double-buffer index (from FRAME_DONE reply).
            camera_names: Camera names in SHM slot order.

        Returns:
            Dict mapping camera name to (H, W, 3) uint8 numpy array (copied).
        """
        if self._image_layout is None:
            raise RuntimeError("Image SHM not available")
        result = {}
        for i, cam_name in enumerate(camera_names):
            if i >= self._image_layout.num_cameras:
                break
            # Copy out of SHM — the buffer will be overwritten on next frame
            result[cam_name] = np.array(self._image_layout.get_image(buf_idx, i))
        return result

    def connect(self, zmq_endpoint: str | None = None):
        endpoint = zmq_endpoint or self.zmq_endpoint
        self._zmq_sock.connect(endpoint)
        self._connected = True
        logger.info(f"[ShmBridge] ZMQ REQ connected to {endpoint}")

    def init_scene(
        self,
        scene_usd_path: str,
        cameras: list[str],
        renderer: str = "PathTracing",
        spp: int = 64,
        width: int = 1280,
        height: int = 720,
        frame_updates: int = 6,
        camera_updates: int = 6,
        capture_timeout: float = 30.0,
        capture_retries: int = 2,
        capture_retry_updates: int = 10,
        optix_denoiser: bool = False,
        cloth_normals: bool = False,
        cloth_subdivision: str = "none",
        cloth_subdivision_level: int = 0,
        cloth_mesh_path: str | None = None,
    ) -> dict:
        """Send INIT to Kit, receive transforms."""
        msg = {
            "type": MSG_INIT,
            "shm_name": self._shm_name,
            "num_verts": self.num_verts,
            "num_links": self.num_links,
            "num_rigid_objects": self.num_rigid_objects,
            "rigid_usd_prims": self._tracked_rigid_usd_prims,
            "scene_usd_path": os.path.abspath(scene_usd_path),
            "cameras": cameras,
            "renderer": renderer,
            "spp": spp,
            "width": width,
            "height": height,
            "frame_updates": frame_updates,
            "camera_updates": camera_updates,
            "capture_timeout": capture_timeout,
            "capture_retries": capture_retries,
            "capture_retry_updates": capture_retry_updates,
            "optix_denoiser": optix_denoiser,
            "cloth_normals": cloth_normals,
            "cloth_subdivision": cloth_subdivision,
            "cloth_subdivision_level": cloth_subdivision_level,
        }

        # Include cloth mesh path for Kit to locate USD prim
        if cloth_mesh_path is not None:
            msg["cloth_mesh_path"] = cloth_mesh_path

        # Include image SHM name so Kit can attach
        if self._image_shm is not None:
            msg["image_shm_name"] = self._image_shm._name

        _zmq_send(self._zmq_sock, msg)
        reply = _zmq_recv(self._zmq_sock, timeout_ms=600_000)

        if reply.get("status") != "ok":
            raise RuntimeError(f"[ShmBridge] INIT failed: {reply}")

        # Parse image SHM confirmation from Kit
        if reply.get("image_shm_ready"):
            self._image_shm_ready = True
            self._image_shm_cameras = reply.get("image_shm_cameras", [])
            logger.info(
                f"[ShmBridge] Image SHM confirmed by Kit, "
                f"cameras={self._image_shm_cameras}"
            )
        elif self._image_shm is not None:
            logger.warning("[ShmBridge] Kit did not confirm image SHM, falling back to PNG I/O")

        # Parse transforms
        w2l_flat = reply.get("w2l_matrix")
        if w2l_flat is not None:
            self._w2l_mat = np.array(w2l_flat, dtype=np.float64).reshape(4, 4)
        else:
            self._w2l_mat = None

        offset = reply.get("robot_root_offset")
        if offset is not None:
            self._robot_root_offset = np.array(offset, dtype=np.float64)

        logger.info(
            f"[ShmBridge] INIT OK. w2l_mat={'set' if self._w2l_mat is not None else 'None'}, "
            f"robot_root_offset={self._robot_root_offset}"
        )
        return reply

    def set_transforms(self, w2l_matrix, robot_root_offset):
        """Set transforms without calling init_scene (for worker/attach mode)."""
        if w2l_matrix is not None:
            self._w2l_mat = np.array(w2l_matrix, dtype=np.float64).reshape(4, 4)
        else:
            self._w2l_mat = None
        self._robot_root_offset = np.array(robot_root_offset, dtype=np.float64)

    def set_tracked_rigid_objects(self, body_names: list[str], usd_prim_paths: list[str]):
        """Configure which MuJoCo free rigid bodies are tracked per frame.

        body_names[i] gets published into rigid_object_poses[i] each frame; the
        renderer applies that pose to usd_prim_paths[i] in the scene. Call before
        init_scene() so the prim mapping is included in the INIT message.
        """
        if len(body_names) != self.num_rigid_objects:
            raise ValueError(
                f"set_tracked_rigid_objects: expected {self.num_rigid_objects} "
                f"body_names, got {len(body_names)}"
            )
        if len(usd_prim_paths) != self.num_rigid_objects:
            raise ValueError(
                f"set_tracked_rigid_objects: expected {self.num_rigid_objects} "
                f"usd_prim_paths, got {len(usd_prim_paths)}"
            )
        self._tracked_rigid_body_names = list(body_names)
        self._tracked_rigid_usd_prims = list(usd_prim_paths)
        # Force re-resolve on next write_frame
        self._tracked_rigid_body_ids = None

    def _resolve_body_ids(self, model):
        """Resolve MuJoCo body IDs for links and tracked rigid objects (cached)."""
        import mujoco as mj

        self._mj_body_ids = []
        for mj_body_name in LINK_BODY_NAMES:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, mj_body_name)
            if body_id < 0:
                logger.warning(f"[ShmBridge] MuJoCo body '{mj_body_name}' not found")
            self._mj_body_ids.append(body_id)

        self._tracked_rigid_body_ids = []
        for body_name in self._tracked_rigid_body_names:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, body_name)
            if body_id < 0:
                raise RuntimeError(
                    f"[ShmBridge] tracked rigid body '{body_name}' not found in MuJoCo model"
                )
            self._tracked_rigid_body_ids.append(body_id)

    def write_frame(
        self,
        env,
        frame_idx: int,
        episode_id: int = 0,
        timeout_ms: int = 180_000,
    ) -> dict:
        """Extract frame data from env, write to SHM, send ZMQ TICK, wait for FRAME_DONE.

        Returns the Kit reply dict with timing info.
        """
        t_start = time.perf_counter()

        # Resolve body IDs on first call
        if self._mj_body_ids is None:
            self._resolve_body_ids(env.model)

        seq = self._write_seq + 1
        buf_idx = seq % 2

        # 1. Cloth vertices: transform to USD local coords
        cloth_world = env.get_cloth_position()  # (N, 3) float64
        t_cloth_start = time.perf_counter()
        cloth_local = _batch_world_to_local(cloth_world, self._w2l_mat)  # (N, 3) float32
        t_cloth_end = time.perf_counter()

        # 2. Write cloth positions to SHM
        self._layout.get_cloth_vertices(buf_idx)[:] = cloth_local

        # 2b. Compute finite-difference velocity for motion vectors
        current_time = float(env.data.time)
        if self._prev_cloth_local is not None and self._prev_frame_time is not None:
            dt = current_time - self._prev_frame_time
            if dt > 1e-8:
                cloth_vel = (cloth_local - self._prev_cloth_local) / np.float32(dt)
            else:
                cloth_vel = np.zeros_like(cloth_local)
        else:
            cloth_vel = np.zeros_like(cloth_local)
        self._layout.get_cloth_velocities(buf_idx)[:] = cloth_vel
        self._prev_cloth_local = cloth_local.copy()
        self._prev_frame_time = current_time

        # 2c. Cloth normals: Style3D → sanitize (NaN/zero/flip) → rotate to local
        if self.enable_cloth_normals and hasattr(env, 'get_cloth_normals'):
            try:
                # Cache triangle topology once (static mesh connectivity)
                if self._cloth_triangles is None and hasattr(env, 'get_cloth_triangles'):
                    tri = env.get_cloth_triangles()
                    if tri is not None:
                        self._cloth_triangles = np.asarray(tri, dtype=np.int32).reshape(-1, 3)
                        logger.info(f"[NormalSanitize] Cached {len(self._cloth_triangles)} triangles for flip detection")

                normals_world = env.get_cloth_normals()
                if normals_world is not None:
                    normals_local = _batch_normals_to_local(normals_world, self._w2l_mat)

                    # Compute geometric reference for flip detection
                    ref_normals = None
                    if self._cloth_triangles is not None:
                        ref_normals = _compute_reference_normals(cloth_local, self._cloth_triangles)

                    normals_local = _sanitize_normals(
                        normals_local, self._prev_normals_local,
                        ref_normals=ref_normals, frame_idx=frame_idx,
                    )
                    self._layout.get_cloth_normals(buf_idx)[:] = normals_local
                    self._prev_normals_local = normals_local.copy()
                else:
                    logger.warning(f"[NormalSanitize] frame={frame_idx}: get_cloth_normals() returned None")
                    self._layout.get_cloth_normals(buf_idx)[:] = 0.0
            except Exception as e:
                logger.warning(f"[NormalSanitize] frame={frame_idx}: exception: {e}")
                self._layout.get_cloth_normals(buf_idx)[:] = 0.0
        else:
            self._layout.get_cloth_normals(buf_idx)[:] = 0.0

        # 3. Robot link poses
        link_view = self._layout.get_link_poses(buf_idx)
        for i, body_id in enumerate(self._mj_body_ids):
            if body_id < 0:
                continue
            world_pos = env.data.xpos[body_id].copy() + _MUJOCO_TO_USD_OFFSET
            local_pos = world_pos - self._robot_root_offset
            quat = env.data.xquat[body_id]  # [qw, qx, qy, qz]
            link_view[i, :3] = local_pos
            link_view[i, 3:] = quat

        # 4. Gripper values
        gripper_view = self._layout.get_gripper_values(buf_idx)
        if hasattr(env, 'get_gripper_value'):
            try:
                gripper_left, gripper_right = env.get_gripper_value()
                gripper_view[0] = float(gripper_left)
                gripper_view[1] = float(gripper_right)
            except Exception:
                gripper_view[:] = 0.0
        else:
            gripper_view[:] = 0.0

        # 4b. Rigid object poses (snickers-like free bodies tracked at /World/<prim>)
        if self.num_rigid_objects > 0 and self._tracked_rigid_body_ids:
            rigid_view = self._layout.get_rigid_object_poses(buf_idx)
            for i, body_id in enumerate(self._tracked_rigid_body_ids):
                # Rigid prims live at /World/<prim> directly, parented under
                # an identity /World — no w2l_mat / robot_root_offset subtraction.
                rigid_view[i, :3] = env.data.xpos[body_id] + _MUJOCO_TO_USD_OFFSET
                rigid_view[i, 3:] = env.data.xquat[body_id]   # MuJoCo wxyz

        # 5. Update header
        self._layout.write_header(seq, frame_idx, episode_id, False, buf_idx)
        self._write_seq = seq

        t_shm_done = time.perf_counter()

        # 6. ZMQ send TICK
        _zmq_send(self._zmq_sock, {
            "type": MSG_TICK,
            "seq": int(seq),
            "buffer_idx": int(buf_idx),
            "frame_idx": int(frame_idx),
            "episode_id": int(episode_id),
        })

        # 7. Wait for FRAME_DONE
        reply = _zmq_recv(self._zmq_sock, timeout_ms=timeout_ms)

        t_end = time.perf_counter()

        # Attach local timing to reply
        reply["_local_timing"] = {
            "cloth_transform_ms": (t_cloth_end - t_cloth_start) * 1000,
            "shm_write_ms": (t_shm_done - t_cloth_end) * 1000,
            "zmq_roundtrip_ms": (t_end - t_shm_done) * 1000,
            "total_ms": (t_end - t_start) * 1000,
        }

        return reply

    def reset_velocity_state(self):
        """Reset finite-difference velocity and normals tracking. Call between episodes."""
        self._prev_cloth_local = None
        self._prev_frame_time = None
        self._prev_normals_local = None

    def start_episode(self, episode_id: int, output_dir: str,
                      camera_extrinsics: dict | None = None,
                      lighting: dict | None = None,
                      texture: dict | None = None) -> dict:
        """Notify Kit of new episode.

        Args:
            camera_extrinsics: Optional per-camera extrinsics to override USD defaults.
                Format: {cam_name: {"pos": [x,y,z], "quat_wxyz": [w,x,y,z]}}
            lighting: Optional per-episode lighting parameters from LightRandomizer.
                Format: {"rectlight_exposure_offset": float, "rectlight_color": [r,g,b],
                         "dome_intensity": float, "dome_color": [r,g,b]}
            texture: Optional per-episode texture parameters from TextureRandomizer.
                Format: {"table_material_path": str, "table_variant_set": str,
                         "table_variant_name": str}
        """
        self.reset_velocity_state()
        msg = {
            "type": MSG_EPISODE_START,
            "episode_id": int(episode_id),
            "output_dir": output_dir,
        }
        if camera_extrinsics:
            msg["camera_extrinsics"] = camera_extrinsics
        if lighting:
            msg["lighting"] = lighting
        if texture:
            msg["texture"] = texture
        _zmq_send(self._zmq_sock, msg)
        return _zmq_recv(self._zmq_sock, timeout_ms=30_000)

    def reconnect(self):
        """Tear down and recreate ZMQ socket so we can connect to a new Kit process."""
        try:
            self._zmq_sock.close()
            self._zmq_ctx.term()
        except Exception:
            pass
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._connected = False
        logger.info("[ShmBridge] ZMQ socket recreated for reconnection")

    def close(self):
        """Clean up resources. Owner sends SHUTDOWN and unlinks SHM; attacher only disconnects."""
        if self._is_owner:
            try:
                if self._connected:
                    _zmq_send(self._zmq_sock, {"type": MSG_SHUTDOWN})
                    try:
                        _zmq_recv(self._zmq_sock, timeout_ms=10_000)
                    except TimeoutError:
                        logger.warning("[ShmBridge] SHUTDOWN reply timed out")
            except Exception as e:
                logger.warning(f"[ShmBridge] Error during shutdown: {e}")
        try:
            self._zmq_sock.close()
            self._zmq_ctx.term()
        except Exception:
            pass
        try:
            self._shm.close()
            if self._is_owner:
                self._shm.unlink()
        except Exception:
            pass
        # Clean up image SHM
        if self._image_shm is not None:
            try:
                self._image_shm.close()
                if self._is_owner:
                    self._image_shm.unlink()
            except Exception:
                pass
            self._image_shm = None
            self._image_layout = None
        logger.info(f"[ShmBridge] Closed ({'owner' if self._is_owner else 'attacher'})")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
