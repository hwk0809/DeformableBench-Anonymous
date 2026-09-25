"""Web viewer adapter for streaming Newton primitives to Three.js.

The viewer implements Newton's viewer protocol without opening a native window.
It collects model shapes and transient viewer logs as JSON-friendly primitive
payloads consumed by the Reflex frontend.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


def _vec(values, length: int | None = None) -> list[float]:
    """Convert a Warp/Numpy vector-like value to a JSON-friendly float list."""

    if hasattr(values, "numpy"):
        values = values.numpy()
    out = [float(v) for v in values]
    return out if length is None else out[:length]


def _transform(values) -> tuple[list[float], list[float]]:
    """Split a Newton/Warp transform into position and quaternion lists."""

    raw = _vec(values, 7)
    return raw[:3], raw[3:7]


def _array(values) -> np.ndarray | None:
    """Best-effort conversion from Warp/Numpy values to a Numpy array."""

    if values is None:
        return None
    try:
        if hasattr(values, "numpy"):
            values = values.numpy()
        return np.asarray(values)
    except Exception:
        return None


def _mesh_payload(mesh: newton.Mesh | None) -> dict[str, Any] | None:
    """Serialize a Newton mesh into vertices and indices for Three.js."""

    if mesh is None:
        return None
    vertices = _array(mesh.vertices)
    indices = _array(mesh.indices)
    if vertices is None or indices is None:
        return None
    payload = {
        "vertices": vertices.astype(float).reshape(-1, 3).round(6).tolist(),
        "indices": indices.astype(int).reshape(-1).tolist(),
    }
    uvs = _array(getattr(mesh, "uvs", None))
    if uvs is not None and uvs.size:
        payload["uvs"] = uvs.astype(float).reshape(-1, 2).round(6).tolist()
    return payload


def _heightfield_mesh_payload(heightfield: newton.Heightfield | None) -> dict[str, Any] | None:
    """Convert a Newton heightfield to a mesh payload with normalized UVs."""

    if heightfield is None:
        return None
    actual_heights = heightfield.min_z + heightfield.data * (heightfield.max_z - heightfield.min_z)
    mesh = newton.Mesh.create_heightfield(
        heightfield=actual_heights.T,
        extent_x=heightfield.hx * 2.0,
        extent_y=heightfield.hy * 2.0,
        ground_z=heightfield.min_z,
        compute_uvs=True,
        compute_inertia=False,
    )
    return _mesh_payload(mesh)


class ViewerWeb(ViewerNull):
    """Headless Newton viewer that exports viewer-style primitives as JSON.

    Newton examples already know how to talk to a viewer through methods like
    ``set_model()``, ``log_mesh()``, and ``log_points()``. By extending
    ``ViewerNull`` we reuse that protocol without opening a native OpenGL
    window.
    """

    def __init__(self):
        super().__init__(num_frames=10**12)
        self.camera_state = {
            "position": [10.0, -1.3, 2.0],
            "pitch": 0.0,
            "yaw": -180.0,
            "fov": 70.0,
        }
        self.logged_primitives: dict[str, dict[str, Any]] = {}
        self.model_mesh_payloads: dict[str, dict[str, Any] | None] = {}
        self.log_version = 0
        self.capture_requested = False

    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        """Store the example camera hint for future frontend use."""

        self.camera_state = {
            "position": _vec(pos, 3),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "fov": 70.0,
        }

    def begin_frame(self, time: float):
        """Start a fresh viewer frame and invalidate dynamic logged geometry."""

        if not self.capture_requested:
            return
        self.logged_primitives = {}
        self.log_version += 1

    def request_capture(self):
        """Enable one viewer logging pass for the next external frame request."""

        self.capture_requested = True

    def finish_capture(self):
        """Disable viewer logging after the requested frame has been captured."""

        self.capture_requested = False

    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """Capture a logged mesh as a transient Three.js mesh primitive."""

        if hidden or not self.capture_requested:
            return
        vertices = _array(points)
        mesh_indices = _array(indices)
        if vertices is None or mesh_indices is None:
            return
        primitive_id = f"log-mesh:{name}"
        # STABLE mesh_key (no per-frame version): a deforming mesh is ONE mesh whose
        # vertices change, not a new mesh each frame. With a stable key the client
        # caches the topology — the static ``indices`` are sent once (then stripped,
        # see _strip_known_mesh) and only ``vertices`` stream per frame. ``dynamic``
        # tells the frontend to update positions in place instead of rebuilding.
        self.logged_primitives[f"log-mesh:{name}"] = {
            "id": primitive_id,
            "type": "mesh",
            "material_kind": "cloth",
            "dynamic": True,
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "geo_scale": [1.0, 1.0, 1.0],
            "color": list(color or (0.55, 0.68, 0.88)),
            "mesh_version": self.log_version,
            "mesh_key": primitive_id,
            # Vertices stream every frame -> base64 of raw float32 bytes (~168KB for
            # 14k verts) instead of a JSON float list (~880KB: float32->float64 repr
            # bloats each number to ~18 chars). Indices are static (sent once, then
            # stripped by _strip_known_mesh).
            "mesh": {
                "vertices_b64": base64.b64encode(
                    np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1).tobytes()
                ).decode("ascii"),
                "vcount": int(vertices.reshape(-1, 3).shape[0]),
                "indices": mesh_indices.astype(int).reshape(-1).tolist(),
            },
        }

    def log_points(
        self,
        name: str,
        points: wp.array[wp.vec3] | None,
        radii: wp.array[wp.float32] | float | None = None,
        colors: (
            wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None
        ) = None,
        hidden: bool = False,
    ):
        """Capture viewer point batches as Three.js ``Points`` primitives."""

        if hidden or not self.capture_requested:
            return
        positions = _array(points)
        if positions is None:
            return
        radius = 0.04
        radii_array = _array(radii)
        if isinstance(radii, int | float):
            radius = float(radii)
        elif radii_array is not None and radii_array.size:
            radius = float(np.mean(radii_array))

        color = [0.8, 0.72, 0.52]
        colors_array = _array(colors)
        if colors_array is not None and colors_array.size >= 3:
            color = colors_array.reshape(-1, 3)[0].astype(float).tolist()
        elif isinstance(colors, tuple | list) and len(colors) >= 3:
            color = [float(colors[0]), float(colors[1]), float(colors[2])]

        self.logged_primitives[f"log-points:{name}"] = {
            "id": f"log-points:{name}",
            "type": "points",
            "positions": positions.astype(float).reshape(-1, 3).round(6).tolist(),
            "radius": radius,
            "color": color,
            "mesh_version": self.log_version,
        }

    def log_lines(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """Capture viewer line batches as Three.js ``LineSegments`` primitives."""

        if hidden or not self.capture_requested:
            return
        start_array = _array(starts)
        end_array = _array(ends)
        if start_array is None or end_array is None:
            return
        pairs = np.stack([start_array.reshape(-1, 3), end_array.reshape(-1, 3)], axis=1).reshape(-1, 3)
        color = [0.78, 0.86, 1.0]
        colors_array = _array(colors)
        if colors_array is not None and colors_array.size >= 3:
            color = colors_array.reshape(-1, 3)[0].astype(float).tolist()
        elif isinstance(colors, tuple | list) and len(colors) >= 3:
            color = [float(colors[0]), float(colors[1]), float(colors[2])]
        self.logged_primitives[f"log-lines:{name}"] = {
            "id": f"log-lines:{name}",
            "type": "lines",
            "positions": pairs.astype(float).round(6).tolist(),
            "color": color,
            "mesh_version": self.log_version,
        }

    def export_primitives(self, state: newton.State | None, known_meshes: set[str] | None = None) -> list[dict[str, Any]]:
        """Export model shapes and transient viewer logs for the current frame."""

        known_meshes = known_meshes or set()
        primitives: list[dict[str, Any]] = []
        if self.model is not None and state is not None:
            primitives.extend(self._export_model_primitives(state, known_meshes))
        for primitive in self.logged_primitives.values():
            primitives.append(self._strip_known_mesh(primitive.copy(), known_meshes))
        return primitives

    def _strip_known_mesh(self, primitive: dict[str, Any], known_meshes: set[str]) -> dict[str, Any]:
        """Drop mesh payloads the browser already has.

        Static meshes (robot links): the whole mesh is cached → drop it entirely.
        Dynamic meshes (deforming cloth): only the topology (``indices``) is static
        → drop indices but keep streaming ``vertices`` each frame.
        """

        mesh_key = primitive.get("mesh_key")
        if mesh_key and mesh_key in known_meshes and primitive.get("mesh"):
            if primitive.get("dynamic"):
                m = primitive["mesh"]
                primitive["mesh"] = {"vertices_b64": m["vertices_b64"], "vcount": m.get("vcount")}
            else:
                primitive.pop("mesh", None)
        return primitive

    def _export_model_primitives(self, state: newton.State, known_meshes: set[str]) -> list[dict[str, Any]]:
        """Export Newton model collision/render shapes using viewer batch data."""

        if self.model is None:
            return []

        self._sync_shape_colors_from_model()
        primitives: list[dict[str, Any]] = []
        shape_scales = self.model.shape_scale.numpy()
        shape_types = self.model.shape_type.numpy()

        # ViewerBase groups compatible shapes into batches. Updating each batch
        # gives us world-space transforms in the same order as viewer_gl would
        # render them.
        for batch in self._shape_instances.values():
            if not self._should_show_shape(batch.flags, batch.static):
                continue

            # Newton's ShapeInstances.update gained a required ``layer_xform``
            # arg; mirror the engine's own call (per-layer render transform).
            layer_xform = getattr(getattr(self, "layer", None), "xform", None)
            if layer_xform is None:
                layer_xform = wp.transform_identity()
            batch.update(state, world_offsets=self.world_offsets, layer_xform=layer_xform)
            xforms = batch.world_xforms.numpy()
            scales = batch.scales.numpy()
            colors = batch.colors.numpy()
            shape_indices = batch.model_shapes.numpy() if hasattr(batch.model_shapes, "numpy") else batch.model_shapes

            for i, shape_index in enumerate(shape_indices):
                shape_index = int(shape_index)
                geo_type = newton.GeoType(int(shape_types[shape_index]))
                position, rotation = _transform(xforms[i])
                scale = _vec(scales[i], 3)
                color = _vec(colors[i], 3)
                source = self.model.shape_source[shape_index]

                primitive: dict[str, Any] = {
                    "id": f"shape-{shape_index}",
                    "type": geo_type.name.lower(),
                    "position": position,
                    "rotation": rotation,
                    "scale": scale,
                    "geo_scale": _vec(shape_scales[shape_index], 3),
                    "color": color,
                }

                # Primitive meshes need their raw topology. Analytic shapes
                # only need their Newton geo_scale and are rebuilt in Three.js.
                if geo_type in {newton.GeoType.MESH, newton.GeoType.CONVEX_MESH}:
                    mesh_key = f"model:{id(source)}"
                    if mesh_key not in self.model_mesh_payloads:
                        self.model_mesh_payloads[mesh_key] = _mesh_payload(source)
                    primitive["mesh_key"] = mesh_key
                    primitive["mesh"] = self.model_mesh_payloads[mesh_key]
                    primitive["material_kind"] = "rigidbody"
                elif geo_type == newton.GeoType.HFIELD:
                    mesh_key = f"model:{id(source)}"
                    if mesh_key not in self.model_mesh_payloads:
                        self.model_mesh_payloads[mesh_key] = _heightfield_mesh_payload(source)
                    primitive["type"] = "mesh"
                    primitive["mesh_key"] = mesh_key
                    primitive["mesh"] = self.model_mesh_payloads[mesh_key]
                    primitive["material_kind"] = "floor"

                primitives.append(self._strip_known_mesh(primitive, known_meshes))

        return primitives

