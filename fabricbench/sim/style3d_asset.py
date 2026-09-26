"""Local Style3D/Newton cloth asset helpers.

This module owns the small subset of the Newton example asset-probe behavior
that the benchmark needs at runtime. Keeping it local avoids importing private
example helper code while still using Newton's public/fork APIs for the actual
solver and model construction.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
NEWTON_ROOT = REPO_ROOT / "thirdparty" / "newton_style3d"


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    source_format: str
    mesh_prim_path: str | None = None
    face_counts: list[int] = field(default_factory=list)
    panel_vertices: np.ndarray | None = None
    panel_faces: np.ndarray | None = None
    style3d_vertices: np.ndarray | None = None
    style3d_faces: np.ndarray | None = None
    style3d_panel_vertices: np.ndarray | None = None
    panel_coordinate_source: str | None = None


@dataclass
class MeshReport:
    path: str
    status: str
    source_format: str | None = None
    vertex_count: int = 0
    face_count: int = 0
    triangle_count: int = 0
    non_triangle_face_count: int = 0
    component_count: int = 0
    component_sizes: list[int] = field(default_factory=list)
    boundary_edge_count: int = 0
    nonmanifold_edge_count: int = 0
    duplicate_vertex_pairs_1e_6: int = 0
    degenerate_face_count: int = 0
    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    bbox_size: list[float] = field(default_factory=list)
    recommended_scale: float | None = None
    newton_readiness: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    generated_usd: str | None = None
    usd_prim_path: str | None = None
    panel_vertex_count: int = 0
    panel_triangle_count: int = 0
    panel_coordinate_source: str | None = None
    style3d_vertex_count: int = 0
    style3d_triangle_count: int = 0
    newton_build: dict[str, object] | None = None


@dataclass(frozen=True)
class NewtonClothParams:
    density: float = 0.3
    particle_radius: float = 5.0e-3
    tri_ka: float = 100.0
    tri_kd: float = 1.5e-6
    tri_aniso_ke: tuple[float, float, float] = (100.0, 100.0, 10.0)
    edge_aniso_ke: tuple[float, float, float] = (2.0e-5, 1.0e-5, 5.0e-6)
    edge_kd: float = 1.0e-3
    soft_contact_margin: float = 3.5e-3
    soft_contact_ke: float = 10.0
    soft_contact_kd: float = 1.0e-6
    soft_contact_mu: float = 0.2
    solver_iterations: int = 4
    linear_iterations: int = 10
    step_dt: float = 1.0 / 120.0
    style3d_self_collision: bool = True
    style3d_collision_radius: float = 3.0e-3
    style3d_collision_stiff_vf: float = 0.5
    style3d_collision_stiff_ee: float = 0.1
    style3d_collision_stiff_ef: float = 1.0
    # IPC-style friction regularization radius [m] of the style3d body-particle
    # contact (solver default 1e-2). At 1e-2 a settled cloth's per-substep
    # micro-slips (~um) see only ~u/eps of the Coulomb force, so static
    # friction never engages and internal stresses (e.g. self-collision
    # pushing overlapped layers apart) make the cloth "walk" across the ground
    # at ~3 mm/s indefinitely. 1e-5 lets static friction bite (measured walk
    # -> ~0.5 mm/s with self-collision on, 0 with it off).
    style3d_friction_epsilon: float = 1.0e-2
    # FUSE1/FUSE2: fuse the PCG step's element-wise kernels (and the translation
    # preconditioner's two) down to 3-4 launches per step; the two dot products
    # use an 8-bucket two-level reduction.  Pure efficiency, but the reduction
    # order changes, so it is NOT bit-identical on GPU (1 ULP after one
    # substep).  DEFAULT ON (2026-09-14); escape hatch = task yaml
    # `physics: {style3d_pcg_fused: false}` (kwarg omitted -> unfused loop).
    style3d_pcg_fused: bool = True


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    for root in (REPO_ROOT, NEWTON_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def parse_obj(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_texcoords: list[list[int | None]] = []
    face_counts: list[int] = []

    def parse_index(text: str, count: int) -> int:
        idx = int(text)
        return count + idx if idx < 0 else idx - 1

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                raw = line.split()[1:]
                indices: list[int] = []
                tex_indices: list[int | None] = []
                for token in raw:
                    if not token:
                        continue
                    token_parts = token.split("/")
                    idx_text = token_parts[0]
                    if not idx_text:
                        continue
                    indices.append(parse_index(idx_text, len(vertices)))
                    if len(token_parts) >= 2 and token_parts[1]:
                        tex_indices.append(parse_index(token_parts[1], len(texcoords)))
                    else:
                        tex_indices.append(None)
                if len(indices) >= 3:
                    face_counts.append(len(indices))
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])
                        face_texcoords.append([tex_indices[0], tex_indices[i], tex_indices[i + 1]])

    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    panel_vertices: np.ndarray | None = None
    panel_faces: np.ndarray | None = None
    style3d_vertices: np.ndarray | None = None
    style3d_faces: np.ndarray | None = None
    style3d_panel_vertices: np.ndarray | None = None
    panel_source: str | None = None

    has_complete_uv_faces = (
        len(texcoords) > 0
        and len(face_texcoords) == len(faces)
        and all(all(t is not None and 0 <= int(t) < len(texcoords) for t in tri) for tri in face_texcoords)
    )
    if has_complete_uv_faces:
        texcoords_np = np.asarray(texcoords, dtype=np.float64)
        panel_vertices = texcoords_np
        panel_faces = np.asarray(face_texcoords, dtype=np.int64).reshape((-1, 3))
        panel_source = "obj_vt_indexed"
        uv_by_vertex: dict[int, int] = {}
        consistent = True
        for tri, tri_uv in zip(faces_np, face_texcoords, strict=True):
            for vertex_idx, uv_idx in zip(tri, tri_uv, strict=True):
                vi = int(vertex_idx)
                ui = int(uv_idx)
                previous = uv_by_vertex.get(vi)
                if previous is None:
                    uv_by_vertex[vi] = ui
                elif previous != ui:
                    consistent = False
                    break
            if not consistent:
                break

        if consistent and len(uv_by_vertex) == len(vertices_np):
            style3d_panel_vertices = texcoords_np[[uv_by_vertex[i] for i in range(len(vertices_np))]]
            panel_source = "obj_vt"
        else:
            pair_to_index: dict[tuple[int, int], int] = {}
            expanded_vertices: list[np.ndarray] = []
            expanded_panel_vertices: list[np.ndarray] = []
            expanded_faces: list[list[int]] = []
            for tri, tri_uv in zip(faces_np, face_texcoords, strict=True):
                expanded_tri: list[int] = []
                for vertex_idx, uv_idx in zip(tri, tri_uv, strict=True):
                    key = (int(vertex_idx), int(uv_idx))
                    mapped = pair_to_index.get(key)
                    if mapped is None:
                        mapped = len(expanded_vertices)
                        pair_to_index[key] = mapped
                        expanded_vertices.append(vertices_np[key[0]])
                        expanded_panel_vertices.append(texcoords_np[key[1]])
                    expanded_tri.append(mapped)
                expanded_faces.append(expanded_tri)
            style3d_vertices = np.asarray(expanded_vertices, dtype=np.float64)
            style3d_faces = np.asarray(expanded_faces, dtype=np.int64).reshape((-1, 3))
            style3d_panel_vertices = np.asarray(expanded_panel_vertices, dtype=np.float64)
            panel_source = "obj_vt_expanded"

    return MeshData(
        vertices=vertices_np,
        faces=faces_np,
        source_format="obj",
        face_counts=face_counts,
        panel_vertices=panel_vertices,
        panel_faces=panel_faces,
        style3d_vertices=style3d_vertices,
        style3d_faces=style3d_faces,
        style3d_panel_vertices=style3d_panel_vertices,
        panel_coordinate_source=panel_source,
    )


def _find_mesh_prims(stage, Usd, UsdGeom, prim_path: str | None):
    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"USD prim path is invalid: {prim_path}")
        if prim.IsA(UsdGeom.Mesh):
            return [prim]
        return [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    return [p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.IsA(UsdGeom.Mesh)]


def parse_usd(path: Path, prim_path: str | None = None) -> MeshData:
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(f"pxr is required to read USD files: {exc}") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD: {path}")
    mesh_prims = _find_mesh_prims(stage, Usd, UsdGeom, prim_path)
    if len(mesh_prims) != 1:
        paths = [str(p.GetPath()) for p in mesh_prims[:8]]
        raise ValueError(f"Expected exactly one Mesh prim, found {len(mesh_prims)}: {paths}")

    prim = mesh_prims[0]
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if points is None or indices is None or counts is None:
        raise ValueError(f"Mesh prim is missing points/indices/counts: {prim.GetPath()}")

    vertices = np.asarray(points, dtype=np.float64)
    flat_indices = np.asarray(indices, dtype=np.int64)
    counts_np = np.asarray(counts, dtype=np.int64)
    faces: list[list[int]] = []
    cursor = 0
    for count in counts_np:
        poly = flat_indices[cursor: cursor + int(count)].tolist()
        cursor += int(count)
        if len(poly) >= 3:
            for i in range(1, len(poly) - 1):
                faces.append([poly[0], poly[i], poly[i + 1]])
    return MeshData(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        source_format=path.suffix.lower().lstrip("."),
        mesh_prim_path=str(prim.GetPath()),
        face_counts=counts_np.astype(int).tolist(),
    )


def load_mesh(path: Path, prim_path: str | None = None) -> MeshData:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return parse_obj(path)
    if suffix in {".usd", ".usda", ".usdc"}:
        return parse_usd(path, prim_path=prim_path)
    raise ValueError(f"Unsupported mesh format: {suffix}")


def connected_components(vertex_count: int, faces: np.ndarray) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces:
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].extend([bi, ci])
        adj[bi].extend([ai, ci])
        adj[ci].extend([ai, bi])

    seen = np.zeros(vertex_count, dtype=bool)
    sizes: list[int] = []
    for start in range(vertex_count):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def edge_stats(faces: np.ndarray) -> tuple[int, int]:
    counts: Counter[tuple[int, int]] = Counter()
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            counts[(u, v)] += 1
    boundary = sum(1 for value in counts.values() if value == 1)
    nonmanifold = sum(1 for value in counts.values() if value > 2)
    return boundary, nonmanifold


def degenerate_faces(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-14) -> int:
    if len(faces) == 0:
        return 0
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    return int(np.count_nonzero((area2 <= eps) | repeated))


def duplicate_pairs(vertices: np.ndarray, tol: float = 1e-6) -> int:
    if len(vertices) == 0:
        return 0
    quant = np.round(vertices / tol).astype(np.int64)
    counts = Counter(map(tuple, quant.tolist()))
    return int(sum(n * (n - 1) // 2 for n in counts.values() if n > 1))


def estimate_recommended_scale(bbox_size: np.ndarray) -> float | None:
    max_dim = float(np.max(bbox_size)) if bbox_size.size else 0.0
    if max_dim <= 0.0:
        return None
    return 0.001 if max_dim > 10.0 else 1.0


def analyze_mesh(path: Path, mesh: MeshData) -> MeshReport:
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            display_path = str(path.relative_to(NEWTON_ROOT))
        except ValueError:
            display_path = str(path)
    report = MeshReport(path=display_path, status="ok")
    report.source_format = mesh.source_format
    report.vertex_count = int(mesh.vertices.shape[0])
    report.face_count = len(mesh.face_counts)
    report.triangle_count = int(mesh.faces.shape[0])
    report.non_triangle_face_count = int(sum(1 for c in mesh.face_counts if c != 3))
    report.usd_prim_path = mesh.mesh_prim_path
    report.panel_vertex_count = int(mesh.panel_vertices.shape[0]) if mesh.panel_vertices is not None else 0
    report.panel_triangle_count = int(mesh.panel_faces.shape[0]) if mesh.panel_faces is not None else 0
    report.panel_coordinate_source = mesh.panel_coordinate_source
    report.style3d_vertex_count = int(mesh.style3d_vertices.shape[0]) if mesh.style3d_vertices is not None else 0
    report.style3d_triangle_count = int(mesh.style3d_faces.shape[0]) if mesh.style3d_faces is not None else 0

    if report.vertex_count == 0 or report.triangle_count == 0:
        report.status = "failed"
        report.newton_readiness = "blocked"
        report.warnings.append("No usable vertices or triangles were found.")
        return report

    invalid_index = bool(np.any(mesh.faces < 0) or np.any(mesh.faces >= report.vertex_count))
    if invalid_index:
        report.status = "failed"
        report.newton_readiness = "blocked"
        report.warnings.append("Face indices reference vertices outside the vertex array.")
        return report

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    report.bbox_min = [float(x) for x in bbox_min]
    report.bbox_max = [float(x) for x in bbox_max]
    report.bbox_size = [float(x) for x in bbox_size]
    report.recommended_scale = estimate_recommended_scale(bbox_size)
    report.component_sizes = connected_components(report.vertex_count, mesh.faces)
    report.component_count = len(report.component_sizes)
    report.boundary_edge_count, report.nonmanifold_edge_count = edge_stats(mesh.faces)
    report.degenerate_face_count = degenerate_faces(mesh.vertices, mesh.faces)
    report.duplicate_vertex_pairs_1e_6 = duplicate_pairs(mesh.vertices, tol=1e-6)

    if report.non_triangle_face_count:
        report.warnings.append("Newton cloth import expects triangle meshes; this asset needs triangulation.")
    if report.component_count > 1:
        report.warnings.append("Mesh has multiple connected components; bags may need seam/stitch constraints between components.")
    if report.boundary_edge_count:
        report.warnings.append("Mesh has boundary edges; this is normal for panels, but not a closed two-manifold cloth.")
    if report.nonmanifold_edge_count:
        report.warnings.append("Mesh has non-manifold edges; Newton add_cloth_mesh notes that the mesh should be two-manifold.")
    if report.degenerate_face_count:
        report.warnings.append("Mesh has degenerate triangles; clean or weld before Newton import.")
    if report.duplicate_vertex_pairs_1e_6:
        report.warnings.append("Mesh has near-duplicate vertices; this often indicates Style3D seam points.")
    if mesh.panel_coordinate_source == "obj_vt_expanded":
        report.warnings.append("OBJ uses face-varying UVs; panel coordinates are imported with separate panel indices.")
    if report.recommended_scale == 0.001:
        report.warnings.append("Asset dimensions look like millimeters; import with scale=0.001.")

    severe = report.nonmanifold_edge_count > 0 or report.degenerate_face_count > 0
    if severe:
        report.newton_readiness = "needs_cleanup"
    elif report.component_count > 1 or report.duplicate_vertex_pairs_1e_6 > 0:
        report.newton_readiness = "needs_stitch_or_weld"
    elif report.non_triangle_face_count > 0:
        report.newton_readiness = "needs_triangulation"
    else:
        report.newton_readiness = "direct_candidate"
    return report


def import_newton(warp_cache_dir: str | None = None, device: str | None = None):
    import warp as wp

    if warp_cache_dir:
        Path(warp_cache_dir).mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(Path(warp_cache_dir).resolve())
    if device:
        wp.set_device(device)
    import newton

    return newton, wp


def apply_contact_params(model, cloth_params: NewtonClothParams) -> None:
    for name, value in (
        ("soft_contact_ke", cloth_params.soft_contact_ke),
        ("soft_contact_kd", cloth_params.soft_contact_kd),
        ("soft_contact_mu", cloth_params.soft_contact_mu),
    ):
        if hasattr(model, name):
            setattr(model, name, float(value))


def configure_style3d_solver_collision(solver, state, cloth_params: NewtonClothParams) -> None:
    collision = getattr(solver, "collision", None)
    if collision is None:
        return
    collision.radius = float(cloth_params.style3d_collision_radius)
    collision.friction_epsilon = float(cloth_params.style3d_friction_epsilon)
    if cloth_params.style3d_self_collision:
        collision.stiff_vf = float(cloth_params.style3d_collision_stiff_vf)
        collision.stiff_ee = float(cloth_params.style3d_collision_stiff_ee)
        collision.stiff_ef = float(cloth_params.style3d_collision_stiff_ef)
    else:
        collision.stiff_vf = 0.0
        collision.stiff_ee = 0.0
        collision.stiff_ef = 0.0
    if hasattr(solver, "rebuild_bvh"):
        solver.rebuild_bvh(state)


def repair_panel_winding(
    faces: np.ndarray,
    panel_faces: np.ndarray,
    panel_vertices: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled or len(faces) == 0:
        return faces, panel_faces
    tri = panel_vertices[panel_faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    neg = area2 < -1.0e-14
    zero = np.abs(area2) <= 1.0e-14
    if not np.any(neg):
        if np.any(zero):
            print(f"[style3d_asset] Warning: {int(np.count_nonzero(zero))} zero-area panel triangles remain.", flush=True)
        return faces, panel_faces

    repaired_faces = faces.copy()
    repaired_panel_faces = panel_faces.copy()
    repaired_faces[neg] = repaired_faces[neg][:, [0, 2, 1]]
    repaired_panel_faces[neg] = repaired_panel_faces[neg][:, [0, 2, 1]]
    tri_repaired = panel_vertices[repaired_panel_faces]
    r1 = tri_repaired[:, 1] - tri_repaired[:, 0]
    r2 = tri_repaired[:, 2] - tri_repaired[:, 0]
    area2_repaired = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
    valid = area2_repaired > 1.0e-14
    print(
        f"[style3d_asset] Repaired Style3D panel winding: flipped={int(np.count_nonzero(neg))}, "
        f"valid_after={int(np.count_nonzero(valid))}/{len(faces)}, "
        f"zero_after={int(np.count_nonzero(np.abs(area2_repaired) <= 1.0e-14))}",
        flush=True,
    )
    return repaired_faces, repaired_panel_faces


def project_panel_vertices(wp, vertices: np.ndarray, axes: str):
    axes = axes.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if len(axes) != 2 or axes[0] not in axis_map or axes[1] not in axis_map or axes[0] == axes[1]:
        raise ValueError(f"Unsupported style3d panel axes: {axes!r}; expected one of xy, xz, yz")
    a0, a1 = axis_map[axes[0]], axis_map[axes[1]]
    return [wp.vec2(float(v[a0]), float(v[a1])) for v in vertices]
