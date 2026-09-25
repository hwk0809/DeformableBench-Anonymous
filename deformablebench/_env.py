"""Process-launch environment fixups.

The dev machine sources ROS Noetic in the shell profile, which puts
``/opt/ros/noetic/lib`` (boost 1.71 built for Python 3.8) on ``LD_LIBRARY_PATH``.
pinocchio (cmeel wheels) bundles its own boost 1.90 for Python 3.12, but the
loader consults ``LD_LIBRARY_PATH`` first and pulls the ROS boost, which fails
with ``undefined symbol: _Py_tracemalloc_config`` (a py3.8-only symbol).

``LD_LIBRARY_PATH`` is read by the dynamic loader at process start, so editing
``os.environ`` after launch has no effect. The only reliable fix is to scrub the
offending entries and re-exec once. The pure-sim stack (warp/newton) does NOT
need ROS on the path, so dropping it is safe.
"""

from __future__ import annotations

import os
import sys

_SENTINEL = "DEFORMABLEBENCH_LD_PATH_CLEANED"
# Entries containing any of these substrings shadow cmeel's bundled boost.
_BAD_MARKERS = ("/ros/", "/ros_ws/", "ros/noetic")


def ensure_clean_ld_path() -> None:
    """Re-exec the current process once with ROS dirs stripped from LD_LIBRARY_PATH.

    Call at the very top of any entry point that imports pinocchio (IK/planning).
    Idempotent: a sentinel env var prevents an exec loop. No-op when the path is
    already clean or empty.
    """

    if os.environ.get(_SENTINEL):
        return
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    if not raw:
        return
    kept = [p for p in raw.split(os.pathsep) if p and not any(m in p for m in _BAD_MARKERS)]
    cleaned = os.pathsep.join(kept)
    if cleaned == raw:
        return  # nothing offending to strip

    os.environ[_SENTINEL] = "1"
    if cleaned:
        os.environ["LD_LIBRARY_PATH"] = cleaned
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    # sys.orig_argv preserves the exact launch form (-m / -c / script.py) so the
    # re-exec reproduces the original invocation faithfully (py3.10+).
    os.execv(sys.executable, sys.orig_argv)


def apply_contact_backend_env(argv: "list[str] | None" = None) -> None:
    """Set the newton-fork tri-SDF backend switches from the task's contact profile.

    The fork bakes ``R16_SDF_EXACT`` / ``T15_SDF_GRIDEXACT`` / ``R13F_SDF_FRICTION``
    into ``wp.constant`` at ``import newton`` time, so they must be in the
    environment BEFORE any sim import. Call right after ``ensure_clean_ld_path()``
    at the top of an entry point. It peeks at ``--task <name>`` (or ``--task=<name>``),
    loads ``config/task/<name>.yaml`` and its ``contact_profile`` fragment with plain
    yaml (no sim import), and reads ``sdf_backend``:

        gridexact -> R13F_SDF_FRICTION=1 R16_SDF_EXACT=1 T15_SDF_GRIDEXACT=1 (gx 0.25 mm)
        exact     -> R13F_SDF_FRICTION=1 R16_SDF_EXACT=1
        voxel     -> R13F_SDF_FRICTION=1 R16_SDF_EXACT=0

    Values already present in the environment win (``setdefault``). No key, no
    ``--task``, or any parse problem => nothing is touched: the default path is
    unchanged bit for bit (docs/GRIPPER-CONTACT.md §6 judgement 7 / T15).
    """
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    args = list(_sys.argv[1:] if argv is None else argv)
    task = None
    for i, a in enumerate(args):
        if a == "--task" and i + 1 < len(args):
            task = args[i + 1]
            break
        if a.startswith("--task="):
            task = a.split("=", 1)[1]
            break
    if not task:
        return
    try:
        import yaml as _yaml
        cfg = _Path(__file__).resolve().parent / "bench" / "config"
        tpath = cfg / "task" / f"{task}.yaml"
        if not tpath.exists():
            return
        with open(tpath) as f:
            d = _yaml.safe_load(f) or {}
        # `base:` inheritance (spec.py::from_yaml) -- a switch variant may put the
        # keys in either file; the child wins, same as the real merge. Bounded so a
        # cycle in a hand-edited yaml cannot hang an entry point.
        for _ in range(8):
            parent = d.pop("base", None)
            if not parent:
                break
            bpath = cfg / "task" / f"{parent}.yaml"
            if not bpath.exists():
                break
            with open(bpath) as f:
                b = _yaml.safe_load(f) or {}
            b.update(d)   # child wins; `base` was popped, so b keeps its own
            d = b
        backend = d.get("sdf_backend")
        harden_kcap = d.get("harden_kcap")
        anchor_fix = d.get("anchor_fix")
        tri_projection = d.get("tri_projection")
        edge_seeds = d.get("edge_seeds")
        profile = d.get("contact_profile")
        if profile:
            ppath = cfg / "contact" / f"{profile}.yaml"
            if ppath.exists():
                with open(ppath) as f:
                    p = _yaml.safe_load(f) or {}
                backend = p.get("sdf_backend", backend)  # preset wins, same as spec.py
                harden_kcap = p.get("harden_kcap", harden_kcap)
                anchor_fix = p.get("anchor_fix", anchor_fix)
                tri_projection = p.get("tri_projection", tri_projection)
                edge_seeds = p.get("edge_seeds", edge_seeds)
    except Exception:  # noqa: BLE001  never let a peek break a run
        return
    # T41/T42 (F2 promotion, T63): triangle-level SDF position projection, two-stage
    # -- only triangles that went through the blade by more than ``tri_projection`` mm
    # are pulled back; the force kernel and the reaction read the pre-projection
    # snapshot.  T42_TRI_DEEP_ONLY_MM is an import-time wp.constant, so it takes the
    # same route as anchor_fix.  ``true`` means the validated 0.5 mm (T42 arm A / T57).
    # No key => nothing is touched.
    if tri_projection:
        _mm = "0.5" if tri_projection is True else str(float(tri_projection))
        _os.environ.setdefault("T41_TRI_SDF_PROJECTION", "1")
        _os.environ.setdefault("T42_TRI_DEEP_ONLY_MM", _mm)
        print(f"[env] tri_projection={tri_projection}: "
              f"T41_TRI_SDF_PROJECTION={_os.environ['T41_TRI_SDF_PROJECTION']} "
              f"T42_TRI_DEEP_ONLY_MM={_os.environ['T42_TRI_DEEP_ONLY_MM']}", flush=True)
    # T52 F2 + T59 (F2 promotion, T63): rigid-mesh-edge-through-triangle seeds for the
    # tri-SDF closest-point search (``edge_seeds`` = seed category mask, ``true`` = the
    # validated 14), always together with the zero-weight seed-cache fix
    # (T59_SEED_CVALID_FIX, docs/GRIPPER-CONTACT.md §1 ㉛ -- F2 without it puts false
    # loads on the blade) and the bit-identical split seed pass (T59_SEED_PASS).
    # All three are import-time wp.constants.  No key => nothing is touched.
    if edge_seeds:
        _mode = "14" if edge_seeds is True else str(int(edge_seeds))
        _os.environ.setdefault("T52_TRI_SDF_EDGE_SEEDS", _mode)
        _os.environ.setdefault("T59_SEED_CVALID_FIX", "1")
        _os.environ.setdefault("T59_SEED_PASS", "1")
        print(f"[env] edge_seeds={edge_seeds}: "
              f"T52_TRI_SDF_EDGE_SEEDS={_os.environ['T52_TRI_SDF_EDGE_SEEDS']} "
              f"T59_SEED_CVALID_FIX={_os.environ['T59_SEED_CVALID_FIX']} "
              f"T59_SEED_PASS={_os.environ['T59_SEED_PASS']}", flush=True)
    # T17: the hardening stiffness cap is another import-time wp.constant in the
    # fork, so it takes the same route.  Independent of sdf_backend on purpose --
    # the two switches are unrelated.  No key => nothing is touched.
    if harden_kcap:
        _mode = "1" if harden_kcap is True else str(int(harden_kcap))
        _os.environ.setdefault("T17_SDF_HARDEN_KCAP", _mode)
        print(f"[env] harden_kcap={harden_kcap}: "
              f"T17_SDF_HARDEN_KCAP={_os.environ['T17_SDF_HARDEN_KCAP']}", flush=True)
    # T18: the contact-Hessian scatter-weight fix is another import-time
    # wp.constant in the fork, so it takes the same route as harden_kcap.
    # Bitmask: 1 = tangential row-sum lump + seeding + C-1, 2 = motion-limited
    # plastic drag, 4 = normal row-sum lump.  ``true`` means the recommended 5
    # (tangential + normal).  Bits 1 and 4 are the two halves of ONE scatter
    # inconsistency; fixing only the tangential half leaves the cloth pinned
    # tangentially while the normal diagonal is still 3x too soft, and the one
    # observed drape collapse in the T18b batch (24 flatten runs) landed
    # exactly in that mixed state -- vertex 0.315 mm inside the blade, 1.75x
    # the +-0.18 mm vertex noise floor, vs 0.022 mm worst for mask 5.  See
    # data/eval_out/gripper_penetration/T18/T18_RESULTS.md section 6.4.
    # No key => nothing is touched.
    if anchor_fix:
        _mode = "5" if anchor_fix is True else str(int(anchor_fix))
        _os.environ.setdefault("T18_SDF_ANCHOR_FIX", _mode)
        print(f"[env] anchor_fix={anchor_fix}: "
              f"T18_SDF_ANCHOR_FIX={_os.environ['T18_SDF_ANCHOR_FIX']}", flush=True)
    if not backend:
        return
    table = {
        "gridexact": {"R13F_SDF_FRICTION": "1", "R16_SDF_EXACT": "1",
                      "T15_SDF_GRIDEXACT": "1", "T15_SDF_GX_VOXEL": "0.00025"},
        "exact": {"R13F_SDF_FRICTION": "1", "R16_SDF_EXACT": "1"},
        "voxel": {"R13F_SDF_FRICTION": "1", "R16_SDF_EXACT": "0"},
    }
    env = table.get(str(backend))
    if env is None:
        print(f"[env] unknown sdf_backend={backend!r} (ignored)", flush=True)
        return
    for k, v in env.items():
        _os.environ.setdefault(k, v)
    print(f"[env] sdf_backend={backend}: " + " ".join(f"{k}={_os.environ[k]}" for k in env), flush=True)
