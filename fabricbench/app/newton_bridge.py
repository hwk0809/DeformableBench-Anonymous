"""Backend bridge between the simulation and the Reflex/Three.js UI.

The browser only renders: it receives small JSON geometry snapshots and never
imports Newton or runs simulation code.

Rule of thumb — whether the sim runs here or in its own process is decided by one
thing: does it need the IK planner (pinocchio)? pinocchio can't load in this Reflex
backend (LD_LIBRARY_PATH), so anything needing it must run elsewhere.

- **Tasks** (``task:*``, robot+cloth, need pinocchio): the sim runs headless in a
  separate process (``run_task --publish-shm``) and streams a per-frame geometry
  snapshot to shared memory. This backend builds the matching scene ``Model`` once
  and only injects those positions before rendering — it never steps the task
  physics ("read-only, low-frequency view", per ``sim-architecture-proposal.md`` §7.2).
- **Bag drop** (``deformable``, cloth-only, no pinocchio) and installed
  **``newton.examples.*``**: cheap, stepped in-process here.
"""

from __future__ import annotations

import importlib
import itertools
import os
import subprocess
import sys
import threading
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse

import newton
import newton.examples
from fabricbench.app.viewer_web import ViewerWeb
from fabricbench.bench.render.geom_shm import GeomSnapshotShm

_shm_counter = itertools.count()


# Synthetic id for the benchmark's own scene (DeformableEnv on the open
# SolverStyle3D), as opposed to the installed ``newton.examples.*`` modules.
DEFORMABLE_ID = "deformable"
DEFAULT_EXAMPLE_ID = DEFORMABLE_ID

# Benchmark task scenes (robot + cloth) are selectable too, under a "task:"
# prefix. They are built from ``config/task/<name>.yaml`` exactly like run_task
# (TaskSpec -> DeformableEnv), with the robot raised to its start pose. The web
# loop steps with no action, so the robot holds its init pose while the cloth
# settles/drapes live — a static-but-correct preview of the task scene.
TASK_PREFIX = "task:"


# Newton examples are usually launched from their own argparse parser. The web
# UI needs a single, non-interactive argument object that is broad enough for the
# examples in the installed package. Individual examples read only the fields
# they need.
ARG_DEFAULTS: dict[str, Any] = {
    "belt_speed": 1.0,
    "broad_phase": "sap",
    "collider": "sphere",
    "collider_basis": "sphere",
    "damping": 0.0,
    "density": 1000.0,
    "device": None,
    "dilatancy": 0.0,
    "drone_path": None,
    "emit_hi": (0.5, 0.5, 1.0),
    "emit_lo": (-0.5, -0.5, 0.0),
    "eps_max": 0.0,
    "fps": 60,
    "friction": 0.5,
    "friction_coeff": 0.5,
    "from_usd": False,
    "funnel_aperture": 0.5,
    "funnel_friction": 0.5,
    "funnel_height": 2.0,
    "funnel_offset_z": 0.0,
    "funnel_top_radius": 1.0,
    "gravity": -9.81,
    "grid_type": "dense",
    "hardening": 10.0,
    "headless": True,
    "height": 32,
    "material_behavior": "corotated",
    "max_iterations": 100,
    "max_worlds": 64,
    "min_response": 0.0,
    "no_dahl": False,
    "num_per_world": 1,
    "num_pyramids": 1,
    "num_rollouts": 1,
    "particles_per_cell": 8,
    "physx": False,
    "ply": None,
    "poisson_ratio": 0.3,
    "points_per_particle": 1,
    "pyramid_size": 4,
    "render_rollouts": False,
    "robot": "cartpole",
    "scene": "box",
    "segments": 16,
    "sim_steps": 100,
    "solver": "xpbd",
    "strain_basis": "p2g",
    "tau": 0.0,
    "tensile_yield_ratio": 0.0,
    "test": False,
    "tolerance": 1.0e-5,
    "use_kamino_contacts": False,
    "use_mujoco_contacts": False,
    "velocity_basis": "apic",
    "verbose": False,
    "viscosity": 0.1,
    "voxel_size": 0.1,
    "width": 32,
    "world_count": 1,
    "yield_pressure": 1.0e5,
    "yield_stress": 1.0e5,
    "young_modulus": 1.0e5,
}


def _module_from_example_id(example_id: str) -> str:
    """Normalize a UI example id into an importable Python module path."""

    if example_id.startswith("newton.examples."):
        return example_id
    return f"newton.examples.{example_id}"


def _example_id_from_module(module_name: str) -> str:
    """Convert an importable module path back into the compact UI id."""

    prefix = "newton.examples."
    return module_name[len(prefix) :] if module_name.startswith(prefix) else module_name


def list_benchmark_tasks() -> list[dict[str, str]]:
    """Return the selectable benchmark task scenes (``config/task/*.yaml``)."""

    task_dir = Path(__file__).resolve().parent.parent / "bench" / "config" / "task"
    tasks: list[dict[str, str]] = []
    for path in sorted(task_dir.glob("*.yaml")):
        name = path.stem
        tasks.append({"id": f"{TASK_PREFIX}{name}", "label": f"task / {name}"})
    return tasks


def list_newton_examples() -> list[dict[str, str]]:
    """Return all installed Newton example modules available to the UI."""

    examples: list[dict[str, str]] = []
    for root in newton.examples.__path__:
        root_path = Path(root)
        for path in root_path.rglob("example*.py"):
            if path.name == "__init__.py":
                continue
            short_id = ".".join(path.relative_to(root_path).with_suffix("").parts)
            label = short_id.replace(".example_", " / ").replace("_", " ")
            examples.append({"id": short_id, "label": label})
    examples = sorted(examples, key=lambda item: item["id"])
    # Surface the benchmark scenes first so the UI defaults to them: the bag
    # drop, then the robot+cloth task scenes, then the installed newton examples.
    return [
        {"id": DEFORMABLE_ID, "label": "deformable / bag drop (Style3D)"},
        *list_benchmark_tasks(),
        *examples,
    ]


class NewtonExampleSimulation:
    """Owns one selected Newton example and exposes frame snapshots."""

    def __init__(self, example_id: str = DEFAULT_EXAMPLE_ID):
        self.example_id = example_id
        self.viewer = ViewerWeb()
        self.args = Namespace(**ARG_DEFAULTS)
        self.running = True
        self.frame_index = 0
        self.error: str | None = None
        self.example: Any | None = None
        # shm-consumer mode (tasks): sim runs in a run_task --publish-shm process;
        # we read its geometry snapshots and inject them, never stepping here.
        self._shm_mode = False
        self._geom_shm: GeomSnapshotShm | None = None
        self._sim_proc: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._step_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._load_example(example_id)
        self._start_loop()

    def _load_example(self, example_id: str):
        """Import and instantiate the selected scene.

        The benchmark scene (``DeformableEnv``) is handled specially; everything
        else is loaded as an installed ``newton.examples.*`` module.
        """

        if example_id == DEFORMABLE_ID:
            # Bag drop: cloth-only, no robot, no pinocchio — cheap to step here.
            from fabricbench.sim import DeformableEnv

            self.example = DeformableEnv(self.viewer)
            self.example_id = DEFORMABLE_ID
            return

        if example_id.startswith(TASK_PREFIX):
            self._load_task_scene(example_id)
            return

        module_name = _module_from_example_id(example_id)
        module = importlib.import_module(module_name)
        example_class = getattr(module, "Example")
        self.example = example_class(self.viewer, self.args)
        self.example_id = _example_id_from_module(module_name)

    def _load_task_scene(self, scene_id: str):
        """Build the task's Model once, then spawn ``run_task`` as the producer.

        A task needs the IK planner (pinocchio), which can't load in this Reflex
        backend (LD_LIBRARY_PATH), so the physics must run in a separate process.
        We build the matching scene ``Model`` (static meshes + topology), allocate a
        geometry snapshot sized for it, and launch ``run_task --publish-shm`` to run
        the episode and stream positions. We never step the env — ``snapshot()``
        injects the latest shm frame and renders it.
        """

        from fabricbench.bench.task.spec import TaskSpec
        from fabricbench.sim import DeformableEnv

        spec = TaskSpec.from_name(scene_id[len(TASK_PREFIX):])
        # No randomizer: only topology/meshes matter here (positions come from shm),
        # so the canonical scene's vertex/body counts match the producer's.
        self.example = DeformableEnv(viewer=self.viewer, cfg=spec.scene_config(None, None))
        self.example_id = scene_id

        model = self.example.model
        n_verts = int(getattr(model, "particle_count", 0) or 0)
        n_bodies = int(getattr(model, "body_count", 0) or 0)
        shm_name = f"fabricbench_geom_{os.getpid()}_{next(_shm_counter)}"
        self._geom_shm = GeomSnapshotShm.create(shm_name, n_verts, n_bodies)

        # One-shot (no --loop): the episode runs once then holds the final frame.
        # Looping continuously was too heavy on this machine. (--loop still exists
        # on run_task if you ever want continuous motion.)
        cmd = [sys.executable, "-m", "fabricbench.bench.run_task",
               "--task", scene_id[len(TASK_PREFIX):], "--publish-shm", shm_name]
        self._sim_proc = subprocess.Popen(cmd, start_new_session=True)
        self._shm_mode = True

    def _inject_snapshot(self):
        """Pull the latest geometry frame from shm into ``state_0`` for rendering."""

        if self._geom_shm is None:
            return
        snap = self._geom_shm.read()
        if snap is None:
            return
        particle_q, body_q, seq = snap
        state = self.example.state_0
        if getattr(state, "particle_q", None) is not None and particle_q.size:
            state.particle_q.assign(particle_q)
        if getattr(state, "body_q", None) is not None and body_q.size:
            state.body_q.assign(body_q)
        self.frame_index = int(seq)

    def _state(self) -> newton.State | None:
        """Find the active Newton state exposed by common example patterns."""

        if self.example is None:
            return None
        for name in ("state_0", "state", "state_q"):
            state = getattr(self.example, name, None)
            if state is not None:
                return state
        return None

    def step_if_running(self):
        """Advance one simulation step when the example is not paused."""

        with self._step_lock:
            if self.running and self.example is not None and hasattr(self.example, "step"):
                self.example.step()
                self.frame_index += 1

    def render_to_viewer(self):
        """Ask the example to log its current frame into ``ViewerWeb``."""

        with self._step_lock:
            if self.example is not None and hasattr(self.example, "render"):
                self.viewer.request_capture()
                try:
                    self.example.render()
                finally:
                    self.viewer.finish_capture()

    def _start_loop(self):
        """Start the background simulation loop for this example."""

        self._thread = threading.Thread(target=self._run_loop, name=f"newton-web-{self.example_id}", daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Continuously advance simulation independent of HTTP frame polling."""

        while not self._stop_event.is_set():
            # shm-consumer scenes don't step here — the producer process does; we
            # only read its snapshots (in snapshot()). Idle the loop.
            if self._shm_mode:
                time.sleep(0.02)
                continue
            if self.error or not self.running:
                time.sleep(0.005)
                continue
            try:
                started_at = time.perf_counter()
                self.step_if_running()
                frame_dt = float(getattr(self.example, "frame_dt", 0.0) or 0.0)
                if frame_dt > 0.0:
                    time.sleep(max(0.0, frame_dt - (time.perf_counter() - started_at)))
            except Exception:
                self.error = traceback.format_exc(limit=8)
                time.sleep(0.05)

    def close(self):
        """Stop the background loop, kill the producer, and release shm/viewer."""

        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sim_proc is not None:
            try:
                self._sim_proc.terminate()
                self._sim_proc.wait(timeout=5)
            except Exception:
                self._sim_proc.kill()
            self._sim_proc = None
        if self._geom_shm is not None:
            self._geom_shm.close(unlink=True)
            self._geom_shm = None
        if hasattr(self.viewer, "close"):
            self.viewer.close()

    def snapshot(self, known_meshes: set[str] | None = None) -> dict[str, Any]:
        """Step, render, and return a JSON-serializable frame payload."""

        if self.error:
            return self._payload([])
        try:
            with self._step_lock:
                if self._shm_mode:
                    self._inject_snapshot()   # pull latest producer frame
                self.render_to_viewer()
                return self._payload(self.viewer.export_primitives(self._state(), known_meshes))
        except Exception:
            # Store the traceback so future polls return a stable error payload
            # instead of repeatedly raising inside the Starlette handler.
            self.error = traceback.format_exc(limit=8)
            return self._payload([])

    def _payload(self, primitives: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the response shared by frame, switch, and error paths."""

        sim_time = getattr(self.example, "sim_time", 0.0) if self.example is not None else 0.0
        return {
            "example_id": self.example_id,
            "frame": self.frame_index,
            "running": self.running,
            "time": float(sim_time),
            "wall_time": time.perf_counter(),
            "camera": self.viewer.camera_state,
            "error": self.error,
            "primitives": primitives,
        }

    def toggle(self) -> dict[str, Any]:
        """Toggle paused/running state for the current example."""

        self.running = not self.running
        return {"running": self.running}


_simulation: NewtonExampleSimulation | None = None
_lock = threading.RLock()


def get_simulation() -> NewtonExampleSimulation:
    """Return the lazily-created singleton simulation."""

    global _simulation
    with _lock:
        if _simulation is None:
            _simulation = NewtonExampleSimulation()
        return _simulation


def set_simulation_example(example_id: str) -> NewtonExampleSimulation:
    """Replace the singleton simulation with a newly selected example."""

    global _simulation
    with _lock:
        if _simulation is not None:
            _simulation.close()
        _simulation = NewtonExampleSimulation(example_id)
        return _simulation


async def newton_frame(_request):
    """Starlette handler returning the latest simulation frame."""

    known_meshes = {
        item for item in _request.query_params.get("known_meshes", "").split(",") if item
    }
    with _lock:
        return JSONResponse(get_simulation().snapshot(known_meshes))


async def newton_toggle(_request):
    """Starlette handler toggling simulation run/pause."""

    with _lock:
        return JSONResponse(get_simulation().toggle())


async def newton_examples(_request):
    """Starlette handler returning selectable Newton examples."""

    with _lock:
        return JSONResponse({"examples": list_newton_examples(), "current": get_simulation().example_id})


async def newton_select_example(request):
    """Starlette handler switching the singleton simulation example."""

    payload = await request.json()
    example_id = payload.get("id")
    available = {item["id"] for item in list_newton_examples()}
    if example_id not in available:
        return JSONResponse({"error": f"Unknown Newton example: {example_id}"}, status_code=400)

    with _lock:
        try:
            simulation = set_simulation_example(example_id)
            return JSONResponse(simulation.snapshot())
        except Exception:
            # Keep the backend usable if a selected example has unsupported
            # arguments, missing assets, or viewer features we do not bridge yet.
            error = traceback.format_exc(limit=8)
            simulation = set_simulation_example(DEFAULT_EXAMPLE_ID)
            simulation.error = error
            return JSONResponse(simulation.snapshot(), status_code=500)


def add_newton_routes(app):
    """Register Newton API routes on Reflex's underlying Starlette app."""

    app._api.add_route("/newton/frame", newton_frame, methods=["GET"])
    app._api.add_route("/newton/toggle", newton_toggle, methods=["POST"])
    app._api.add_route("/newton/examples", newton_examples, methods=["GET"])
    app._api.add_route("/newton/example", newton_select_example, methods=["POST"])
