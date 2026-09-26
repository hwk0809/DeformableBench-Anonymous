"""RenderBridge: turn a live sim state into per-camera RGB.

Pluggable so the closed loop can run today without Omniverse and later swap to
Kit without touching the collector:

- ``NewtonOffscreenBridge`` (interim, default): a headless newton ``ViewerGL``
  renders the same ``newton.Model``/``State`` the env steps, reading pixels back
  off-screen (no window). Good enough to wire data collection end-to-end.
- ``ShmKitBridge`` (canonical, TODO R4): push ``particle_q``/``body_q``/gripper
  over shared memory to a Kit app and read RGB back. This is the final visual sink.

Contract (what the collector relies on):
    bridge.camera_names -> list[str]
    bridge.reset(env)
    bridge.render(env) -> {cam_name: np.ndarray (H, W, 3) uint8}
    bridge.close()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Newton off-screen debug camera(s): ``eye`` looks at the workspace centre where
# the cloth lands (~ (0.4, 0, 0.05)). Edit these numbers to reframe the saved
# videos — the off-screen path is non-interactive, so this constant is the only
# control (open a window with ``run_task --viewer gl`` to hunt for an angle).
DEFAULT_CAMERAS: dict[str, dict] = {
    "front": dict(eye=(1.30, 0.0, 0.75), look_at=(0.40, 0.0, 0.05), vfov=50.0),
}


class RenderBridge(ABC):
    """Maps the current sim state to one RGB image per camera."""

    camera_names: list[str]
    image_width: int
    image_height: int

    @abstractmethod
    def reset(self, env) -> None:
        """Bind to a (possibly new) env/model before an episode."""

    @abstractmethod
    def render(self, env) -> dict[str, np.ndarray]:
        """Return ``{cam: (H, W, 3) uint8}`` for the env's current state."""

    def start_episode(self, episode_id: int, output_dir: str | None = None,
                      lighting: dict | None = None, texture: dict | None = None) -> None:
        """Mark an episode boundary; apply per-episode visual DR (Kit only).
        No-op for backends without a scriptable scene (e.g. newton offscreen)."""

    def close(self) -> None:  # optional
        pass


class NewtonOffscreenBridge(RenderBridge):
    """Headless newton ViewerGL → off-screen RGB. No window, no Omniverse.

    Owns its own ViewerGL (separate from any viewer the env holds); it only reads
    ``env.model``/``env.state_0`` to log geometry, so the env must run with
    ``viewer=None`` during collection to avoid two GL contexts fighting.
    """

    def __init__(self, cameras: dict | None = None, width: int = 1280, height: int = 720):
        import warp as wp
        import newton.viewer as V

        self.cameras = cameras or DEFAULT_CAMERAS
        self.camera_names = list(self.cameras)
        self.image_width = int(width)
        self.image_height = int(height)
        self._wp = wp
        self.viewer = V.ViewerGL(width=self.image_width, height=self.image_height,
                                 headless=True, vsync=False)
        self.viewer.show_triangles = True
        self._bound = False

    def reset(self, env) -> None:
        self.viewer.set_model(env.model)
        self._bound = True

    def render(self, env) -> dict[str, np.ndarray]:
        if not self._bound:
            self.reset(env)
        out: dict[str, np.ndarray] = {}
        cam = self.viewer.camera
        for name, c in self.cameras.items():
            cam.fov = float(c.get("vfov", 45.0))
            cam.pos = cam._as_vec3(c["eye"])
            cam.look_at(c["look_at"])  # point at target; sets orientation + pivot
            self.viewer.begin_frame(getattr(env, "sim_time", 0.0))
            self.viewer.log_state(env.state_0)
            self.viewer.end_frame()
            out[name] = self.viewer.get_frame().numpy().copy()  # (H, W, 3) uint8
        return out

    def close(self) -> None:
        if hasattr(self.viewer, "close"):
            self.viewer.close()
