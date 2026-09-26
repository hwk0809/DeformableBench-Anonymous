# FabricBench — source code (anonymized for review)

Source-only snapshot accompanying the submission. Assets, demonstrations and
environment setup are omitted here and will be released with the camera-ready version.

- `fabricbench/sim/`   — physics layer: Newton `SolverStyle3D` (cloth) + `SolverMuJoCo` (robots) co-simulation.
- `fabricbench/bench/` — tasks, skills/planners, data collection, policy evaluation (`config/` holds task and contact configs).
- `fabricbench/app/`   — debugging web UI.
- `newton_patch/`   — our modifications to [Newton](https://github.com/newton-physics/newton)
  (Apache-2.0), relative to upstream commit `d4a98997`: the cloth–gripper contact model
  (`newton/_src/solvers/style3d/`), plus `sim/graph_coloring.py` and `solvers/vbd/rigid_vbd_kernels.py`.
  Files keep their original paths inside the Newton tree.
