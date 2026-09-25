# GR00T baseline (NVIDIA Isaac-GR00T N1.7)

A 3B vision-language-action model as a closed-loop baseline for the deformablebench
cloth/bag bimanual tasks. This directory is the **single front door** for the
baseline: training, serving, and the eval adapter, plus this guide.

## Design in one paragraph

GR00T runs in **its own env** (Python 3.10 / torch 2.7), which cannot coexist
with deformablebench's Python 3.12 `.venv`. So it is not a pip extra — instead the
model lives in the pinned submodule `thirdparty/isaac_gr00t` (a fork of
NVIDIA/Isaac-GR00T carrying LeRobot-v3.0 loader support + the dual-piper
embodiment config), and it is **served out-of-process over ZMQ**. The eval loop,
running in the sim env, connects through the thin adapter in `adapter.py`
(`--policy_type gr00t`). That ZMQ boundary is what lets a heavy, env-conflicting
baseline be evaluated without touching the core install — each baseline is
opt-in. What lives where:

| Concern | Where | Runs in |
|---|---|---|
| Model, training (`finetune.sh`), server (`run_gr00t_server.py`), embodiment schema | submodule `thirdparty/isaac_gr00t` | GR00T env (py3.10) |
| `train.sh` / `serve.sh` — thin facades filling deformablebench defaults | here | GR00T env (py3.10) |
| `adapter.py` — ZMQ eval client (the only deformablebench-runtime code) | here | sim/eval env (py3.12) |

The dual-piper embodiment (14-D state/action, 3 cameras) is identical across all
deformablebench cloth/bag tasks, so one config serves them all:
`thirdparty/isaac_gr00t/examples/DeformableBimanual/{modality.json,dual_piper_config.py}`.

State/action layout (14-D): `[left_arm_q1..q6, left_gripper, right_arm_q1..q6,
right_gripper]`. Arms are predicted RELATIVE to the current state, grippers
ABSOLUTE; the server reconstructs absolute joint targets, so actions feed the sim
directly. Cameras map `static_cam→static`, `left_hand_cam→left_wrist`,
`right_hand_cam→right_wrist`.

---

## 0. One-time setup

### 0a. Fetch the submodule (code only; LFS is optional)

```bash
git submodule update --init thirdparty/isaac_gr00t
# The submodule is pinned with LFS smudge skipped (code is ~17 MB). You do NOT
# need the LFS blobs (prebuilt wheels / media) to train or serve from source.
```

### 0b. Build the GR00T env (py3.10, separate from deformablebench's .venv)

```bash
cd thirdparty/isaac_gr00t
uv venv --python 3.10 && source .venv/bin/activate
uv sync --all-extras           # or the platform script under scripts/deployment/
```

Notes / known env quirks:
- **flash-attn**: the official wheel needs glibc ≥ 2.32; on Ubuntu 20.04 the
  import fails but GR00T falls back to `sdpa` attention automatically — fine.
- **torchcodec** (video decode for training) needs FFmpeg 7 libs on
  `LD_LIBRARY_PATH`.
- **CN networks**: use a pip mirror (e.g. Tsinghua) and `curl -4` for torch
  wheels if IPv6 / huggingface.co is blocked.

### 0c. Base model + VLM backbone

- `GR00T-N1.7-3B` checkpoint (the base to finetune from).
- `nvidia/Cosmos-Reason2-2B` backbone (gated). **Its local path must contain the
  substring `nvidia/Cosmos-Reason2`** (a load-time check requires it), and the
  base model's `config.json` / `processor_config.json` must point at that local
  path.

### 0d. Eval-side adapter deps (in deformablebench's .venv, sim env)

```bash
uv sync --extra gr00t     # installs only pyzmq + msgpack + msgpack-numpy
```

---

## 1. Train  (run in the GR00T env)

GR00T reads a `meta/modality.json` inside the dataset. Add it once (additive, does
not touch your robot data):

```bash
cp thirdparty/isaac_gr00t/examples/DeformableBimanual/modality.json \
   data/lerobot_dataset/<task>/meta/modality.json
```

Then launch via the facade (fills embodiment + modality config for you):

```bash
BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B \
DATASET_PATH=$PWD/data/lerobot_dataset/grasp_silk_bimanual \
OUTPUT_DIR=/ssd/gr00t_out/grasp_silk \
NUM_GPUS=8 MAX_STEPS=20000 USE_WANDB=1 \
bash src/deformablebench/bench/policy/gr00t/train.sh
```

Defaults (from `finetune.sh`, override via env): `MAX_STEPS=10000`,
`GLOBAL_BATCH_SIZE=32`, `SAVE_STEPS=1000`, `lr=1e-4`, warmup 0.05. The default
finetune freezes the LLM + vision tower and trains the projector + the 1.09B-param
diffusion action head; batch 32 uses ~40 GB/GPU. Multi-GPU uses `torchrun`;
`NUM_GPUS=1` pins one GPU to avoid a DataParallel crash. Measured throughput:
8×A100 ≈ 0.51 s/step (~2.8 h for 20k steps). Checkpoints land in
`OUTPUT_DIR/checkpoint-<step>`.

---

## 2. Serve  (run in the GR00T env, one GPU)

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/ssd/gr00t_out/grasp_silk/checkpoint-20000 PORT=5555 \
bash src/deformablebench/bench/policy/gr00t/serve.sh
```

Leave it running; it prints readiness then serves on the port. Model load takes
~1–2 min (the adapter pings for up to 4 min).

---

## 3. Evaluate  (run in the sim env)

The adapter is loaded by name through `policy._base.load_policy("gr00t", args)`.
Once the R5 eval runner exists it wires up as:

```bash
python -m deformablebench.bench.eval_sim ... \
    --policy_type gr00t --gr00t_port 5555 \
    --gr00t_task "dual-arm grasp silk and lift" \
    --gr00t_exec_horizon 16
```

Adapter flags (see `adapter.py::add_arguments`): `--gr00t_host` (default
127.0.0.1), `--gr00t_port` (5555), `--gr00t_task`, `--gr00t_exec_horizon` (16;
how many of the 16 predicted steps to execute before re-querying).

**Standalone inference smoke** (verify serve+adapter before the full eval), run
in deformablebench's `.venv` with `--extra gr00t`:

```python
import numpy as np
from types import SimpleNamespace
from deformablebench.bench.policy.gr00t import build

pol = build(SimpleNamespace(gr00t_port=5555))
pol.reset()
obs = {
    "observation.state": np.zeros(14, np.float32),
    "observation.images.static_cam": np.zeros((720, 1280, 3), np.uint8),
    "observation.images.left_hand_cam": np.zeros((720, 1280, 3), np.uint8),
    "observation.images.right_hand_cam": np.zeros((720, 1280, 3), np.uint8),
}
a = pol.predict(obs, "dual-arm grasp silk and lift")
print(a.shape, a.dtype)   # (14,) float32 absolute joint targets
```

---

## Troubleshooting

- **`No built-in modality config for embodiment tag 'new_embodiment'`** — pass
  `--modality-config-path` (the facades already do). It imports the `.py` which
  registers the dual-piper config.
- **`Unsupported model name` / backbone not found** — the Cosmos path doesn't
  contain `nvidia/Cosmos-Reason2`, or `config.json` still points at a remote id.
- **Adapter hangs on connect** — the server isn't up yet or the port/host is
  wrong; the adapter waits ~4 min for model load before giving up.
- **Actions look wrong / arms drift** — confirm the sim state is exactly the
  14-D layout above and cameras are RGB uint8 at the trained resolution.
