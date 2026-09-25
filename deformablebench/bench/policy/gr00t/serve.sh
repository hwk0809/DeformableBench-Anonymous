#!/usr/bin/env bash
# Serve a trained GR00T checkpoint over ZMQ for closed-loop eval.
#
# Thin facade over the submodule's gr00t/eval/run_gr00t_server.py — it only
# fills in the dual-piper embodiment + modality config and forwards the rest.
# The server already supports custom embodiments, so there is no deformablebench-side
# serving code beyond this launcher.
#
# RUN THIS IN THE GR00T ENV (py3.10 uv venv), NOT deformablebench's .venv. The eval
# connects from the sim env via the ZMQ adapter (policy.gr00t, --policy_type
# gr00t). Full guide: README.md next to this file.
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-20000 [PORT=5555 DEVICE=cuda] \
#   bash serve.sh [-- <extra run_gr00t_server.py args>]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
GR00T="${GR00T_DIR:-$ROOT/thirdparty/isaac_gr00t}"

: "${MODEL_PATH:?set MODEL_PATH to a trained GR00T checkpoint dir}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
MODALITY_CFG="${MODALITY_CONFIG_PATH:-$GR00T/examples/DeformableBimanual/dual_piper_config.py}"

exec python "$GR00T/gr00t/eval/run_gr00t_server.py" \
  --model-path "$MODEL_PATH" \
  --embodiment-tag new_embodiment \
  --modality-config-path "$MODALITY_CFG" \
  --device "$DEVICE" \
  --port "$PORT" \
  "$@"
