#!/usr/bin/env bash
set -Eeuo pipefail

# Start Track B: Ray cluster + distributed vLLM + benchmark.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NODE_ROOT=${NODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
ROOT=${ROOT:-$NODE_ROOT}
ENV_FILE=${ENV_FILE:-$NODE_ROOT/.env}
RUN_STAMP=${RUN_STAMP:-phys_track_b_full_$(date -u +%Y%m%dT%H%M%SZ)}
LOG_DIR="$NODE_ROOT/logs/$RUN_STAMP"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${KV_BENCH_CONDA_ENV:-kv_bench}" >/dev/null 2>&1 || true
fi

mkdir -p "$LOG_DIR"

export NODE_ROOT ROOT ENV_FILE RUN_STAMP
export PYTHONPATH="$NODE_ROOT:${PYTHONPATH:-}"

echo "[start-track-b] run_stamp=$RUN_STAMP"
echo "[start-track-b] driver_log=$LOG_DIR/driver.log"
echo "[start-track-b] entrypoint=run-track-b-full-cluster"

python3 -m kvbench_physical run-track-b-full-cluster "$@" 2>&1 | tee -a "$LOG_DIR/driver.log"
