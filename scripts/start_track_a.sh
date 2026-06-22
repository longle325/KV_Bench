#!/usr/bin/env bash
set -Eeuo pipefail

# Run Track A: independent replicas, cache-mode matrix, and Redis proof.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NODE_ROOT=${NODE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
ROOT=${ROOT:-$NODE_ROOT}
ENV_FILE=${ENV_FILE:-$NODE_ROOT/.env}

export NODE_ROOT ROOT ENV_FILE
export PYTHONPATH="$NODE_ROOT:${PYTHONPATH:-}"

exec python3 -m kvbench_physical run-track-a-full "$@"
