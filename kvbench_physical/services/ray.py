"""Ray service lifecycle helpers for the distributed Track B workflow."""

from __future__ import annotations

import time

from ..config import Config
from ..shell import die, run_runtime


def start_head_a(cfg: Config) -> None:
    body = """
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="$RAY_PYTHON_VERSION_MATCH_LEVEL"
export VLLM_HOST_IP="${VLLM_HOST_IP:-$A_IP}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$A_IFACE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$A_IFACE}"
ray stop || true
ray_args=(
  --head
  --node-ip-address="$A_IP"
  --port="$GCS_PORT"
  --dashboard-host="$RAY_DASHBOARD_HOST"
  --dashboard-port="$RAY_DASHBOARD_PORT"
  --min-worker-port="$MIN_WORKER_PORT"
  --max-worker-port="$MAX_WORKER_PORT"
  --num-gpus="$RAY_NUM_GPUS"
  --disable-usage-stats
)
if [ "$RAY_BLOCK" = "1" ]; then ray_args+=(--block); fi
ray start "${ray_args[@]}"
if [ "$RAY_BLOCK" != "1" ]; then ray status; fi
"""
    run_runtime(cfg, body, tools=["ray"])


def start_worker_b(cfg: Config) -> None:
    body = """
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="$RAY_PYTHON_VERSION_MATCH_LEVEL"
export VLLM_HOST_IP="${VLLM_HOST_IP:-$B_IP}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$B_IFACE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$B_IFACE}"
ray stop || true
ray_args=(
  --address="${A_IP}:${GCS_PORT}"
  --node-ip-address="$B_IP"
  --min-worker-port="$MIN_WORKER_PORT"
  --max-worker-port="$MAX_WORKER_PORT"
  --num-gpus="$RAY_NUM_GPUS"
  --disable-usage-stats
)
if [ "$RAY_BLOCK" = "1" ]; then ray_args+=(--block); fi
ray start "${ray_args[@]}"
if [ "$RAY_BLOCK" != "1" ]; then ray status; fi
"""
    run_runtime(cfg, body, tools=["ray"])


def stop_ray(cfg: Config) -> None:
    run_runtime(cfg, "ray stop --force >/dev/null 2>&1 || true", tools=["ray"], check=False)


def wait_for_two_nodes(cfg: Config) -> None:
    script = """
timeout "$PYTHON_CHECK_TIMEOUT_SECONDS" python - "$A_IP:$GCS_PORT" <<'PY'
import sys
import ray

ray.init(address=sys.argv[1], ignore_reinit_error=True, logging_level="ERROR")
alive = [n for n in ray.nodes() if n.get("Alive")]
print("alive_nodes=%d addresses=%s" % (len(alive), [n.get("NodeManagerAddress") for n in alive]))
raise SystemExit(0 if len(alive) >= 2 else 1)
PY
"""
    deadline = time.monotonic() + cfg.int("RAY_CLUSTER_TIMEOUT_SECONDS")
    last_output = ""
    while True:
        proc = run_runtime(
            cfg,
            script,
            tools=["python", "ray"],
            check=False,
            capture=True,
        )
        last_output = proc.stdout or ""
        if proc.returncode == 0:
            print(last_output.strip())
            return
        if time.monotonic() >= deadline:
            print(last_output)
            run_runtime(cfg, f"ray status --address {cfg.get('A_IP')}:{cfg.get('GCS_PORT')} || true", tools=["ray"], check=False)
            die("ray", "Ray cluster did not reach 2 alive nodes")
        time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))
