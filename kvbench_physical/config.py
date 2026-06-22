"""Configuration loading for the physical-node benchmark runner.

The config layer intentionally keeps environment files simple and moves stable
implementation defaults into Python, where they are easier to validate.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp(prefix: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{now}" if prefix else now


@dataclass(frozen=True)
class Paths:
    package_root: Path
    node_root: Path
    repo_root: Path
    scripts_dir: Path
    env_file: Path


class Config:
    """Loaded benchmark configuration.

    Values come from process environment, then local .env, then Python defaults.
    Process environment wins so one-off CLI overrides remain predictable even
    though .env is a plain KEY="value" file.
    """

    def __init__(self, env: dict[str, str], paths: Paths):
        self.env = env
        self.paths = paths

    @classmethod
    def load(cls) -> "Config":
        package_root = Path(__file__).resolve().parent
        node_root = Path(os.environ.get("NODE_ROOT", package_root.parent)).resolve()
        repo_root = Path(os.environ.get("ROOT", node_root)).resolve()
        scripts_dir = Path(os.environ.get("SCRIPT_DIR", node_root / "scripts")).resolve()
        env_file = Path(os.environ.get("ENV_FILE", node_root / ".env")).resolve()
        paths = Paths(package_root, node_root, repo_root, scripts_dir, env_file)

        env = os.environ.copy()
        if env_file.exists():
            loaded = _source_env(env_file, env)
            for key, value in loaded.items():
                env.setdefault(key, value)

        env.setdefault("ROOT", str(repo_root))
        env.setdefault("NODE_ROOT", str(node_root))
        env.setdefault("SCRIPT_DIR", str(scripts_dir))
        env.setdefault("ENV_FILE", str(env_file))
        _apply_defaults(env)
        return cls(env, paths)

    def get(self, key: str, default: str | None = None) -> str:
        value = self.env.get(key, default)
        if value is None:
            raise KeyError(f"missing required env value: {key}")
        return value

    def int(self, key: str, default: int | None = None) -> int:
        value = self.env.get(key)
        if value is None:
            if default is None:
                raise KeyError(f"missing required env value: {key}")
            return default
        return int(value)

    def float(self, key: str, default: float | None = None) -> float:
        value = self.env.get(key)
        if value is None:
            if default is None:
                raise KeyError(f"missing required env value: {key}")
            return default
        return float(value)

    def bool(self, key: str, default: bool = False) -> bool:
        value = self.env.get(key)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    def merged_env(self, **overrides: object) -> dict[str, str]:
        env = self.env.copy()
        for key, value in overrides.items():
            if value is not None:
                env[key] = str(value)
        return env

    @property
    def remote_login(self) -> str:
        return f"{self.get('REMOTE_USER')}@{self.get('REMOTE_HOST')}"


def _source_env(env_file: Path, base_env: dict[str, str]) -> dict[str, str]:
    """Load a shell-compatible .env file and return its exported environment."""

    proc = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(env_file)],
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to source {env_file}:\n{proc.stderr.decode(errors='replace')}"
        )

    loaded: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        loaded[key.decode()] = value.decode(errors="replace")
    return loaded


def _apply_defaults(env: dict[str, str]) -> None:
    """Keep .env small while preserving stable benchmark defaults."""

    home = env.get("HOME", str(Path.home()))

    def set_default(key: str, value: str) -> None:
        env.setdefault(key, value)

    set_default("A_IP", "10.0.0.10")
    set_default("B_IP", "10.0.0.11")
    set_default("A_IFACE", "eth0")
    set_default("B_IFACE", "eth0")

    set_default("REMOTE_USER", "benchmark")
    set_default("REMOTE_HOST", env["B_IP"])
    set_default("SSH_PORT", "22")
    set_default("SSH_CONNECT_TIMEOUT", "10")
    set_default("REMOTE_ROOT", "/tmp/kvbench_physical")

    set_default("CONDA_ENV", "kv_bench")
    set_default("VENV_DIR", f"{home}/kv_bench_runtime/.venv")

    set_default("VLLM_BIND_HOST", "0.0.0.0")
    set_default("PORT", "8001")
    set_default("REDIS_PORT", "6379")
    set_default("REDIS_BIND_LOCAL", "127.0.0.1")
    set_default("REDIS_URL", f"redis://{env['A_IP']}:{env['REDIS_PORT']}")

    set_default("RAY_GCS_PORT", "6378")
    set_default("GCS_PORT", env["RAY_GCS_PORT"])
    set_default("RAY_MIN_WORKER_PORT", "10002")
    set_default("RAY_MAX_WORKER_PORT", "10100")
    set_default("MIN_WORKER_PORT", env["RAY_MIN_WORKER_PORT"])
    set_default("MAX_WORKER_PORT", env["RAY_MAX_WORKER_PORT"])
    set_default("RAY_NUM_GPUS", "1")
    set_default("RAY_BLOCK", "0")
    set_default("RAY_HEAD_BLOCK", "1")
    set_default("RAY_WORKER_BLOCK", "1")
    set_default("RAY_DASHBOARD_HOST", "0.0.0.0")
    set_default("RAY_DASHBOARD_PORT", "8265")
    set_default("RAY_PYTHON_VERSION_MATCH_LEVEL", "minor")

    set_default("NCCL_DEBUG", "INFO")
    set_default("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,COLL")
    set_default("NCCL_IB_DISABLE", "1")
    set_default("PYTHONHASHSEED", "0")

    set_default("MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    set_default("DTYPE", "float16")
    set_default("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    set_default("VLLM_EXTRA_ARGS", "--enforce-eager --attention-backend TRITON_ATTN")
    set_default("VLLM_USE_FLASHINFER_SAMPLER", "0")

    set_default("TRACK_A_PLAN", "proof,matrix")
    set_default("TRACK_A_MODE", "lmcache_redis")
    set_default("TRACK_A_MODES", "no_cache,lmcache_cpu,lmcache_redis")
    set_default("TRACK_A_WORKLOADS", "unique,repeated_prefix,rag_like")
    set_default("TRACK_A_ROUTINGS", "same,alternating")
    set_default("TRACK_A_WORKLOAD", "repeated_prefix")
    set_default("TRACK_A_ROUTING", "prime_a_then_b")
    set_default("TRACK_A_REQUESTS", "80")
    set_default("TRACK_A_PROOF_REQUESTS", env["TRACK_A_REQUESTS"])
    set_default("TRACK_A_MATRIX_REQUESTS", "300")
    set_default("TRACK_A_CONCURRENCY", "1")
    set_default("TRACK_A_PREFIX_TOKENS", "3000")
    set_default("TRACK_A_MAX_TOKENS", "1")
    set_default("TRACK_A_CACHE_CHUNK_SIZE", "16")
    set_default("TRACK_A_REQUEST_TIMEOUT", "900")
    set_default("TRACK_A_HEALTH_TIMEOUT_SECONDS", "900")
    set_default("TRACK_A_PORT_A", env["PORT"])
    set_default("TRACK_A_PORT_B", env["PORT"])
    set_default("TRACK_A_GPU_A", "0")
    set_default("TRACK_A_GPU_B", "0")
    set_default("REPLICA_TP", "1")
    set_default("REPLICA_MAX_MODEL_LEN", "4096")
    set_default("REPLICA_MAX_NUM_SEQS", "8")

    set_default("TRACK_B_CASES", "pp2,tp2,single_a")
    set_default("TRACK_B_CONFIG_NAME", f"pp2_{env['A_IFACE']}_{env['B_IFACE']}")
    set_default("TRACK_B_MODE", "no_cache")
    set_default("TRACK_B_WORKLOAD", "repeated_prefix")
    set_default("TRACK_B_ROUTING", "same")
    set_default("TRACK_B_TP", "1")
    set_default("TRACK_B_PP", "2")
    set_default("TRACK_B_SINGLE_GPU", "0")
    set_default("TRACK_B_REQUESTS", "80")
    set_default("TRACK_B_CONCURRENCY", "1")
    set_default("TRACK_B_PREFIX_TOKENS", "2048")
    set_default("TRACK_B_MAX_TOKENS", "1")
    set_default("TRACK_B_MAX_MODEL_LEN", "8192")
    set_default("TRACK_B_REQUEST_TIMEOUT", "900")
    set_default("TRACK_B_HEALTH_TIMEOUT_SECONDS", "900")
    set_default("STOP_EXISTING", "1")
    set_default("START_B_RAY", "1")
    set_default("RUN_BENCH", "1")
    set_default("WAIT_BENCH", "1")
    set_default("CLEANUP_AFTER_RUN", "1")
    set_default("BENCH_HEALTH_TIMEOUT_SECONDS", "60")
    set_default("BENCH_WAIT_HEALTH", "1")
    set_default("BENCH_WAIT_TAIL_LINES", "80")

    set_default("RAY_SMOKE_STOP_EXISTING", "1")
    set_default("RAY_SMOKE_MATRIX_SIZE", "8192")
    set_default("RAY_SMOKE_MATRIX_REPEATS", "2")
    set_default("RAY_SMOKE_DTYPE", "float16")
    set_default("RAY_SMOKE_RUN_NCCL", "1")
    set_default("RAY_SMOKE_NCCL_ELEMENTS", "67108864")
    set_default("RAY_SMOKE_NCCL_TIMEOUT_SECONDS", "180")

    set_default("PREFLIGHT_BIND_HOST", "0.0.0.0")
    set_default("PREFLIGHT_PORT_A", "19091")
    set_default("PREFLIGHT_PORT_B", "19090")
    set_default("PREFLIGHT_SERVER_TIMEOUT_SECONDS", "15")
    set_default("PREFLIGHT_CONNECT_TIMEOUT_SECONDS", "10")
    set_default("PREFLIGHT_RECV_BYTES", "128")
    set_default("PREFLIGHT_PING_COUNT", "3")
    set_default("PREFLIGHT_PING_WAIT_SECONDS", "2")
    set_default("PREFLIGHT_STARTUP_SLEEP_SECONDS", "1")
    set_default("IPERF_PARALLEL_STREAMS", "4")
    set_default("IPERF_SECONDS", "10")
    set_default("NET_MONITOR_SECONDS", "60")

    set_default("WAIT_TCP_TIMEOUT_SECONDS", "180")
    set_default("WAIT_HTTP_TIMEOUT_SECONDS", "300")
    set_default("RAY_CLUSTER_TIMEOUT_SECONDS", "240")
    set_default("PYTHON_CHECK_TIMEOUT_SECONDS", "20")
    set_default("BENCH_EXTRA_TIMEOUT_SECONDS", "180")
    set_default("REDIS_RESTART_SLEEP_SECONDS", "1")
    set_default("REDIS_STARTUP_SLEEP_SECONDS", "1")
    set_default("POLL_INTERVAL_SECONDS", "2")
    set_default("CURL_MAX_TIME_SECONDS", "3")
    set_default("LOG_TAIL_LINES", "160")
    set_default("BENCH_LOG_TAIL_LINES", "100")

    set_default("LMCACHE_CPU_CHUNK_SIZE", "256")
    set_default("LMCACHE_REDIS_CHUNK_SIZE", "16")
    set_default("LMCACHE_LOCAL_CPU", "true")
    set_default("LMCACHE_MAX_LOCAL_CPU_SIZE", "8")
    set_default("LMCACHE_REMOTE_SERDE", "naive")

    set_default("PYTHON_BIN", "python3")
    set_default("PACKAGE_SET", "replica")
    set_default("WHEEL_FIND_LINKS", "")
    set_default("VLLM_PACKAGE", "vllm==0.21.0")
    set_default(
        "LMCACHE_PACKAGE",
        "https://files.pythonhosted.org/packages/77/a1/b5aa14a3c8f095c180b6df0a9e7bddd44e1d446726148f5bc3a7b6afad36/lmcache-0.4.3.tar.gz",
    )
    set_default("TORCH_PACKAGE", "torch==2.11.0+cu129")
    set_default("TORCHVISION_PACKAGE", "torchvision==0.21.0+cu129")
    set_default("REDIS_PACKAGE", "redis==7.4.0")
    set_default("REQUESTS_PACKAGE", "requests>=2.31.0")
    set_default("RAY_PACKAGE", "ray")
    set_default("UNINSTALL_TORCHVISION", "0")
    set_default("PATCH_TORCHVISION_NMS", "1")
    set_default(
        "LMCACHE_DEPS",
        "numpy<=2.2.6 aiofile aiofiles awscrt cufile-python cupy-cuda12x nixl nvtx opentelemetry-exporter-prometheus setuptools-scm sortedcontainers",
    )
