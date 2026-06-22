"""vLLM startup helpers for independent replicas and Ray-backed serving."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from ..config import Config, utc_stamp
from ..shell import cuda_exports, log, q, qjoin, run_runtime


KV_TRANSFER_CONFIG = {
    "kv_connector": "LMCacheConnectorV1Dynamic",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_connector_v1",
}


def _with_extra_args(cfg: Config, args: list[str]) -> list[str]:
    extra = cfg.get("VLLM_EXTRA_ARGS", "")
    if extra:
        args.extend(shlex.split(extra))
    return args


def start_distributed_vllm_a(cfg: Config) -> None:
    model = cfg.get("MODEL")
    args = _with_extra_args(
        cfg,
        [
            "vllm",
            "serve",
            model,
            "--served-model-name",
            model,
            "--host",
            cfg.get("HOST", cfg.get("VLLM_BIND_HOST")),
            "--port",
            cfg.get("PORT"),
            "--distributed-executor-backend",
            "ray",
            "--tensor-parallel-size",
            cfg.get("TP", cfg.get("TRACK_B_TP")),
            "--pipeline-parallel-size",
            cfg.get("PP", cfg.get("TRACK_B_PP")),
            "--max-model-len",
            cfg.get("MAX_MODEL_LEN", cfg.get("TRACK_B_MAX_MODEL_LEN")),
            "--dtype",
            cfg.get("DTYPE"),
            "--gpu-memory-utilization",
            cfg.get("GPU_MEMORY_UTILIZATION", cfg.get("VLLM_GPU_MEMORY_UTILIZATION")),
            "--no-enable-prefix-caching",
        ],
    )
    body = f"""
{cuda_exports()}
export NCCL_DEBUG="$NCCL_DEBUG"
export NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS"
export NCCL_IB_DISABLE="$NCCL_IB_DISABLE"
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="$RAY_PYTHON_VERSION_MATCH_LEVEL"
export VLLM_HOST_IP="${{VLLM_HOST_IP:-$A_IP}}"
export VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"
# The public API port is passed by --port. Leaving env VLLM_PORT set makes
# vLLM reuse that value as the base for several internal port probes.
unset VLLM_PORT
echo '[vllm-distributed] api_port={cfg.get('PORT')} internal_ports=auto'
exec {qjoin(args)}
"""
    run_runtime(cfg, body, tools=["vllm", "ray"])


def start_replica(cfg: Config) -> None:
    node = cfg.get("NODE", "A")
    mode = cfg.get("MODE", cfg.get("TRACK_A_MODE"))
    model = cfg.get("MODEL")
    run_stamp = cfg.get("RUN_STAMP", utc_stamp(""))
    log_dir = Path(cfg.get("LOG_DIR", str(cfg.paths.node_root / "logs" / run_stamp)))
    log_dir.mkdir(parents=True, exist_ok=True)

    env_exports = [
        f"export CUDA_VISIBLE_DEVICES={q(cfg.get('GPU', cfg.get('TRACK_A_GPU_A')))}",
        'export PYTHONHASHSEED="$PYTHONHASHSEED"',
        'export VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"',
        "unset VLLM_PORT",
        "unset LMCACHE_USE_EXPERIMENTAL LMCACHE_CONFIG_FILE LMCACHE_REMOTE_URL LMCACHE_LOCAL_CPU LMCACHE_MAX_LOCAL_CPU_SIZE",
    ]
    if mode == "lmcache_cpu":
        config_path = log_dir / f"lmcache_cpu_{node}.yaml"
        _write_lmcache_config(cfg, config_path, remote_url=None, chunk_size=cfg.get("LMCACHE_CPU_CHUNK_SIZE"))
        env_exports.extend(
            [
                "export LMCACHE_USE_EXPERIMENTAL=True",
                f"export LMCACHE_CONFIG_FILE={q(config_path)}",
            ]
        )
    elif mode == "lmcache_redis":
        config_path = log_dir / f"lmcache_redis_{node}.yaml"
        _write_lmcache_config(cfg, config_path, remote_url=cfg.get("REDIS_URL"), chunk_size=cfg.get("LMCACHE_REDIS_CHUNK_SIZE"))
        env_exports.extend(
            [
                "export LMCACHE_USE_EXPERIMENTAL=True",
                f"export LMCACHE_CONFIG_FILE={q(config_path)}",
            ]
        )
    elif mode != "no_cache":
        raise SystemExit(f"unknown MODE={mode}; use no_cache, lmcache_cpu, or lmcache_redis")

    args = _with_extra_args(
        cfg,
        [
            "vllm",
            "serve",
            model,
            "--served-model-name",
            model,
            "--host",
            cfg.get("HOST", cfg.get("VLLM_BIND_HOST")),
            "--port",
            cfg.get("PORT"),
            "--tensor-parallel-size",
            cfg.get("TP", cfg.get("REPLICA_TP")),
            "--max-model-len",
            cfg.get("MAX_MODEL_LEN", cfg.get("REPLICA_MAX_MODEL_LEN")),
            "--max-num-seqs",
            cfg.get("MAX_NUM_SEQS", cfg.get("REPLICA_MAX_NUM_SEQS")),
            "--dtype",
            cfg.get("DTYPE"),
            "--gpu-memory-utilization",
            cfg.get("GPU_MEMORY_UTILIZATION", cfg.get("VLLM_GPU_MEMORY_UTILIZATION")),
            "--no-enable-prefix-caching",
            "--api-key",
            cfg.get("LLM_API_KEY", "EMPTY"),
        ],
    )
    if mode != "no_cache":
        args.extend(["--kv-transfer-config", json.dumps(KV_TRANSFER_CONFIG, separators=(",", ":"))])

    server_log = log_dir / f"vllm_{mode}_{node}.log"
    log("vllm-replica", f"host={cfg.get('HOSTNAME', '')} node={node} mode={mode} port={cfg.get('PORT')} model={model}")
    body = f"""
{cuda_exports()}
{chr(10).join(env_exports)}
echo '[vllm-replica] node={node} mode={mode} port={cfg.get('PORT')} model={model}' | tee -a {q(server_log)}
echo '[vllm-replica] log={server_log}' | tee -a {q(server_log)}
{qjoin(args)} 2>&1 | tee -a {q(server_log)}
"""
    run_runtime(cfg, body, tools=["vllm"])


def _write_lmcache_config(cfg: Config, path: Path, *, remote_url: str | None, chunk_size: str) -> None:
    lines = [
        f"chunk_size: {chunk_size}",
        f"local_cpu: {cfg.get('LMCACHE_LOCAL_CPU')}",
        f"max_local_cpu_size: {cfg.get('LMCACHE_MAX_LOCAL_CPU_SIZE')}",
        f"remote_url: {json.dumps(remote_url) if remote_url else 'null'}",
        f"remote_serde: {json.dumps(cfg.get('LMCACHE_REMOTE_SERDE'))}",
    ]
    if remote_url:
        lines.extend(['store_location: "RemoteBackend"', "retrieve_locations:", '  - "RemoteBackend"'])
    path.write_text("\n".join(lines) + "\n")
