"""Benchmark client invocation and result summarization utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import Config, utc_stamp
from ..shell import log, q, qjoin, run_runtime
from ..wait import wait_http


def run_track_a_case(cfg: Config) -> Path:
    run_stamp = cfg.get("RUN_STAMP", utc_stamp(""))
    result_dir = cfg.paths.node_root / "results" / run_stamp
    log_dir = cfg.paths.node_root / "logs" / run_stamp
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg.get("MODE", cfg.get("TRACK_A_MODE"))
    workload = cfg.get("WORKLOAD", cfg.get("TRACK_A_WORKLOAD"))
    routing = cfg.get("ROUTING", cfg.get("TRACK_A_ROUTING"))
    concurrency = cfg.get("CONCURRENCY", cfg.get("TRACK_A_CONCURRENCY"))
    requests = cfg.get("REQUESTS", cfg.get("TRACK_A_REQUESTS"))
    out = result_dir / f"{mode}_{workload}_{routing}_c{concurrency}_n{requests}.jsonl"
    log_file = log_dir / f"bench_{mode}_{workload}_{routing}_c{concurrency}_n{requests}.log"

    args = [
        "python",
        str(cfg.paths.package_root / "clients" / "llm_benchmark_multi.py"),
        "--base-url-a",
        f"http://{cfg.get('A_IP')}:{cfg.get('PORT_A', cfg.get('TRACK_A_PORT_A'))}",
        "--base-url-b",
        f"http://{cfg.get('B_IP')}:{cfg.get('PORT_B', cfg.get('TRACK_A_PORT_B'))}",
        "--api-key",
        cfg.get("LLM_API_KEY", "EMPTY"),
        "--model",
        cfg.get("MODEL"),
        "--mode",
        mode,
        "--workload",
        workload,
        "--routing",
        routing,
        "--requests",
        requests,
        "--concurrency",
        concurrency,
        "--prefix-tokens",
        cfg.get("PREFIX_TOKENS", cfg.get("TRACK_A_PREFIX_TOKENS")),
        "--max-tokens",
        cfg.get("MAX_TOKENS", cfg.get("TRACK_A_MAX_TOKENS")),
        "--request-timeout",
        cfg.get("REQUEST_TIMEOUT", cfg.get("TRACK_A_REQUEST_TIMEOUT")),
    ]
    if mode == "lmcache_redis":
        args.extend(["--redis-url", cfg.get("REDIS_URL"), "--cache-chunk-size", cfg.get("CACHE_CHUNK_SIZE", cfg.get("TRACK_A_CACHE_CHUNK_SIZE"))])
    args.extend(["--output", str(out)])

    run_runtime(cfg, f"{qjoin(args)} 2>&1 | tee -a {q(log_file)}", tools=["python"], cwd=cfg.paths.repo_root)
    log("track-a-case", f"wrote {out}")
    return out


def run_track_b_case(cfg: Config) -> Path:
    run_stamp = cfg.get("RUN_STAMP", utc_stamp(""))
    result_dir = cfg.paths.node_root / "results" / run_stamp
    log_dir = cfg.paths.node_root / "logs" / run_stamp
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_name = cfg.get("CONFIG_NAME", cfg.get("TRACK_B_CONFIG_NAME"))
    requests = cfg.get("REQUESTS", cfg.get("TRACK_B_REQUESTS"))
    prefix_tokens = cfg.get("PREFIX_TOKENS", cfg.get("TRACK_B_PREFIX_TOKENS"))
    max_tokens = cfg.get("MAX_TOKENS", cfg.get("TRACK_B_MAX_TOKENS"))
    out = result_dir / f"distributed_{config_name}_input{prefix_tokens}_out{max_tokens}_n{requests}.jsonl"
    log_file = log_dir / f"distributed_{config_name}_input{prefix_tokens}_out{max_tokens}_n{requests}.log"

    if cfg.bool("WAIT_HEALTH", True):
        url = f"http://{cfg.get('A_IP')}:{cfg.get('PORT')}/health"
        log("track-b-case", f"waiting for {url}")
        wait_http(cfg, url, "vLLM health", int(cfg.get("HEALTH_TIMEOUT_SECONDS", cfg.get("WAIT_HTTP_TIMEOUT_SECONDS"))))
        log("track-b-case", "health OK")

    args = [
        "python",
        str(cfg.paths.package_root / "clients" / "llm_benchmark_multi.py"),
        "--base-url-a",
        f"http://{cfg.get('A_IP')}:{cfg.get('PORT')}",
        "--base-url-b",
        f"http://{cfg.get('A_IP')}:{cfg.get('PORT')}",
        "--api-key",
        cfg.get("LLM_API_KEY", "EMPTY"),
        "--model",
        cfg.get("MODEL"),
        "--mode",
        cfg.get("MODE", cfg.get("TRACK_B_MODE")),
        "--workload",
        cfg.get("WORKLOAD", cfg.get("TRACK_B_WORKLOAD")),
        "--routing",
        cfg.get("ROUTING", cfg.get("TRACK_B_ROUTING")),
        "--requests",
        requests,
        "--concurrency",
        cfg.get("CONCURRENCY", cfg.get("TRACK_B_CONCURRENCY")),
        "--prefix-tokens",
        prefix_tokens,
        "--max-tokens",
        max_tokens,
        "--request-timeout",
        cfg.get("REQUEST_TIMEOUT", cfg.get("TRACK_B_REQUEST_TIMEOUT")),
        "--output",
        str(out),
    ]

    run_runtime(cfg, f"{qjoin(args)} 2>&1 | tee -a {q(log_file)}", tools=["python"], cwd=cfg.paths.repo_root)
    log("track-b-case", f"wrote {out}")
    return out


def summarize_result(result_file: Path, extra: dict[str, object] | None = None) -> dict[str, object]:
    rows = [json.loads(line) for line in result_file.read_text().splitlines() if line.strip()]
    ok = [row for row in rows if row.get("status") == "success"]

    def percentile(values: list[float], qvalue: float):
        if not values:
            return ""
        values = sorted(values)
        idx = (len(values) - 1) * qvalue
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        return round(values[lo] * (hi - idx) + values[hi] * (idx - lo), 3)

    def values(records: list[dict], key: str) -> list[float]:
        return [row[key] for row in records if isinstance(row.get(key), (int, float))]

    def scoped(prefix: str, records: list[dict]) -> dict[str, object]:
        warm = [row for row in records if row.get("is_warm")]
        cold = [row for row in records if row.get("is_cold")]
        return {
            f"{prefix}successes": len(records),
            f"{prefix}cold_count": len(cold),
            f"{prefix}warm_count": len(warm),
            f"{prefix}ttft_p50_ms": percentile(values(records, "ttft_ms"), 0.50),
            f"{prefix}ttft_p95_ms": percentile(values(records, "ttft_ms"), 0.95),
            f"{prefix}warm_ttft_p50_ms": percentile(values(warm, "ttft_ms"), 0.50),
            f"{prefix}warm_ttft_p95_ms": percentile(values(warm, "ttft_ms"), 0.95),
            f"{prefix}latency_p50_ms": percentile(values(records, "total_latency_ms"), 0.50),
            f"{prefix}latency_p95_ms": percentile(values(records, "total_latency_ms"), 0.95),
        }

    ttft = values(ok, "ttft_ms")
    lat = values(ok, "total_latency_ms")
    warm_ok = [row for row in ok if row.get("is_warm")]
    started = min((row.get("started_at_unix") for row in ok), default=None)
    completed = max((row.get("completed_at_unix") for row in ok), default=None)
    wall = completed - started if started and completed else None

    summary: dict[str, object] = {
        "run_stamp": result_file.parent.name,
        "file": str(result_file),
        "rows": len(rows),
        "successes": len(ok),
        "errors": len(rows) - len(ok),
        "prefix_tokens": ok[0].get("prefix_tokens", "") if ok else "",
        "max_new_tokens": ok[0].get("max_new_tokens", "") if ok else "",
        "cold_count": sum(1 for row in ok if row.get("is_cold")),
        "warm_count": len(warm_ok),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "warm_ttft_p50_ms": percentile(values(warm_ok, "ttft_ms"), 0.50),
        "warm_ttft_p95_ms": percentile(values(warm_ok, "ttft_ms"), 0.95),
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "wall_seconds": round(wall, 3) if wall else "",
        "rps": round(len(ok) / wall, 6) if wall else "",
    }
    summary.update(scoped("a_", [row for row in ok if row.get("instance") == "A"]))
    summary.update(scoped("b_", [row for row in ok if row.get("instance") == "B"]))
    summary.update(_redis_summary(rows))
    if extra:
        summary = {**extra, **summary}
    return summary


def _redis_summary(rows: list[dict]) -> dict[str, object]:
    redis_rows = [
        row for row in rows
        if isinstance(row.get("completed_at_unix"), (int, float))
        and any(str(key).startswith("redis_") for key in row)
    ]
    if not redis_rows:
        return {}

    redis_rows.sort(key=lambda row: row["completed_at_unix"])
    first = redis_rows[0]
    last = redis_rows[-1]
    summary: dict[str, object] = {}
    end_keys = [
        "redis_dbsize",
        "redis_used_memory_mb",
        "redis_total_commands_processed",
        "redis_total_net_input_bytes",
        "redis_total_net_output_bytes",
        "redis_keyspace_hits",
        "redis_keyspace_misses",
        "redis_cmdstat_get_calls",
        "redis_cmdstat_set_calls",
    ]
    for key in end_keys:
        if key in last:
            summary[f"{key}_end"] = last[key]
        if isinstance(last.get(key), int) and isinstance(first.get(key), int):
            summary[f"{key}_delta"] = last[key] - first[key]
    return summary


def write_summary_rows(rows: list[dict[str, object]], summary_file: Path) -> Path:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(summary_file)
    return summary_file


def write_summary_csv(result_file: Path, summary_file: Path) -> Path:
    return write_summary_rows([summarize_result(result_file)], summary_file)
