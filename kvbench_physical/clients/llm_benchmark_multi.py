#!/usr/bin/env python3
"""Two-endpoint OpenAI-compatible streaming benchmark for TTFT.

The client is intentionally dependency-light and records one JSON object per
request. It supports same-instance, alternating A/B, and A-prime/B-route
routing so LMCache CPU and Redis behavior can be compared across independent
vLLM replicas.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def pct(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * percentile / 100.0
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    weight = k - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(
    base_url: str,
    path: str,
    api_key: str | None,
    payload: dict[str, Any],
    timeout: float,
) -> requests.Response:
    url = base_url.rstrip("/") + path
    return requests.post(url, headers=auth_headers(api_key), json=payload, timeout=timeout)


def discover_model(base_url: str, api_key: str | None, timeout: float) -> str:
    response = requests.get(
        base_url.rstrip("/") + "/v1/models",
        headers=auth_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or body.get("models") or []
    if not data:
        raise RuntimeError(f"No models returned by {base_url}/v1/models")
    first = data[0]
    return first.get("id") or first.get("model") or first.get("name")


def tokenize_count(
    base_url: str,
    api_key: str | None,
    content: str,
    timeout: float,
) -> int | None:
    if not content:
        return 0
    try:
        response = post_json(base_url, "/tokenize", api_key, {"prompt": content}, timeout)
        if response.status_code >= 400:
            return None
        body = response.json()
        tokens = body.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        count = body.get("count")
        if isinstance(count, int):
            return count
    except Exception:
        return None
    return None


def make_text_near_tokens(
    base_url: str,
    api_key: str | None,
    target_tokens: int,
    label: str,
    seed: int,
    timeout: float,
) -> tuple[str, int | None]:
    rng = random.Random(seed)
    facts = [
        "service request tracing",
        "database lock contention",
        "retrieval augmented generation",
        "payment reconciliation",
        "policy validation",
        "feature flag rollout",
        "token budget accounting",
        "incident response timeline",
    ]
    unit = (
        f"{label} reference block. "
        f"The benchmark paragraph describes {rng.choice(facts)}, "
        "records stable identifiers, and repeats neutral prose to create "
        "a long prompt prefix without changing the expected answer. "
        f"Section marker {seed}. "
    )
    unit_tokens = tokenize_count(base_url, api_key, unit, timeout)
    if not unit_tokens:
        approx_words = max(1, int(target_tokens * 0.75))
        words = (unit.split() * ((approx_words // max(1, len(unit.split()))) + 2))[
            :approx_words
        ]
        return " ".join(words), None

    repeats = max(1, target_tokens // unit_tokens)
    text = unit * repeats
    count = tokenize_count(base_url, api_key, text, timeout)
    while count is not None and count < target_tokens:
        text += unit
        count = tokenize_count(base_url, api_key, text, timeout)
    return text, count


def route_for_index(args: argparse.Namespace, idx: int) -> dict[str, Any]:
    if args.routing == "same":
        instance = "A"
    elif args.routing == "alternating":
        instance = "A" if idx % 2 == 0 else "B"
    elif args.routing == "prime_a_then_b":
        instance = "A" if idx == 0 else "B"
    else:
        raise ValueError(f"unknown routing: {args.routing}")

    base_url = args.base_url_a if instance == "A" else args.base_url_b
    gpu_id = args.gpu_id_a if instance == "A" else args.gpu_id_b
    parsed = urlparse(base_url)
    return {
        "instance": instance,
        "base_url": base_url,
        "gpu_id": gpu_id,
        "port": parsed.port,
    }


def cache_scope(args: argparse.Namespace, instance: str, prefix_id: str) -> tuple[str, str]:
    if args.mode == "lmcache_cpu":
        return instance, prefix_id
    if args.mode == "lmcache_redis":
        return "global", prefix_id
    return "logical", prefix_id


def build_workload(
    args: argparse.Namespace,
    api_key: str | None,
) -> list[dict[str, Any]]:
    tokenizer_url = args.base_url_a
    fixed_prefix, fixed_prefix_count = make_text_near_tokens(
        tokenizer_url,
        api_key,
        args.prefix_tokens,
        "Shared prefix",
        args.seed,
        args.request_timeout,
    )

    rag_docs = []
    for doc_idx in range(max(1, args.rag_documents)):
        doc, count = make_text_near_tokens(
            tokenizer_url,
            api_key,
            args.prefix_tokens,
            f"RAG document {doc_idx}",
            args.seed + 100 + doc_idx,
            args.request_timeout,
        )
        rag_docs.append((doc, count))

    seen_scopes: set[tuple[str, str]] = set()
    estimated_input_cache: dict[tuple[str, str], int | None] = {}
    prompts: list[dict[str, Any]] = []
    for idx in range(args.requests):
        route = route_for_index(args, idx)
        question = (
            f"Question {idx}: answer with at most {max(1, args.max_tokens)} token(s). "
            "Return a short deterministic answer."
        )

        if args.workload == "unique":
            context, prefix_count = make_text_near_tokens(
                tokenizer_url,
                api_key,
                args.prefix_tokens,
                f"Unique context {idx}",
                args.seed + idx + 1_000,
                args.request_timeout,
            )
            prefix_id = f"unique-{idx}"
        elif args.workload == "repeated_prefix":
            context, prefix_count = fixed_prefix, fixed_prefix_count
            prefix_id = "shared-prefix-0"
        elif args.workload == "rag_like":
            doc_idx = (idx // max(1, args.rag_questions_per_document)) % len(rag_docs)
            context, prefix_count = rag_docs[doc_idx]
            prefix_id = f"rag-doc-{doc_idx}"
        else:
            raise ValueError(f"unknown workload: {args.workload}")

        scope = cache_scope(args, route["instance"], prefix_id)
        is_cold = scope not in seen_scopes
        seen_scopes.add(scope)

        messages = [
            {
                "role": "system",
                "content": "You are a concise benchmark assistant. Keep answers short.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n{question}",
            },
        ]
        prompt_text = "\n".join(m["content"] for m in messages)
        cache_key = (args.workload, prefix_id)
        if args.workload in {"repeated_prefix", "rag_like"} and cache_key in estimated_input_cache:
            estimated_input_tokens = estimated_input_cache[cache_key]
        else:
            estimated_input_tokens = tokenize_count(
                tokenizer_url, api_key, prompt_text, args.request_timeout
            )
            if args.workload in {"repeated_prefix", "rag_like"}:
                estimated_input_cache[cache_key] = estimated_input_tokens
        prompts.append(
            {
                "index": idx,
                "messages": messages,
                "prefix_tokens": prefix_count or args.prefix_tokens,
                "estimated_input_tokens": estimated_input_tokens,
                "workload_prefix_id": prefix_id,
                "is_cold": is_cold,
                "is_warm": not is_cold,
                **route,
            }
        )
    return prompts


def parse_redis_url(redis_url: str | None) -> tuple[str, int] | None:
    if not redis_url:
        return None
    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", ""}:
        raise ValueError(f"Unsupported redis URL scheme: {parsed.scheme}")
    return parsed.hostname or "127.0.0.1", parsed.port or 6379


def redis_cli(redis_url: str | None, *args: str, timeout: float = 5.0) -> str | None:
    endpoint = parse_redis_url(redis_url)
    if endpoint is None:
        return None
    host, port = endpoint
    cmd = ["redis-cli", "-h", host, "-p", str(port), *args]
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_info(text: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not text:
        return result
    for line in text.splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key] = value.strip()
    return result


def redis_snapshot(redis_url: str | None) -> dict[str, Any]:
    if not redis_url:
        return {}
    dbsize = redis_cli(redis_url, "DBSIZE")
    memory = parse_info(redis_cli(redis_url, "INFO", "memory"))
    stats = parse_info(redis_cli(redis_url, "INFO", "stats"))
    commandstats = parse_info(redis_cli(redis_url, "INFO", "commandstats"))
    snap: dict[str, Any] = {}
    if dbsize is not None and dbsize.isdigit():
        snap["redis_dbsize"] = int(dbsize)
    if "used_memory" in memory:
        snap["redis_used_memory_bytes"] = int(memory["used_memory"])
        snap["redis_used_memory_mb"] = round(int(memory["used_memory"]) / 1024 / 1024, 3)
    for key in (
        "total_commands_processed",
        "total_net_input_bytes",
        "total_net_output_bytes",
        "keyspace_hits",
        "keyspace_misses",
    ):
        if key in stats and re.fullmatch(r"\d+", stats[key]):
            snap[f"redis_{key}"] = int(stats[key])
    for command in ("cmdstat_get", "cmdstat_set"):
        value = commandstats.get(command)
        if not value:
            continue
        calls = re.search(r"calls=(\d+)", value)
        if calls:
            snap[f"redis_{command}_calls"] = int(calls.group(1))
    return snap


def stream_chat_once(
    args: argparse.Namespace,
    api_key: str | None,
    model: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    request_id = f"req-{item['index']:06d}"
    payload = {
        "model": model,
        "messages": item["messages"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    started_at_unix = time.time()
    start = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    chunks = 0

    result: dict[str, Any] = {
        "request_id": request_id,
        "run_id": args.run_id,
        "mode": args.mode,
        "workload": args.workload,
        "routing_pattern": args.routing,
        "instance": item["instance"],
        "gpu_id": item["gpu_id"],
        "port": item["port"],
        "base_url": item["base_url"],
        "workload_prefix_id": item["workload_prefix_id"],
        "is_cold": item["is_cold"],
        "is_warm": item["is_warm"],
        "prefix_tokens": args.prefix_tokens,
        "actual_prefix_tokens": item["prefix_tokens"],
        "estimated_input_tokens": item["estimated_input_tokens"],
        "max_new_tokens": args.max_tokens,
        "cache_chunk_size": args.cache_chunk_size,
        "concurrency": args.concurrency,
        "timestamp": utc_now(),
        "started_at_unix": started_at_unix,
    }

    try:
        response = requests.post(
            item["base_url"].rstrip("/") + "/v1/chat/completions",
            headers=auth_headers(api_key),
            json=payload,
            stream=True,
            timeout=args.request_timeout,
        )
        result["http_status"] = response.status_code
        if response.status_code >= 400:
            result.update(
                {
                    "status": "error",
                    "error": response.text[:1000],
                    "total_latency_ms": round((time.perf_counter() - start) * 1000, 3),
                    "completed_at_unix": time.time(),
                }
            )
            return result

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            chunks += 1
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(piece)

        total_latency_ms = round((time.perf_counter() - start) * 1000, 3)
        output_text = "".join(output_parts)
        output_tokens = tokenize_count(
            item["base_url"], api_key, output_text, args.request_timeout
        )
        result.update(
            {
                "status": "success",
                "ttft_ms": round((first_token_at - start) * 1000, 3)
                if first_token_at is not None
                else None,
                "total_latency_ms": total_latency_ms,
                "completed_at_unix": time.time(),
                "chunks": chunks,
                "output_tokens": output_tokens,
                "output_chars": len(output_text),
                "tokens_per_second": round((output_tokens or 0) / (total_latency_ms / 1000), 6)
                if total_latency_ms > 0 and output_tokens is not None
                else None,
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": repr(exc),
                "total_latency_ms": round((time.perf_counter() - start) * 1000, 3),
                "completed_at_unix": time.time(),
            }
        )
        return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in records if r.get("status") == "success"]
    ttft = [r["ttft_ms"] for r in successes if isinstance(r.get("ttft_ms"), (int, float))]
    warm = [r for r in successes if r.get("is_warm")]
    warm_ttft = [r["ttft_ms"] for r in warm if isinstance(r.get("ttft_ms"), (int, float))]
    lat = [
        r["total_latency_ms"]
        for r in successes
        if isinstance(r.get("total_latency_ms"), (int, float))
    ]
    starts = [r["started_at_unix"] for r in records if isinstance(r.get("started_at_unix"), float)]
    ends = [r["completed_at_unix"] for r in records if isinstance(r.get("completed_at_unix"), float)]
    wall = max(ends) - min(starts) if starts and ends else None
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in successes)
    return {
        "requests": len(records),
        "successes": len(successes),
        "errors": len(records) - len(successes),
        "cold_count": sum(1 for r in successes if r.get("is_cold")),
        "warm_count": len(warm),
        "ttft_p50_ms": pct(ttft, 50),
        "ttft_p95_ms": pct(ttft, 95),
        "ttft_p99_ms": pct(ttft, 99),
        "warm_ttft_p50_ms": pct(warm_ttft, 50),
        "warm_ttft_p95_ms": pct(warm_ttft, 95),
        "warm_ttft_p99_ms": pct(warm_ttft, 99),
        "latency_p50_ms": pct(lat, 50),
        "latency_p95_ms": pct(lat, 95),
        "latency_p99_ms": pct(lat, 99),
        "wall_seconds": wall,
        "requests_per_second": len(successes) / wall if wall and wall > 0 else None,
        "output_tokens_per_second": output_tokens / wall if wall and wall > 0 else None,
        "output_tokens": output_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url-a", default=os.getenv("LLM_BASE_URL_A", "http://127.0.0.1:8001"))
    parser.add_argument("--base-url-b", default=os.getenv("LLM_BASE_URL_B", "http://127.0.0.1:8002"))
    parser.add_argument("--gpu-id-a", type=int, default=0)
    parser.add_argument("--gpu-id-b", type=int, default=1)
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--mode",
        choices=["no_cache", "lmcache_cpu", "lmcache_redis"],
        default="no_cache",
    )
    parser.add_argument(
        "--workload",
        choices=["unique", "repeated_prefix", "rag_like"],
        default="repeated_prefix",
    )
    parser.add_argument(
        "--routing",
        choices=["same", "alternating", "prime_a_then_b"],
        default="alternating",
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=3000)
    parser.add_argument("--cache-chunk-size", type=int)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rag-documents", type=int, default=10)
    parser.add_argument("--rag-questions-per-document", type=int, default=30)
    parser.add_argument("--redis-url", default=os.getenv("LMCACHE_REMOTE_URL"))
    parser.add_argument("--clear-redis", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear_redis and args.redis_url:
        redis_cli(args.redis_url, "FLUSHDB")

    if not args.model:
        args.model = discover_model(args.base_url_a, api_key, args.request_timeout)

    prompts = build_workload(args, api_key)
    before_redis = redis_snapshot(args.redis_url)
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    with output_path.open("a", encoding="utf-8") as fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(stream_chat_once, args, api_key, args.model, item)
                for item in prompts
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                if args.redis_url:
                    record.update(redis_snapshot(args.redis_url))
                with lock:
                    records.append(record)
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                    fh.flush()
                print(
                    f"{record['request_id']} {record.get('instance')} "
                    f"{record.get('status')} ttft_ms={record.get('ttft_ms')}",
                    flush=True,
                )

    after_redis = redis_snapshot(args.redis_url)
    summary = summarize(records)
    summary.update(
        {
            "run_id": args.run_id,
            "mode": args.mode,
            "workload": args.workload,
            "routing_pattern": args.routing,
            "concurrency": args.concurrency,
            "prefix_tokens": args.prefix_tokens,
            "cache_chunk_size": args.cache_chunk_size,
            "max_new_tokens": args.max_tokens,
            "output": str(output_path),
            "redis_before": before_redis,
            "redis_after": after_redis,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
