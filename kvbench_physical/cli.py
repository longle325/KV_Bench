"""Command-line dispatch for physical-node benchmark operations.

This module keeps the public command surface in one place and delegates all
real work to service or workflow modules.
"""

from __future__ import annotations

import argparse

from .config import Config
from .remote import Remote
from .services.ray import start_head_a, start_worker_b, stop_ray
from .services.redis import start_redis_a
from .services.runtime import bootstrap_runtime, check_runtime
from .services.vllm import start_distributed_vllm_a, start_replica
from .workflows.benchmark import run_track_a_case, run_track_b_case
from .workflows.env_check import remote_env_check
from .workflows.monitor import monitor_net_bytes
from .workflows.preflight import preflight_network
from .workflows.ray_smoke import run_ray_smoke_tests
from .workflows.track_a import run_full as run_track_a_full, start_replicas, stop_replicas
from .workflows.track_b import run_full_cluster


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvbench-physical")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync-to-b")
    sub.add_parser("bootstrap-runtime")
    check = sub.add_parser("check-runtime")
    check.add_argument("--tools", nargs="*", default=[])
    check.add_argument("--imports", nargs="*", default=[])
    sub.add_parser("remote-env-check")
    sub.add_parser("preflight-network")
    sub.add_parser("monitor-net-bytes")
    sub.add_parser("run-ray-smoke-tests")
    sub.add_parser("start-redis-a")
    sub.add_parser("start-vllm-replica")
    sub.add_parser("start-track-a-replicas")
    sub.add_parser("run-track-a-full")
    sub.add_parser("stop-track-a-replicas")
    sub.add_parser("run-track-a-case")
    sub.add_parser("start-ray-head-a")
    sub.add_parser("start-ray-worker-b")
    sub.add_parser("stop-ray")
    sub.add_parser("start-distributed-vllm-a")
    sub.add_parser("run-track-b-case")
    sub.add_parser("run-track-b-full-cluster")

    args = parser.parse_args(argv)
    cfg = Config.load()

    if args.command == "sync-to-b":
        Remote(cfg).sync_bundle()
    elif args.command == "bootstrap-runtime":
        bootstrap_runtime(cfg)
    elif args.command == "check-runtime":
        check_runtime(cfg, args.tools, args.imports)
    elif args.command == "remote-env-check":
        remote_env_check()
    elif args.command == "preflight-network":
        preflight_network(cfg)
    elif args.command == "monitor-net-bytes":
        monitor_net_bytes(cfg)
    elif args.command == "run-ray-smoke-tests":
        run_ray_smoke_tests(cfg)
    elif args.command == "start-redis-a":
        start_redis_a(cfg)
    elif args.command == "start-vllm-replica":
        start_replica(cfg)
    elif args.command == "start-track-a-replicas":
        start_replicas(cfg)
    elif args.command == "run-track-a-full":
        run_track_a_full(cfg)
    elif args.command == "stop-track-a-replicas":
        stop_replicas(cfg)
    elif args.command == "run-track-a-case":
        run_track_a_case(cfg)
    elif args.command == "start-ray-head-a":
        start_head_a(cfg)
    elif args.command == "start-ray-worker-b":
        start_worker_b(cfg)
    elif args.command == "stop-ray":
        stop_ray(cfg)
    elif args.command == "start-distributed-vllm-a":
        start_distributed_vllm_a(cfg)
    elif args.command == "run-track-b-case":
        run_track_b_case(cfg)
    elif args.command == "run-track-b-full-cluster":
        run_full_cluster(cfg)
    else:  # pragma: no cover - argparse prevents this.
        parser.error(f"unknown command: {args.command}")
    return 0
