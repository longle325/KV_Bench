"""Track A orchestration for two independent replicas sharing optional Redis KV.

The full workflow runs the focused A-prime/B-route Redis proof and the PRD
cache-mode matrix from a single entrypoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter

from ..config import Config, utc_stamp
from ..remote import Remote, remote_python_command
from ..services.redis import flush_redis, start_redis_a
from ..services.runtime import bootstrap_runtime, check_runtime
from ..shell import die, log, python_module_command, run, start_screen, stop_screens_matching
from ..wait import wait_http
from .benchmark import run_track_a_case, summarize_result, write_summary_rows


@dataclass(frozen=True)
class TrackACase:
    group: str
    mode: str
    workload: str
    routing: str
    requests: int


def run_full(cfg: Config) -> None:
    run_stamp = cfg.get("RUN_STAMP", utc_stamp("phys_track_a_full"))
    local_cfg = Config(cfg.merged_env(RUN_STAMP=run_stamp), cfg.paths)
    cases = _track_a_cases(local_cfg)
    summary_file = cfg.paths.node_root / "results" / f"summary_{run_stamp}.csv"
    summary_rows: list[dict[str, object]] = []

    _ensure_local_runtime(local_cfg, cases)
    remote = Remote(local_cfg)
    remote.sync_bundle()
    _ensure_remote_runtime(local_cfg, remote, cases)

    log("track-a", f"run_id={run_stamp}")
    log("track-a", f"A={cfg.get('A_IP')}:{cfg.get('TRACK_A_PORT_A')} B={cfg.get('B_IP')}:{cfg.get('TRACK_A_PORT_B')}")
    log("track-a", "cases=" + ",".join(f"{c.group}:{c.mode}:{c.workload}:{c.routing}:n{c.requests}" for c in cases))

    for mode, mode_cases_iter in groupby(cases, key=attrgetter("mode")):
        mode_cases = list(mode_cases_iter)
        mode_cfg = Config(local_cfg.merged_env(MODE=mode, SYNC_TO_B=0), cfg.paths)

        _stop_track_a_processes(mode_cfg, remote, run_stamp)
        start_replicas(mode_cfg)
        _wait_replicas(mode_cfg)

        for case in mode_cases:
            if case.mode == "lmcache_redis":
                flush_redis(mode_cfg)

            case_cfg = Config(
                mode_cfg.merged_env(
                    MODE=case.mode,
                    WORKLOAD=case.workload,
                    ROUTING=case.routing,
                    REQUESTS=case.requests,
                    RUN_STAMP=run_stamp,
                ),
                cfg.paths,
            )
            log("track-a", f"running {case.group}: mode={case.mode} workload={case.workload} routing={case.routing} requests={case.requests}")
            result_file = run_track_a_case(case_cfg)
            summary_rows.append(
                summarize_result(
                    result_file,
                    {
                        "case_group": case.group,
                        "mode": case.mode,
                        "workload": case.workload,
                        "routing": case.routing,
                    },
                )
            )
            write_summary_rows(summary_rows, summary_file)

    log("track-a", f"summary={summary_file}")


def start_replicas(cfg: Config) -> None:
    mode = cfg.get("MODE", cfg.get("TRACK_A_MODE"))
    run_stamp = cfg.get("RUN_STAMP", utc_stamp("phys_track_a"))
    log_dir = cfg.paths.node_root / "logs" / run_stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    if mode == "lmcache_redis":
        start_redis_a(Config(cfg.merged_env(RUN_STAMP=run_stamp), cfg.paths))
        flush_redis(cfg)

    remote = Remote(cfg)
    if cfg.bool("SYNC_TO_B", True):
        remote.sync_bundle()

    screen_a = f"mph_{run_stamp}_{mode}_A"
    screen_b = f"mph_{run_stamp}_{mode}_B"
    log("track-a", f"starting A screen={screen_a}")
    start_screen(
        screen_a,
        python_module_command(
            cfg,
            "start-vllm-replica",
            NODE="A",
            MODE=mode,
            PORT=cfg.get("PORT_A", cfg.get("TRACK_A_PORT_A")),
            GPU=cfg.get("GPU_A", cfg.get("TRACK_A_GPU_A")),
            RUN_STAMP=run_stamp,
        ),
    )

    log("track-a", f"starting B screen={screen_b}")
    remote.ssh(
        "screen -dmS "
        f"{screen_b!r} bash -lc "
        + repr(
            remote_python_command(
                cfg,
                "start-vllm-replica",
                NODE="B",
                MODE=mode,
                PORT=cfg.get("PORT_B", cfg.get("TRACK_A_PORT_B")),
                GPU=cfg.get("GPU_B", cfg.get("TRACK_A_GPU_B")),
                RUN_STAMP=run_stamp,
            )
        )
    )

    log("track-a", f"A endpoint: http://{cfg.get('A_IP')}:{cfg.get('PORT_A', cfg.get('TRACK_A_PORT_A'))}")
    log("track-a", f"B endpoint: http://{cfg.get('B_IP')}:{cfg.get('PORT_B', cfg.get('TRACK_A_PORT_B'))}")
    log("track-a", f"local logs: {log_dir}")
    log("track-a", f"remote logs: {cfg.get('REMOTE_ROOT')}/logs/{run_stamp}")
    log("track-a", f"screens kept: {screen_a}, {screen_b}")


def stop_replicas(cfg: Config) -> None:
    run_stamp = cfg.get("RUN_STAMP", "")
    pattern = f"mph_{run_stamp}" if run_stamp else "mph_.*_(no_cache|lmcache_cpu|lmcache_redis)_"
    log("stop-track-a", f"sending Ctrl-C to local screens matching {pattern}")
    stop_screens_matching(pattern)
    log("stop-track-a", f"sending Ctrl-C to remote screens matching {pattern}")
    Remote(cfg).ssh(
        "screen -ls | awk -v pat="
        + repr(pattern)
        + " '$1 ~ pat {print $1}' | while read -r screen_id; "
        + "do screen -S \"$screen_id\" -X stuff $'\\003' || true; done",
        check=False,
    )
    log("stop-track-a", "screens are intentionally not deleted")


def _track_a_cases(cfg: Config) -> list[TrackACase]:
    plan = _csv(cfg.get("TRACK_A_PLAN"))
    cases: list[TrackACase] = []

    if "proof" in plan:
        cases.append(
            TrackACase(
                group="proof",
                mode="lmcache_redis",
                workload="repeated_prefix",
                routing="prime_a_then_b",
                requests=cfg.int("TRACK_A_PROOF_REQUESTS"),
            )
        )

    if "matrix" in plan:
        for mode in _csv(cfg.get("TRACK_A_MODES")):
            for workload in _csv(cfg.get("TRACK_A_WORKLOADS")):
                for routing in _csv(cfg.get("TRACK_A_ROUTINGS")):
                    cases.append(
                        TrackACase(
                            group="matrix",
                            mode=mode,
                            workload=workload,
                            routing=routing,
                            requests=cfg.int("TRACK_A_MATRIX_REQUESTS"),
                        )
                    )

    if not cases:
        die("track-a", "TRACK_A_PLAN is empty; use proof,matrix")
    return cases


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]


def _wait_replicas(cfg: Config) -> None:
    timeout = cfg.int("TRACK_A_HEALTH_TIMEOUT_SECONDS")
    wait_http(
        cfg,
        f"http://{cfg.get('A_IP')}:{cfg.get('PORT_A', cfg.get('TRACK_A_PORT_A'))}/health",
        "Track A vLLM A",
        timeout,
    )
    wait_http(
        cfg,
        f"http://{cfg.get('B_IP')}:{cfg.get('PORT_B', cfg.get('TRACK_A_PORT_B'))}/health",
        "Track A vLLM B",
        timeout,
    )
    log("track-a", "A/B replicas are healthy")


def _ensure_local_runtime(cfg: Config, cases: list[TrackACase]) -> None:
    imports = ["vllm", "torch", "requests"]
    if any(case.mode != "no_cache" for case in cases):
        imports.append("lmcache")
    try:
        check_runtime(cfg, ["python", "vllm"], imports)
        run(["screen", "-v"], capture=True)
        run(["ssh", "-V"], capture=True, check=False)
    except Exception as exc:  # noqa: BLE001
        if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
            die("runtime", f"local Track A runtime missing ({exc}). Re-run with ALLOW_RUNTIME_INSTALL=1.")
        bootstrap_runtime(Config(cfg.merged_env(PACKAGE_SET="replica", ALLOW_RUNTIME_INSTALL=1), cfg.paths))


def _ensure_remote_runtime(cfg: Config, remote: Remote, cases: list[TrackACase]) -> None:
    imports = ["vllm", "torch", "requests"]
    if any(case.mode != "no_cache" for case in cases):
        imports.append("lmcache")
    command = remote_python_command(
        cfg,
        "check-runtime --tools python vllm --imports " + " ".join(imports),
    )
    proc = remote.ssh(command, check=False, capture=True)
    if proc.returncode == 0:
        return
    if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
        die("runtime", f"remote Track A runtime missing on B. Re-run with ALLOW_RUNTIME_INSTALL=1. Output:\n{proc.stdout}")
    remote.ssh(remote_python_command(cfg, "bootstrap-runtime", PACKAGE_SET="replica", ALLOW_RUNTIME_INSTALL=1))


def _stop_track_a_processes(cfg: Config, remote: Remote, run_stamp: str) -> None:
    stop_screens_matching(f"mph_{run_stamp}")
    run(["pkill", "-f", f"vllm serve .*--port {cfg.get('TRACK_A_PORT_A')}"], check=False)
    run(["pkill", "-f", "VLLM::EngineCore"], check=False)
    run(["pkill", "-f", "llm_benchmark_multi.py"], check=False)
    remote.ssh(
        "screen -ls | awk -v pat="
        + repr(f"mph_{run_stamp}")
        + " '$1 ~ pat {print $1}' | while read -r screen_id; "
        + "do screen -S \"$screen_id\" -X stuff $'\\003' || true; done; "
        + f"pkill -f 'vllm serve .*--port {cfg.get('TRACK_A_PORT_B')}' || true; "
        + "pkill -f VLLM::EngineCore || true",
        check=False,
    )
    time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))
