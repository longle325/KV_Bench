"""Track B orchestration for Ray-backed distributed vLLM comparisons.

The workflow compares PP=2, TP=2, and single-machine baseline cases while
keeping Machine B as a Ray worker instead of a public API server.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..config import Config, utc_stamp
from ..remote import Remote, remote_python_command
from ..services.ray import stop_ray, wait_for_two_nodes
from ..services.runtime import bootstrap_runtime, check_runtime
from ..shell import die, log, python_module_command, q, run, screen_exists, start_screen, stop_screens_matching
from ..wait import wait_tcp
from .benchmark import summarize_result, write_summary_rows


@dataclass(frozen=True)
class TrackBCase:
    name: str
    config_name: str
    tp: int
    pp: int
    distributed: bool


def run_full_cluster(cfg: Config) -> None:
    run_stamp = cfg.get("RUN_STAMP", utc_stamp("phys_track_b_full"))
    cases = _track_b_cases(cfg)
    requests = cfg.get("REQUESTS", cfg.get("TRACK_B_REQUESTS"))
    prefix_tokens = cfg.get("PREFIX_TOKENS", cfg.get("TRACK_B_PREFIX_TOKENS"))
    max_tokens = cfg.get("MAX_TOKENS", cfg.get("TRACK_B_MAX_TOKENS"))

    log_dir = cfg.paths.node_root / "logs" / run_stamp
    result_dir = cfg.paths.node_root / "results" / run_stamp
    remote_log_dir = f"{cfg.get('REMOTE_ROOT')}/logs/{run_stamp}"
    summary_file = cfg.paths.node_root / "results" / f"summary_{run_stamp}.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    local_cfg = Config(cfg.merged_env(RUN_STAMP=run_stamp), cfg.paths)
    _ensure_local_runtime(local_cfg)

    has_distributed_case = any(case.distributed for case in cases)
    log("track-b", f"run_id={run_stamp}")
    log("track-b", f"A={cfg.get('A_IP')}/{cfg.get('A_IFACE')} B={cfg.get('B_IP')}/{cfg.get('B_IFACE')} ssh_port={cfg.get('SSH_PORT')}")
    log("track-b", "cases=" + ",".join(case.name for case in cases))
    log("track-b", "B has no public vLLM API in distributed Track B; it joins through Ray workers.")

    if cfg.bool("STOP_EXISTING", True):
        _stop_track_b(local_cfg, stop_remote=has_distributed_case)

    remote: Remote | None = None
    if has_distributed_case:
        remote = Remote(local_cfg)
        log("track-b", "syncing runnable bundle to B")
        remote.sync_bundle()
        _ensure_remote_runtime(local_cfg, remote)
        remote.ssh(f"mkdir -p {q(remote_log_dir)} {q(cfg.get('REMOTE_ROOT') + '/results/' + run_stamp)}")
        _start_ray_cluster(local_cfg, remote, run_stamp, log_dir)

    cleanup_after_run = cfg.bool("CLEANUP_AFTER_RUN", True) and cfg.bool("RUN_BENCH", True)
    summary_rows: list[dict[str, object]] = []
    try:
        for index, case in enumerate(cases, start=1):
            log("track-b", f"case {index}/{len(cases)}: {case.name} TP={case.tp} PP={case.pp}")
            _stop_case_processes(local_cfg, remote)

            case_cfg = Config(
                local_cfg.merged_env(
                    CONFIG_NAME=case.config_name,
                    TP=case.tp,
                    PP=case.pp,
                    RUN_STAMP=run_stamp,
                    REQUESTS=requests,
                    PREFIX_TOKENS=prefix_tokens,
                    MAX_TOKENS=max_tokens,
                ),
                cfg.paths,
            )
            server_log = log_dir / f"{case.config_name}_server.log"
            bench_log = _bench_log(case_cfg)
            result_file = _result_file(case_cfg)

            if case.distributed:
                server_screen = _start_distributed_case(case_cfg, case, run_stamp, server_log)
                if remote:
                    remote.ssh(
                        'pgrep -af "RayWorkerProc|raylet|VLLM::EngineCore" || true; '
                        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
                        check=False,
                    )
            else:
                server_screen = _start_single_a_case(case_cfg, case, run_stamp, server_log)

            _wait_server_health(case_cfg, f"http://{cfg.get('A_IP')}:{cfg.get('PORT')}/health", server_screen, server_log)
            log("track-b", f"{case.name} vLLM health OK")

            if not cfg.bool("RUN_BENCH", True):
                log("track-b", "RUN_BENCH=0; current server is left running for manual requests.")
                break

            log("track-b", f"starting benchmark: mph_bench_{case.name}_{run_stamp}")
            start_screen(
                f"mph_bench_{case.name}_{run_stamp}",
                python_module_command(
                    case_cfg,
                    "run-track-b-case",
                    WAIT_HEALTH=case_cfg.get("BENCH_WAIT_HEALTH"),
                    HEALTH_TIMEOUT_SECONDS=case_cfg.get("BENCH_HEALTH_TIMEOUT_SECONDS"),
                ),
            )

            if cfg.bool("WAIT_BENCH", True):
                _wait_benchmark(case_cfg, bench_log)
                summary_rows.append(
                    summarize_result(
                        result_file,
                        {
                            "case": case.name,
                            "config_name": case.config_name,
                            "distributed": int(case.distributed),
                            "tp": case.tp,
                            "pp": case.pp,
                        },
                    )
                )
                write_summary_rows(summary_rows, summary_file)

        log("track-b", f"summary={summary_file}")
        log("track-b", f"local_logs={log_dir}")
        if has_distributed_case:
            log("track-b", f"remote_logs={remote_log_dir}")
    finally:
        if cleanup_after_run:
            log("track-b", "cleaning up Track B services")
            _stop_track_b(local_cfg, stop_remote=has_distributed_case)


def _track_b_cases(cfg: Config) -> list[TrackBCase]:
    aliases = {
        "pipeline": "pp2",
        "pipeline_parallel": "pp2",
        "pp": "pp2",
        "tensor": "tp2",
        "tensor_parallel": "tp2",
        "tp": "tp2",
        "single": "single_a",
        "baseline": "single_a",
    }
    cases: list[TrackBCase] = []
    for raw_name in cfg.get("TRACK_B_CASES").split(","):
        name = aliases.get(raw_name.strip().lower(), raw_name.strip().lower())
        if not name:
            continue
        if name == "pp2":
            cases.append(TrackBCase(name, f"pp2_{cfg.get('A_IFACE')}_{cfg.get('B_IFACE')}", tp=1, pp=2, distributed=True))
        elif name == "tp2":
            cases.append(TrackBCase(name, f"tp2_{cfg.get('A_IFACE')}_{cfg.get('B_IFACE')}", tp=2, pp=1, distributed=True))
        elif name == "single_a":
            cases.append(TrackBCase(name, f"single_a_{cfg.get('A_IFACE')}", tp=1, pp=1, distributed=False))
        else:
            die("track-b", f"unknown TRACK_B_CASES entry {raw_name!r}; use pp2,tp2,single_a")
    if not cases:
        die("track-b", "TRACK_B_CASES is empty; use pp2,tp2,single_a")
    return cases


def _start_ray_cluster(cfg: Config, remote: Remote, run_stamp: str, log_dir: Path) -> None:
    log("track-b", f"starting Ray head on A: mph_ray_head_{run_stamp}")
    start_screen(
        f"mph_ray_head_{run_stamp}",
        python_module_command(
            cfg,
            "start-ray-head-a",
            RUN_STAMP=run_stamp,
            RAY_BLOCK=cfg.get("RAY_HEAD_BLOCK"),
        )
        + f" 2>&1 | tee {q(log_dir / 'ray_head.log')}",
    )
    wait_tcp(cfg, cfg.get("A_IP"), cfg.int("GCS_PORT"), "Ray GCS")

    if cfg.bool("START_B_RAY", True):
        log("track-b", f"starting Ray worker on B: mph_ray_worker_{run_stamp}")
        remote.ssh(
            "screen -dmS "
            f"{'mph_ray_worker_' + run_stamp!r} bash -lc "
            + repr(
                remote_python_command(
                    cfg,
                    "start-ray-worker-b",
                    RUN_STAMP=run_stamp,
                    RAY_BLOCK=cfg.get("RAY_WORKER_BLOCK"),
                )
                + f" 2>&1 | tee logs/{run_stamp}/ray_worker.log"
            )
        )
    wait_for_two_nodes(cfg)


def _start_distributed_case(cfg: Config, case: TrackBCase, run_stamp: str, server_log: Path) -> str:
    screen_name = f"mph_dist_{case.name}_{run_stamp}"
    log("track-b", f"starting distributed vLLM: {screen_name}")
    start_screen(
        screen_name,
        python_module_command(
            cfg,
            "start-distributed-vllm-a",
            RUN_STAMP=run_stamp,
            RAY_ADDRESS=f"{cfg.get('A_IP')}:{cfg.get('GCS_PORT')}",
            VLLM_HOST_IP=cfg.get("A_IP"),
            GLOO_SOCKET_IFNAME=cfg.get("A_IFACE"),
            NCCL_SOCKET_IFNAME=cfg.get("A_IFACE"),
            TP=case.tp,
            PP=case.pp,
            MAX_MODEL_LEN=cfg.get("MAX_MODEL_LEN", cfg.get("TRACK_B_MAX_MODEL_LEN")),
            GPU_MEMORY_UTILIZATION=cfg.get("GPU_MEMORY_UTILIZATION", cfg.get("VLLM_GPU_MEMORY_UTILIZATION")),
        )
        + f" 2>&1 | tee {q(server_log)}",
    )
    return screen_name


def _start_single_a_case(cfg: Config, case: TrackBCase, run_stamp: str, server_log: Path) -> str:
    screen_name = f"mph_single_a_{run_stamp}"
    log("track-b", f"starting single-A baseline vLLM: {screen_name}")
    start_screen(
        screen_name,
        python_module_command(
            cfg,
            "start-vllm-replica",
            NODE="A",
            MODE="no_cache",
            GPU=cfg.get("TRACK_B_SINGLE_GPU"),
            RUN_STAMP=run_stamp,
            LOG_DIR=str(server_log.parent),
            PORT=cfg.get("PORT"),
            MAX_MODEL_LEN=cfg.get("MAX_MODEL_LEN", cfg.get("TRACK_B_MAX_MODEL_LEN")),
            TP=case.tp,
        )
        + f" 2>&1 | tee {q(server_log)}",
    )
    return screen_name


def _wait_server_health(cfg: Config, url: str, screen_name: str, server_log: Path) -> None:
    deadline = time.monotonic() + cfg.int("TRACK_B_HEALTH_TIMEOUT_SECONDS")
    while True:
        try:
            with urllib.request.urlopen(url, timeout=cfg.float("CURL_MAX_TIME_SECONDS")) as response:
                if response.status < 500:
                    return
        except Exception:
            pass

        if not screen_exists(screen_name):
            if server_log.exists():
                for line in server_log.read_text(errors="replace").splitlines()[-cfg.int("LOG_TAIL_LINES"):]:
                    print(line)
            die("track-b", f"vLLM server screen exited before health OK: {screen_name}")

        if time.monotonic() >= deadline:
            if server_log.exists():
                for line in server_log.read_text(errors="replace").splitlines()[-cfg.int("LOG_TAIL_LINES"):]:
                    print(line)
            die("track-b", f"timeout waiting for vLLM health: {url}")

        time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))


def _result_file(cfg: Config) -> Path:
    config_name = cfg.get("CONFIG_NAME")
    requests = cfg.get("REQUESTS", cfg.get("TRACK_B_REQUESTS"))
    prefix_tokens = cfg.get("PREFIX_TOKENS", cfg.get("TRACK_B_PREFIX_TOKENS"))
    max_tokens = cfg.get("MAX_TOKENS", cfg.get("TRACK_B_MAX_TOKENS"))
    return cfg.paths.node_root / "results" / cfg.get("RUN_STAMP") / f"distributed_{config_name}_input{prefix_tokens}_out{max_tokens}_n{requests}.jsonl"


def _bench_log(cfg: Config) -> Path:
    config_name = cfg.get("CONFIG_NAME")
    requests = cfg.get("REQUESTS", cfg.get("TRACK_B_REQUESTS"))
    prefix_tokens = cfg.get("PREFIX_TOKENS", cfg.get("TRACK_B_PREFIX_TOKENS"))
    max_tokens = cfg.get("MAX_TOKENS", cfg.get("TRACK_B_MAX_TOKENS"))
    return cfg.paths.node_root / "logs" / cfg.get("RUN_STAMP") / f"distributed_{config_name}_input{prefix_tokens}_out{max_tokens}_n{requests}.log"


def _ensure_local_runtime(cfg: Config) -> None:
    try:
        check_runtime(cfg, ["python", "ray", "vllm"], ["ray", "vllm", "torch"])
        run(["screen", "-v"], capture=True)
        run(["ssh", "-V"], capture=True, check=False)
    except Exception as exc:  # noqa: BLE001
        if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
            die("runtime", f"local runtime missing ({exc}). Re-run with ALLOW_RUNTIME_INSTALL=1.")
        bootstrap_runtime(Config(cfg.merged_env(PACKAGE_SET="distributed", ALLOW_RUNTIME_INSTALL=1), cfg.paths))


def _ensure_remote_runtime(cfg: Config, remote: Remote) -> None:
    command = remote_python_command(cfg, "check-runtime --tools python ray vllm --imports ray vllm torch")
    proc = remote.ssh(command, check=False, capture=True)
    if proc.returncode == 0:
        return
    if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
        die("runtime", f"remote runtime missing on B. Re-run with ALLOW_RUNTIME_INSTALL=1. Output:\n{proc.stdout}")
    remote.ssh(remote_python_command(cfg, "bootstrap-runtime", PACKAGE_SET="distributed", ALLOW_RUNTIME_INSTALL=1))


def _stop_track_b(cfg: Config, *, stop_remote: bool) -> None:
    log("track-b", "stopping old Track B processes; screen sessions are kept")
    stop_screens_matching("mph_(ray_head|ray_worker|dist_|single_|bench_)")
    _stop_case_processes(cfg, None)
    stop_ray(cfg)

    if stop_remote and cfg.bool("START_B_RAY", True):
        Remote(cfg).ssh(
            remote_python_command(cfg, "stop-ray")
            + "; pkill -f VLLM::EngineCore || true",
            check=False,
        )


def _stop_case_processes(cfg: Config, remote: Remote | None) -> None:
    run(["pkill", "-f", f"vllm serve .*--port {cfg.get('PORT')}"], check=False)
    run(["pkill", "-f", "VLLM::EngineCore"], check=False)
    run(["pkill", "-f", "llm_benchmark_multi.py"], check=False)
    if remote:
        remote.ssh("pkill -f VLLM::EngineCore || true", check=False)
    time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))


def _wait_benchmark(cfg: Config, bench_log: Path) -> None:
    request_timeout = int(cfg.get("REQUEST_TIMEOUT", cfg.get("TRACK_B_REQUEST_TIMEOUT")))
    deadline = time.monotonic() + request_timeout + cfg.int("BENCH_EXTRA_TIMEOUT_SECONDS")
    tail_lines = cfg.int("BENCH_WAIT_TAIL_LINES")
    while True:
        if bench_log.exists() and '"successes"' in "\n".join(bench_log.read_text(errors="replace").splitlines()[-tail_lines:]):
            for line in bench_log.read_text(errors="replace").splitlines()[-cfg.int("BENCH_LOG_TAIL_LINES"):]:
                print(line)
            return
        if time.monotonic() >= deadline:
            die("track-b", f"benchmark summary did not appear in {bench_log}")
        time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))
