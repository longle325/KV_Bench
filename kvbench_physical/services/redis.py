"""Redis lifecycle helpers for Track A shared KV-cache experiments."""

from __future__ import annotations

import time
from pathlib import Path

from ..config import Config, utc_stamp
from ..shell import log, require_tools, run


def start_redis_a(cfg: Config) -> None:
    require_tools(["redis-server", "redis-cli", "ss"], "redis")

    run_stamp = cfg.get("RUN_STAMP", utc_stamp(""))
    log_dir = cfg.paths.node_root / "logs" / run_stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    port = cfg.get("REDIS_PORT")
    conf = log_dir / f"redis_{port}.conf"
    log_file = log_dir / f"redis_{port}.log"

    ss = run(["ss", "-ltn"], capture=True, check=False)
    if f":{port} " in (ss.stdout or ""):
        log("redis", f"port {port} already listening; checking {cfg.get('A_IP')}:{port}")
        ping = run(
            ["redis-cli", "-h", cfg.get("A_IP"), "-p", port, "PING"],
            check=False,
            capture=True,
        )
        if ping.returncode == 0:
            print((ping.stdout or "").strip())
            return

        log("redis", f"existing Redis is not reachable on {cfg.get('A_IP')}:{port}; restarting")
        run(
            ["redis-cli", "-h", cfg.get("REDIS_BIND_LOCAL"), "-p", port, "SHUTDOWN", "NOSAVE"],
            check=False,
        )
        time.sleep(cfg.float("REDIS_RESTART_SLEEP_SECONDS"))

    conf.write_text(
        "\n".join(
            [
                f"bind {cfg.get('REDIS_BIND_LOCAL')} {cfg.get('A_IP')}",
                f"port {port}",
                "protected-mode no",
                "daemonize yes",
                f"dir {log_dir}",
                f"logfile {log_file}",
                'save ""',
                "appendonly no",
                "",
            ]
        )
    )
    run(["redis-server", str(conf)])
    time.sleep(cfg.float("REDIS_STARTUP_SLEEP_SECONDS"))
    run(["redis-cli", "-h", cfg.get("A_IP"), "-p", port, "PING"])
    log("redis", f"started {cfg.get('A_IP')}:{port}; log={log_file}")


def flush_redis(cfg: Config) -> None:
    run(["redis-cli", "-h", cfg.get("A_IP"), "-p", cfg.get("REDIS_PORT"), "FLUSHALL"], check=False)
