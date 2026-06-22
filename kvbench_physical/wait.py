"""Readiness probes for TCP sockets and HTTP endpoints."""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path

from .config import Config
from .shell import die


def wait_tcp(cfg: Config, host: str, port: int, label: str, timeout_seconds: int | None = None) -> None:
    timeout = timeout_seconds or cfg.int("WAIT_TCP_TIMEOUT_SECONDS")
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=3):
                return
        except OSError:
            if time.monotonic() >= deadline:
                die("wait", f"timeout waiting for {label} at {host}:{port}")
            time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))


def wait_http(
    cfg: Config,
    url: str,
    label: str,
    timeout_seconds: int | None = None,
    log_file: Path | None = None,
) -> None:
    timeout = timeout_seconds or cfg.int("WAIT_HTTP_TIMEOUT_SECONDS")
    deadline = time.monotonic() + timeout
    last_error = ""
    while True:
        try:
            with urllib.request.urlopen(url, timeout=cfg.float("CURL_MAX_TIME_SECONDS")) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - error text is useful here
            last_error = str(exc)

        if time.monotonic() >= deadline:
            if log_file and log_file.exists():
                lines = log_file.read_text(errors="replace").splitlines()
                for line in lines[-cfg.int("LOG_TAIL_LINES"):]:
                    print(line)
            die("wait", f"timeout waiting for {label}: {url}; last_error={last_error}")
        time.sleep(cfg.float("POLL_INTERVAL_SECONDS"))
