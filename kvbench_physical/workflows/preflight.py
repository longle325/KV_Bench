"""Preflight checks for SSH, routes, direct TCP reachability, and bandwidth."""

from __future__ import annotations

import socket
import subprocess
import time

from ..config import Config
from ..remote import Remote, remote_python_command
from ..shell import log, q, run


SERVER_CODE = r"""
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
reply = sys.argv[3].encode()
timeout_seconds = float(sys.argv[4])
recv_bytes = int(sys.argv[5])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host, port))
s.listen(1)
s.settimeout(timeout_seconds)
conn, addr = s.accept()
data = conn.recv(recv_bytes)
conn.sendall(reply)
print("accepted", addr, data.decode(errors="ignore"), flush=True)
conn.close()
s.close()
"""


def preflight_network(cfg: Config) -> None:
    remote = Remote(cfg)
    port_a = cfg.get("PORT_A", cfg.get("PREFLIGHT_PORT_A"))
    port_b = cfg.get("PORT_B", cfg.get("PREFLIGHT_PORT_B"))

    log("preflight", f"A={socket.gethostname()} {cfg.get('A_IP')}")
    log("preflight", f"B={cfg.remote_login} ssh_port={cfg.get('SSH_PORT')}")

    log("preflight", "route A -> B")
    run(["ip", "route", "get", cfg.get("B_IP")])
    run(["ping", "-c", cfg.get("PREFLIGHT_PING_COUNT"), "-W", cfg.get("PREFLIGHT_PING_WAIT_SECONDS"), cfg.get("B_IP")])

    log("preflight", "route B -> A")
    remote.ssh(
        f"hostname; ip route get {q(cfg.get('A_IP'))}; "
        f"ping -c {q(cfg.get('PREFLIGHT_PING_COUNT'))} -W {q(cfg.get('PREFLIGHT_PING_WAIT_SECONDS'))} {q(cfg.get('A_IP'))}"
    )

    log("preflight", f"TCP A -> B port {port_b}")
    remote.ssh(
        "python3 -c "
        + q(SERVER_CODE)
        + f" {q(cfg.get('PREFLIGHT_BIND_HOST'))} {q(port_b)} ok-from-b "
        + f"{q(cfg.get('PREFLIGHT_SERVER_TIMEOUT_SECONDS'))} {q(cfg.get('PREFLIGHT_RECV_BYTES'))} "
        + ">/tmp/kvbench_port_b.log 2>&1 &"
    )
    time.sleep(cfg.float("PREFLIGHT_STARTUP_SLEEP_SECONDS"))
    _tcp_client(cfg.get("B_IP"), int(port_b), "hello-from-a", cfg)
    remote.ssh("cat /tmp/kvbench_port_b.log")

    log("preflight", f"TCP B -> A port {port_a}")
    local_server = subprocess.Popen(
        [
            "python3",
            "-c",
            SERVER_CODE,
            cfg.get("PREFLIGHT_BIND_HOST"),
            port_a,
            "ok-from-a",
            cfg.get("PREFLIGHT_SERVER_TIMEOUT_SECONDS"),
            cfg.get("PREFLIGHT_RECV_BYTES"),
        ],
        stdout=open("/tmp/kvbench_port_a.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(cfg.float("PREFLIGHT_STARTUP_SLEEP_SECONDS"))
        remote.ssh(
            "python3 - <<'PY'\n"
            + _tcp_client_code(cfg.get("A_IP"), int(port_a), "hello-from-b", cfg)
            + "\nPY"
        )
        local_server.wait(timeout=cfg.float("PREFLIGHT_SERVER_TIMEOUT_SECONDS") + 1)
        print(open("/tmp/kvbench_port_a.log").read(), end="")
    finally:
        if local_server.poll() is None:
            local_server.terminate()

    log("preflight", "local env A")
    from .env_check import remote_env_check

    remote_env_check()

    log("preflight", "remote env B")
    remote.ssh(remote_python_command(cfg, "remote-env-check"))

    _optional_iperf(cfg, remote)


def _tcp_client(host: str, port: int, payload: str, cfg: Config) -> None:
    with socket.create_connection((host, port), timeout=cfg.float("PREFLIGHT_CONNECT_TIMEOUT_SECONDS")) as sock:
        sock.sendall(payload.encode())
        print(sock.recv(cfg.int("PREFLIGHT_RECV_BYTES")).decode(errors="ignore"))


def _tcp_client_code(host: str, port: int, payload: str, cfg: Config) -> str:
    return f"""
import socket
with socket.create_connection(({host!r}, {port}), timeout={cfg.float('PREFLIGHT_CONNECT_TIMEOUT_SECONDS')}) as sock:
    sock.sendall({payload!r}.encode())
    print(sock.recv({cfg.int('PREFLIGHT_RECV_BYTES')}).decode(errors='ignore'))
"""


def _optional_iperf(cfg: Config, remote: Remote) -> None:
    local = run(["bash", "-lc", "command -v iperf3 >/dev/null"], check=False)
    remote_ok = remote.ssh("command -v iperf3 >/dev/null", check=False)
    if local.returncode != 0 or remote_ok.returncode != 0:
        log("preflight", "iperf3 unavailable on at least one host; skipped")
        return

    log("preflight", "optional iperf3 A -> B")
    remote.ssh("iperf3 -s -1 >/tmp/kvbench_iperf_b.log 2>&1 &")
    time.sleep(cfg.float("PREFLIGHT_STARTUP_SLEEP_SECONDS"))
    run(["iperf3", "-c", cfg.get("B_IP"), "-P", cfg.get("IPERF_PARALLEL_STREAMS"), "-t", cfg.get("IPERF_SECONDS")], check=False)
    remote.ssh("cat /tmp/kvbench_iperf_b.log || true")

    log("preflight", "optional iperf3 B -> A")
    subprocess.Popen(["iperf3", "-s", "-1"], stdout=open("/tmp/kvbench_iperf_a.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(cfg.float("PREFLIGHT_STARTUP_SLEEP_SECONDS"))
    remote.ssh(
        f"iperf3 -c {q(cfg.get('A_IP'))} -P {q(cfg.get('IPERF_PARALLEL_STREAMS'))} -t {q(cfg.get('IPERF_SECONDS'))}",
        check=False,
    )
    print(open("/tmp/kvbench_iperf_a.log").read(), end="")
