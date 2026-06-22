"""Host inventory reporting for debugging runtime and network mismatches."""

from __future__ import annotations

import importlib.metadata as metadata
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

from ..shell import run


def remote_env_check() -> None:
    print(f"[host] {socket.gethostname()}")
    print(f"[date] {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

    print("[network]")
    run(["ip", "-brief", "addr"], check=False)
    run(["ip", "route"], check=False)

    print("[tools]")
    for tool in ("python3", "python", "conda", "vllm", "ray", "redis-cli", "redis-server", "iperf3", "nvidia-smi", "screen"):
        path = shutil.which(tool)
        print(f"{tool}={path if path else 'missing'}")

    print("[gpu]")
    run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=False,
    )

    print("[python-packages]")
    for pkg in ("vllm", "lmcache", "ray", "torch", "transformers", "redis", "requests"):
        try:
            print(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"{pkg}=missing")

    print("[interfaces]")
    for path in sorted(Path("/sys/class/net").glob("*")):
        speed = _read_text(path / "speed", "unknown")
        carrier = _read_text(path / "carrier", "unknown")
        print(f"{path.name} speed_mbps={speed} carrier={carrier}")

    print("[rdma]")
    run(["ls", "-l", "/sys/class/infiniband"], check=False)
    if shutil.which("rdma"):
        run(["rdma", "link"], check=False)


def _read_text(path: Path, default: str) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default
