"""Network byte-rate monitor used to validate the active benchmark path."""

from __future__ import annotations

import time
from pathlib import Path

from ..config import Config


def monitor_net_bytes(cfg: Config) -> None:
    iface = cfg.get("IFACE", "")
    seconds = cfg.float("SECONDS", cfg.float("NET_MONITOR_SECONDS"))
    if not iface:
        raise SystemExit(f"Usage: IFACE=<interface> SECONDS={cfg.get('NET_MONITOR_SECONDS')} monitor-net-bytes")

    base = Path("/sys/class/net") / iface / "statistics"
    rx1 = int((base / "rx_bytes").read_text())
    tx1 = int((base / "tx_bytes").read_text())
    time.sleep(seconds)
    rx2 = int((base / "rx_bytes").read_text())
    tx2 = int((base / "tx_bytes").read_text())

    print(f"RX_Gbps={(rx2 - rx1) * 8 / seconds / 1_000_000_000:.6f}")
    print(f"TX_Gbps={(tx2 - tx1) * 8 / seconds / 1_000_000_000:.6f}")
