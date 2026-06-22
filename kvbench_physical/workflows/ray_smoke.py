"""Ray, GPU, and NCCL smoke tests for the physical-node cluster.

These tests validate the runtime layer below vLLM: Ray scheduling can place
work on both machines, CUDA matmul works on each GPU, and torch.distributed can
run a cross-node NCCL collective through the Ray workers.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

from ..config import Config, utc_stamp
from ..remote import Remote, remote_python_command
from ..services.ray import stop_ray, wait_for_two_nodes
from ..services.runtime import bootstrap_runtime, check_runtime
from ..shell import die, log, python_module_command, q, run, run_runtime, start_screen, stop_screens_matching
from ..wait import wait_tcp


def run_ray_smoke_tests(cfg: Config) -> None:
    """Start a two-node Ray cluster and run GPU/NCCL sanity checks."""

    run_stamp = cfg.get("RUN_STAMP", utc_stamp("phys_ray_smoke"))
    local_cfg = Config(cfg.merged_env(RUN_STAMP=run_stamp), cfg.paths)
    log_dir = cfg.paths.node_root / "logs" / run_stamp
    result_dir = cfg.paths.node_root / "results" / run_stamp
    result_file = result_dir / "ray_smoke.json"
    report_file = cfg.paths.node_root / "reports" / f"ray_smoke_{run_stamp}.md"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    _ensure_local_runtime(local_cfg)
    remote = Remote(local_cfg)
    remote.sync_bundle()
    _ensure_remote_runtime(local_cfg, remote)
    remote.ssh(f"mkdir -p {q(local_cfg.get('REMOTE_ROOT'))}/logs/{q(run_stamp)}")

    if cfg.bool("RAY_SMOKE_STOP_EXISTING", True):
        _stop_existing(local_cfg, remote)

    _start_ray_cluster(local_cfg, remote, run_stamp, log_dir)
    _run_smoke_script(local_cfg, result_file)

    report = json.loads(result_file.read_text())
    _write_markdown_report(report, report_file)
    log("ray-smoke", f"json={result_file}")
    log("ray-smoke", f"report={report_file}")
    if report.get("status") != "pass":
        die("ray-smoke", f"smoke test failed; see {result_file}")
    log("ray-smoke", "PASS")


def _ensure_local_runtime(cfg: Config) -> None:
    try:
        check_runtime(cfg, ["python", "ray", "nvidia-smi"], ["ray", "torch"])
        run(["screen", "-v"], capture=True)
        run(["ssh", "-V"], capture=True, check=False)
    except Exception as exc:  # noqa: BLE001
        if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
            die("runtime", f"local Ray smoke runtime missing ({exc}). Re-run with ALLOW_RUNTIME_INSTALL=1.")
        bootstrap_runtime(Config(cfg.merged_env(PACKAGE_SET="distributed", ALLOW_RUNTIME_INSTALL=1), cfg.paths))


def _ensure_remote_runtime(cfg: Config, remote: Remote) -> None:
    command = remote_python_command(cfg, "check-runtime --tools python ray nvidia-smi --imports ray torch")
    proc = remote.ssh(command, check=False, capture=True)
    if proc.returncode == 0:
        return
    if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
        die("runtime", f"remote Ray smoke runtime missing on B. Re-run with ALLOW_RUNTIME_INSTALL=1. Output:\n{proc.stdout}")
    remote.ssh(remote_python_command(cfg, "bootstrap-runtime", PACKAGE_SET="distributed", ALLOW_RUNTIME_INSTALL=1))


def _stop_existing(cfg: Config, remote: Remote) -> None:
    log("ray-smoke", "stopping old Ray/vLLM processes; screen sessions are kept")
    stop_screens_matching("mph_(ray_smoke|ray_head|ray_worker|dist_|single_|bench_)")
    stop_ray(cfg)
    run(["pkill", "-f", f"vllm serve .*--port {cfg.get('PORT')}"], check=False)
    run(["pkill", "-f", "VLLM::EngineCore"], check=False)
    remote.ssh(
        remote_python_command(cfg, "stop-ray")
        + "; pkill -f 'vllm serve' || true; pkill -f VLLM::EngineCore || true",
        check=False,
    )


def _start_ray_cluster(cfg: Config, remote: Remote, run_stamp: str, log_dir: Path) -> None:
    log("ray-smoke", f"starting Ray head on A: mph_ray_smoke_head_{run_stamp}")
    start_screen(
        f"mph_ray_smoke_head_{run_stamp}",
        python_module_command(
            cfg,
            "start-ray-head-a",
            RUN_STAMP=run_stamp,
            RAY_BLOCK=cfg.get("RAY_HEAD_BLOCK"),
        )
        + f" 2>&1 | tee {q(log_dir / 'ray_head.log')}",
    )
    wait_tcp(cfg, cfg.get("A_IP"), cfg.int("GCS_PORT"), "Ray GCS")

    log("ray-smoke", f"starting Ray worker on B: mph_ray_smoke_worker_{run_stamp}")
    remote.ssh(
        "screen -dmS "
        f"{'mph_ray_smoke_worker_' + run_stamp!r} bash -lc "
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


def _run_smoke_script(cfg: Config, result_file: Path) -> None:
    port = _free_port()
    script = r"""
python - <<'PY'
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def node_summary(node):
    return {
        "node_id": node.get("NodeID"),
        "ip": node.get("NodeManagerAddress"),
        "alive": node.get("Alive"),
        "gpu": node.get("Resources", {}).get("GPU", 0),
        "resources": node.get("Resources", {}),
    }


@ray.remote
def gpu_inventory():
    proc = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "returncode": proc.returncode,
        "nvidia_smi": proc.stdout,
    }


@ray.remote(num_gpus=1)
def matmul_smoke(size, repeats, dtype_name):
    import ray
    import torch

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    a = torch.randn((size, size), device="cuda", dtype=dtype)
    b = torch.randn((size, size), device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checksum = float(c[0, 0].detach().float().cpu())
    tflops = (2 * (size ** 3) * repeats) / elapsed / 1e12
    memory_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    torch.cuda.empty_cache()
    return {
        "status": "ok",
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ray_gpu_ids": ray.get_gpu_ids(),
        "device": device_name,
        "dtype": dtype_name,
        "size": size,
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 4),
        "tflops": round(tflops, 4),
        "max_memory_mb": memory_mb,
        "checksum": checksum,
    }


@ray.remote(num_gpus=1)
class NcclRank:
    def run(self, rank, world_size, init_method, elements, dtype_name, timeout_seconds, iface, nccl_ib_disable):
        import datetime as dt
        import ray
        import torch
        import torch.distributed as dist

        os.environ["NCCL_SOCKET_IFNAME"] = iface
        os.environ["GLOO_SOCKET_IFNAME"] = iface
        os.environ["NCCL_IB_DISABLE"] = nccl_ib_disable
        os.environ.setdefault("NCCL_DEBUG", "WARN")

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=dt.timedelta(seconds=timeout_seconds),
        )
        tensor = torch.ones(elements, device="cuda", dtype=dtype) * (rank + 1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        sample = float(tensor[0].detach().float().cpu())
        expected = float(world_size * (world_size + 1) / 2)
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        return {
            "status": "ok" if sample == expected else "bad_value",
            "hostname": socket.gethostname(),
            "rank": rank,
            "world_size": world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "device": torch.cuda.get_device_name(0),
            "dtype": dtype_name,
            "elements": elements,
            "bytes": elements * torch.tensor([], dtype=dtype).element_size(),
            "elapsed_seconds": round(elapsed, 4),
            "sample": sample,
            "expected": expected,
        }


report = {
    "run_stamp": os.environ["RUN_STAMP"],
    "status": "fail",
    "address": f"{os.environ['A_IP']}:{os.environ['GCS_PORT']}",
    "config": {
        "a_ip": os.environ["A_IP"],
        "b_ip": os.environ["B_IP"],
        "a_iface": os.environ["A_IFACE"],
        "b_iface": os.environ["B_IFACE"],
        "matrix_size": int(os.environ["RAY_SMOKE_MATRIX_SIZE"]),
        "matrix_repeats": int(os.environ["RAY_SMOKE_MATRIX_REPEATS"]),
        "dtype": os.environ["RAY_SMOKE_DTYPE"],
        "nccl_elements": int(os.environ["RAY_SMOKE_NCCL_ELEMENTS"]),
        "nccl_init_method": os.environ["RAY_SMOKE_NCCL_INIT_METHOD"],
    },
}

try:
    ray.init(address=report["address"], ignore_reinit_error=True, logging_level="ERROR")
    alive = [node for node in ray.nodes() if node.get("Alive")]
    gpu_nodes = [node for node in alive if float(node.get("Resources", {}).get("GPU", 0)) > 0]
    gpu_nodes.sort(key=lambda node: (node.get("NodeManagerAddress") != os.environ["A_IP"], node.get("NodeManagerAddress", "")))
    report["alive_nodes"] = [node_summary(node) for node in alive]
    report["gpu_nodes"] = [node_summary(node) for node in gpu_nodes]

    inventory_refs = [
        gpu_inventory.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote()
        for node in alive
    ]
    report["gpu_inventory"] = ray.get(inventory_refs)

    size = int(os.environ["RAY_SMOKE_MATRIX_SIZE"])
    repeats = int(os.environ["RAY_SMOKE_MATRIX_REPEATS"])
    dtype_name = os.environ["RAY_SMOKE_DTYPE"]
    matmul_refs = [
        matmul_smoke.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote(size, repeats, dtype_name)
        for node in gpu_nodes
    ]
    report["matmul"] = ray.get(matmul_refs)

    report["nccl"] = {"status": "skipped"}
    if os.environ["RAY_SMOKE_RUN_NCCL"] == "1":
        if len(gpu_nodes) < 2:
            report["nccl"] = {"status": "failed", "error": "need at least two alive GPU Ray nodes"}
        else:
            world_nodes = gpu_nodes[:2]
            actors = [
                NcclRank.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote()
                for node in world_nodes
            ]
            timeout = int(os.environ["RAY_SMOKE_NCCL_TIMEOUT_SECONDS"])
            refs = [
                actor.run.remote(
                    rank,
                    len(actors),
                    os.environ["RAY_SMOKE_NCCL_INIT_METHOD"],
                    int(os.environ["RAY_SMOKE_NCCL_ELEMENTS"]),
                    dtype_name,
                    timeout,
                    os.environ["A_IFACE"] if rank == 0 else os.environ["B_IFACE"],
                    os.environ["NCCL_IB_DISABLE"],
                )
                for rank, actor in enumerate(actors)
            ]
            report["nccl"] = {
                "status": "ok",
                "ranks": ray.get(refs, timeout=timeout + 60),
            }

    inventory_ok = bool(report["gpu_inventory"]) and all(row["returncode"] == 0 for row in report["gpu_inventory"])
    matmul_ok = len(report["matmul"]) == len(gpu_nodes) and all(row["status"] == "ok" for row in report["matmul"])
    nccl_ok = report["nccl"].get("status") in {"ok", "skipped"}
    enough_nodes = len(alive) >= 2 and len(gpu_nodes) >= 2
    report["status"] = "pass" if inventory_ok and matmul_ok and nccl_ok and enough_nodes else "fail"
except Exception as exc:  # noqa: BLE001
    report["status"] = "fail"
    report["error"] = str(exc)
    report["traceback"] = traceback.format_exc()
finally:
    Path(os.environ["RAY_SMOKE_RESULT_FILE"]).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"status": report["status"], "result": os.environ["RAY_SMOKE_RESULT_FILE"]}, sort_keys=True))
PY
"""
    env = {
        "RAY_SMOKE_RESULT_FILE": str(result_file),
        "RAY_SMOKE_NCCL_INIT_METHOD": f"tcp://{cfg.get('A_IP')}:{port}",
    }
    run_runtime(cfg, script, tools=["python", "ray"], env=env)


def _write_markdown_report(report: dict, report_file: Path) -> None:
    lines = [
        "# Ray/NCCL Smoke Test",
        "",
        f"Run ID: `{report.get('run_stamp', '')}`",
        f"Status: **{report.get('status', 'unknown')}**",
        "",
        "## Config",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in report.get("config", {}).items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Alive Ray Nodes", "", "| IP | GPU resources | Alive |", "| --- | ---: | --- |"])
    for node in report.get("alive_nodes", []):
        lines.append(f"| `{node.get('ip')}` | `{node.get('gpu')}` | `{node.get('alive')}` |")

    lines.extend(["", "## Matmul Smoke", "", "| Host | Device | Size | Repeats | Seconds | TFLOP/s | Max MB |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in report.get("matmul", []):
        lines.append(
            f"| `{row.get('hostname')}` | `{row.get('device')}` | `{row.get('size')}` | "
            f"`{row.get('repeats')}` | `{row.get('elapsed_seconds')}` | `{row.get('tflops')}` | `{row.get('max_memory_mb')}` |"
        )

    lines.extend(["", "## NCCL All-Reduce", ""])
    nccl = report.get("nccl", {})
    lines.append(f"Status: `{nccl.get('status')}`")
    if "error" in nccl:
        lines.append(f"Error: `{nccl['error']}`")
    if nccl.get("ranks"):
        lines.extend(["", "| Rank | Host | Device | Bytes | Seconds | Sample | Expected |", "| ---: | --- | --- | ---: | ---: | ---: | ---: |"])
        for row in nccl["ranks"]:
            lines.append(
                f"| `{row.get('rank')}` | `{row.get('hostname')}` | `{row.get('device')}` | "
                f"`{row.get('bytes')}` | `{row.get('elapsed_seconds')}` | `{row.get('sample')}` | `{row.get('expected')}` |"
            )

    if report.get("error"):
        lines.extend(["", "## Error", "", "```text", report.get("traceback", report["error"]), "```"])

    report_file.write_text("\n".join(lines) + "\n")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])
