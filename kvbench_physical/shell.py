"""Shared process helpers for local shell commands and long-running services.

The workflow layer uses these helpers so command execution, runtime activation,
quoting, and screen handling stay consistent across Track A and Track B.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .config import Config


def log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}", flush=True)


def die(prefix: str, message: str, code: int = 1) -> None:
    log(prefix, f"ERROR: {message}")
    raise SystemExit(code)


def q(value: object) -> str:
    return shlex.quote(str(value))


def qjoin(parts: Iterable[object]) -> str:
    return " ".join(q(part) for part in parts)


def run(
    argv: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.STDOUT if capture else None
    proc = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )
    if check and proc.returncode != 0:
        cmd = qjoin(argv)
        output = proc.stdout if capture and proc.stdout else ""
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n{output}")
    return proc


def run_shell(
    script: str,
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    return run(
        ["bash", "-lc", script],
        cwd=cwd,
        env=env,
        check=check,
        capture=capture,
    )


def require_tools(tools: Iterable[str], prefix: str) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        die(prefix, f"missing tools: {' '.join(missing)}")


def runtime_script(cfg: Config, body: str, tools: Iterable[str] = ()) -> str:
    tools_text = " ".join(q(tool) for tool in tools)
    return f"""
set -Eeuo pipefail
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" >/dev/null 2>&1 || true
fi
for tool in {tools_text}; do
  command -v "$tool" >/dev/null 2>&1 && continue
  if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    break
  fi
done
{body}
""".strip()


def run_runtime(
    cfg: Config,
    body: str,
    *,
    tools: Iterable[str] = (),
    cwd: Path | str | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    merged = cfg.env.copy()
    if env:
        merged.update(env)
    return run_shell(
        runtime_script(cfg, body, tools),
        cwd=cwd,
        env=merged,
        check=check,
        capture=capture,
    )


def cuda_exports() -> str:
    return r"""
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
fi
""".strip()


def python_module_command(cfg: Config, command: str, **env: object) -> str:
    exports = {
        "PYTHONPATH": f"{cfg.paths.node_root}:{os.environ.get('PYTHONPATH', '')}",
        "NODE_ROOT": str(cfg.paths.node_root),
        "ROOT": str(cfg.paths.repo_root),
        "ENV_FILE": str(cfg.paths.env_file),
    }
    exports.update({key: value for key, value in env.items() if value is not None})
    prefix = " ".join(f"{key}={q(value)}" for key, value in exports.items())
    return f"{prefix} {q(sys.executable)} -m kvbench_physical {command}"


def start_screen(name: str, command: str) -> None:
    """Start a command in screen and let the session close when it exits."""
    run(["screen", "-dmS", name, "bash", "-lc", command])


def stop_screens_matching(pattern: str) -> None:
    regex = re.compile(pattern)
    proc = run(["screen", "-ls"], check=False, capture=True)
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split()
        if parts and regex.search(parts[0]):
            run(["screen", "-S", parts[0], "-X", "stuff", "\003"], check=False)


def screen_exists(name: str) -> bool:
    proc = run(["screen", "-ls"], check=False, capture=True)
    return any(name in line for line in (proc.stdout or "").splitlines())
