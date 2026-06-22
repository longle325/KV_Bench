"""SSH and bundle-sync helpers for Machine B.

Only orchestration traffic should go through SSH; benchmark data paths use the
private IPs configured in .env.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Config
from .shell import log, q, run


class Remote:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def ssh(self, command: str, *, check: bool = True, capture: bool = False):
        return run(
            [
                "ssh",
                "-p",
                self.cfg.get("SSH_PORT"),
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.cfg.get('SSH_CONNECT_TIMEOUT')}",
                self.cfg.remote_login,
                command,
            ],
            check=check,
            capture=capture,
        )

    def sync_bundle(self) -> None:
        cfg = self.cfg
        remote_root = cfg.get("REMOTE_ROOT")
        if not cfg.paths.env_file.exists():
            raise RuntimeError(
                f"{cfg.paths.env_file} is missing. Copy .env.example to .env "
                "and fill machine-specific values before syncing."
            )
        self.ssh(
            f"mkdir -p {q(remote_root)} && rm -rf "
            f"{q(remote_root + '/configs')} "
            f"{q(remote_root + '/scripts')} {q(remote_root + '/kvbench_physical')} "
            f"{q(remote_root + '/requirements.txt')}"
        )

        tar_cmd = [
            "tar",
            "--exclude=logs",
            "--exclude=results",
            "--exclude=__pycache__",
            "-C",
            str(cfg.paths.node_root),
            "-czf",
            "-",
            ".env",
            "requirements.txt",
            "configs",
            "scripts",
            "kvbench_physical",
        ]
        ssh_cmd = [
            "ssh",
            "-p",
            cfg.get("SSH_PORT"),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={cfg.get('SSH_CONNECT_TIMEOUT')}",
            cfg.remote_login,
            f"tar -xzf - -C {q(remote_root)}",
        ]
        tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
        ssh_proc = subprocess.Popen(ssh_cmd, stdin=tar_proc.stdout)
        assert tar_proc.stdout is not None
        tar_proc.stdout.close()
        ssh_rc = ssh_proc.wait()
        tar_rc = tar_proc.wait()
        if tar_rc != 0 or ssh_rc != 0:
            raise RuntimeError(f"sync failed: tar={tar_rc} ssh={ssh_rc}")

        self.ssh(f"chmod +x {q(remote_root)}/scripts/*.sh")
        log("sync", f"copied runtime bundle to {cfg.remote_login}:{remote_root}")


def remote_python_command(cfg: Config, command: str, **env: object) -> str:
    remote_root = cfg.get("REMOTE_ROOT")
    exports = {
        "PYTHONPATH": remote_root,
        "NODE_ROOT": remote_root,
        "ROOT": remote_root,
        "ENV_FILE": f"{remote_root}/.env",
    }
    exports.update({key: value for key, value in env.items() if value is not None})
    prefix = " ".join(f"{key}={q(value)}" for key, value in exports.items())
    return f"cd {q(remote_root)} && {prefix} python3 -m kvbench_physical {command}"
