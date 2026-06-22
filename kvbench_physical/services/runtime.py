"""Runtime validation and opt-in package bootstrap for both machines.

Install actions are intentionally gated by ALLOW_RUNTIME_INSTALL=1 so normal
runs fail fast instead of mutating a host unexpectedly.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from ..config import Config
from ..shell import die, log, run, run_runtime


def check_runtime(cfg: Config, tools: list[str], imports: list[str]) -> None:
    checks = " && ".join(f"command -v {shlex.quote(tool)} >/dev/null" for tool in tools)
    if checks:
        proc = run_runtime(cfg, checks, tools=tools, check=False, capture=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout or f"missing tools: {tools}")

    if imports:
        import_lines = "\n".join(f"import {name}" for name in imports)
        proc = run_runtime(
            cfg,
            f"python - <<'PY'\n{import_lines}\nPY",
            tools=list(dict.fromkeys([*tools, "python"])),
            check=False,
            capture=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout or f"missing python imports: {imports}")


def ensure_runtime_or_bootstrap(cfg: Config, *, package_set: str, remote: bool = False) -> None:
    try:
        check_runtime(cfg, ["python", "ray", "vllm"], ["ray", "vllm", "torch"])
        return
    except Exception as exc:  # noqa: BLE001
        if cfg.get("ALLOW_RUNTIME_INSTALL", "0") != "1":
            where = "remote" if remote else "local"
            die(
                "runtime",
                f"{where} runtime missing ({exc}). Re-run with ALLOW_RUNTIME_INSTALL=1.",
            )

    env = cfg.merged_env(PACKAGE_SET=package_set, ALLOW_RUNTIME_INSTALL=1)
    bootstrap_runtime(cfg, env=env)


def bootstrap_runtime(cfg: Config, *, env: dict[str, str] | None = None) -> None:
    run_env = cfg.env.copy()
    if env:
        run_env.update(env)

    if run_env.get("ALLOW_RUNTIME_INSTALL") != "1":
        die(
            "bootstrap",
            "Refusing to install packages until ALLOW_RUNTIME_INSTALL=1 is set. "
            "This installs Python packages only, not VSCode/extensions/system packages.",
            code=2,
        )

    python_bin = run_env["PYTHON_BIN"]
    venv_dir = Path(run_env["VENV_DIR"]).expanduser()
    pip_args = ["--prefer-binary"]
    wheel_find_links = run_env.get("WHEEL_FIND_LINKS", "").strip()
    if wheel_find_links:
        pip_args = ["--find-links", wheel_find_links, *pip_args]

    ready_tools = ["python", "vllm"]
    ready_imports = ["vllm", "torch", "torchvision", "requests", "redis", "lmcache"]
    if run_env["PACKAGE_SET"] == "distributed":
        ready_tools.append("ray")
        ready_imports.append("ray")
    elif run_env["PACKAGE_SET"] != "replica":
        die("bootstrap", f"Unknown PACKAGE_SET={run_env['PACKAGE_SET']}; use replica or distributed.", code=2)

    try:
        check_runtime(Config(run_env, cfg.paths), ready_tools, ready_imports)
        log("bootstrap", f"Runtime already ready: {venv_dir}")
        return
    except Exception as exc:  # noqa: BLE001
        log("bootstrap", f"Runtime incomplete; installing/updating packages: {exc}")

    cuda_home = run_env.get("CUDA_HOME", "")
    if not cuda_home and Path("/usr/local/cuda").is_dir():
        run_env["CUDA_HOME"] = "/usr/local/cuda"

    if run_env.get("CUDA_HOME"):
        cuda = run_env["CUDA_HOME"]
        run_env["PATH"] = f"{cuda}/bin:{run_env.get('PATH', '')}"
        run_env["LD_LIBRARY_PATH"] = (
            f"{cuda}/lib64:{cuda}/targets/sbsa-linux/lib:{cuda}/extras/CUPTI/lib64:"
            f"{run_env.get('LD_LIBRARY_PATH', '')}"
        )

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    run([python_bin, "-m", "venv", str(venv_dir)], env=run_env)
    python = str(venv_dir / "bin" / "python")

    def pip_install(*packages: str) -> None:
        run([python, "-m", "pip", "install", *packages], env=run_env)

    pip_install("--upgrade", "pip", "setuptools", "wheel")
    pip_install(*pip_args, "--upgrade", "pip", "setuptools", "wheel", "packaging", "numpy", "ninja", "cmake")
    pip_install(*pip_args, run_env["TORCH_PACKAGE"])
    pip_install(*pip_args, "--no-deps", run_env["TORCHVISION_PACKAGE"])
    _patch_torchvision_nms(python, run_env)

    common_packages = [
        run_env["VLLM_PACKAGE"],
        run_env["REDIS_PACKAGE"],
        run_env["REQUESTS_PACKAGE"],
    ]
    if run_env["PACKAGE_SET"] == "distributed":
        common_packages.append(run_env["RAY_PACKAGE"])

    pip_install(*pip_args, *common_packages)
    if run_env.get("UNINSTALL_TORCHVISION") == "1":
        run([python, "-m", "pip", "uninstall", "-y", "torchvision"], env=run_env, check=False)

    lmcache_deps = shlex.split(run_env.get("LMCACHE_DEPS", ""))
    if lmcache_deps:
        pip_install(*pip_args, *lmcache_deps)
    pip_install(*pip_args, "--no-build-isolation", "--no-deps", run_env["LMCACHE_PACKAGE"])

    run(
        [
            python,
            "-c",
            (
                "import importlib.metadata as im\n"
                "for pkg in ('vllm','lmcache','ray','torch','torchvision','transformers','redis','requests'):\n"
                "    try: print(f'{pkg}=={im.version(pkg)}')\n"
                "    except im.PackageNotFoundError: print(f'{pkg}=missing')\n"
            ),
        ],
        env=run_env,
        check=False,
    )
    log("bootstrap", f"Runtime ready: {venv_dir}")


def _patch_torchvision_nms(python: str, env: dict[str, str]) -> None:
    """Patch a known torchvision wheel import issue on the aarch64 runtime.

    The available CUDA 12.9 torchvision wheel can import before registering its
    C++ ops. Newer torch versions then fail while registering a fake
    torchvision::nms op. Qwen3.5 only needs torchvision's Python transform
    modules during vLLM architecture inspection, so safely no-op that fake
    registration when the custom op is absent.
    """
    if env.get("PATCH_TORCHVISION_NMS", "1") != "1":
        return
    script = r'''
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])
path = site / "torchvision" / "_meta_registrations.py"
if not path.exists():
    raise SystemExit(0)

text = path.read_text()
old = '@torch.library.register_fake("torchvision::nms")'
new = """if torchvision.extension._has_ops():
    _register_fake_nms = torch.library.register_fake("torchvision::nms")
else:
    def _register_fake_nms(fn):
        return fn

@_register_fake_nms"""

if "_register_fake_nms" not in text and old in text:
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
else:
    print(f"torchvision nms patch already present or not needed: {path}")
'''
    run([python, "-c", script], env=env, check=False)
