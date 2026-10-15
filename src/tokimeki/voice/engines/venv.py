"""Environments for the external engines, sharing tokimeki's PyTorch.

Each engine gets a venv built on tokimeki's own Python with its site packages visible, so it
runs on the CUDA PyTorch already installed (no second copy of it is downloaded); its other
requirements go into the venv, never into tokimeki's environment. PyTorch, torchaudio and
NumPy are kept out of the venv's resolution, and the torchaudio calls the engines make come
from the stand-in in `standin/` (conda-forge builds no torchaudio for this PyTorch).
"""

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from tokimeki.voice.paths import engines_dir

SHARED = ("torch", "torchaudio", "torchvision", "triton", "numpy")
"""Packages the engines take from tokimeki's environment, never from their own resolution."""
STANDIN = Path(__file__).parent / "standin"


class EngineError(RuntimeError):
    pass


def run(
    args: Sequence[str | Path],
    log: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run a command with its output appended to `log`; an error carries the log's tail."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        out.write(f"\n$ {' '.join(map(str, args))}\n")
        out.flush()
        result = subprocess.run(
            [str(a) for a in args],
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise EngineError(f"{Path(str(args[0])).name} failed (exit {result.returncode}):\n{tail}")


def checkout(url: str, commit: str, repo: Path, log: Path) -> None:
    if not (repo / ".git").exists():
        run(["git", "clone", "--quiet", url, repo], log)
    run(["git", "-C", repo, "fetch", "--quiet", "origin"], log)
    run(["git", "-C", repo, "checkout", "--quiet", commit], log)


def site_packages(venv: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv / "lib" / version / "site-packages"


def build(venv: Path, requirements: Sequence[str], log: Path) -> None:
    """A venv on tokimeki's Python and PyTorch with `requirements` and the torchaudio stand-in."""
    uv = shutil.which("uv")
    if uv is None:
        raise EngineError("uv is needed to build an engine's environment (pixi installs it)")
    if not (venv / "bin" / "python").exists():
        run(
            [uv, "venv", "--quiet", "--system-site-packages", "--python", sys.executable, venv], log
        )
    excludes = venv.parent / "shared.txt"
    excludes.write_text("".join(f"{name}\n" for name in SHARED))
    wanted = venv.parent / "requirements.txt"
    wanted.write_text("".join(f"{r}\n" for r in requirements))
    pip = [uv, "pip", "install", "--quiet", "--python", venv / "bin" / "python"]
    run([*pip, "-r", wanted, "--excludes", excludes], log)
    standin = engines_dir() / "standin"
    if standin.exists():
        shutil.rmtree(standin)
    shutil.copytree(STANDIN, standin, ignore=shutil.ignore_patterns("README.md", "__pycache__"))
    (site_packages(venv) / "tokimeki-standin.pth").write_text(f"{standin}\n")


def fetch_to(repo_id: str, filename: str, directory: Path) -> Path:
    """A Hub file downloaded to `directory/filename` (the layout an engine expects)."""
    download = cast(Callable[..., str], importlib.import_module("huggingface_hub").hf_hub_download)
    return Path(download(repo_id=repo_id, filename=filename, local_dir=str(directory)))
