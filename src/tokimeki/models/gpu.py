"""GPU lifecycle: one model on the GPU at a time, loaded for a batch and freed after it."""

import gc
import importlib
import shutil
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast

import torch

WSL_DRIVER_DIR = Path("/usr/lib/wsl/lib")

GPU_MEMORY_LIMIT = 4 * 2**30
"""Most GPU memory one model may allocate.

Past the card's memory, the Windows driver under WSL quietly pages to system RAM and runs
orders of magnitude slower; with a cap, an oversized batch fails with an error instead.
"""


class GpuUnavailableError(RuntimeError):
    pass


class GpuModel(Protocol):
    def close(self) -> None: ...


_loaded: str | None = None
_set_memory_fraction = cast(
    Callable[[float], None],
    importlib.import_module("torch.cuda.memory").set_per_process_memory_fraction,
)


@contextmanager
def loaded[M: GpuModel](name: str, factory: Callable[[], M]) -> Generator[M, None, None]:
    """Load a model onto the GPU for the duration of the block, refusing to stack models."""
    global _loaded
    if _loaded is not None:
        raise RuntimeError(f"cannot load {name}: {_loaded} is still on the GPU")
    _loaded = name
    try:
        model = factory()
        try:
            yield model
        finally:
            model.close()
            del model
    finally:
        _loaded = None
        release_memory()


def loaded_model() -> str | None:
    return _loaded


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def torch_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            "PyTorch sees no CUDA device; this pipeline does not run on the CPU."
            " Check `nvidia-smi` and that the pixi environment has a CUDA build of PyTorch."
        )
    device = torch.device("cuda")
    _, total = torch.cuda.mem_get_info(device)
    _set_memory_fraction(min(1.0, GPU_MEMORY_LIMIT / total))
    return device


def nvidia_smi() -> Path:
    found = shutil.which("nvidia-smi")
    if found is not None:
        return Path(found)
    wsl = WSL_DRIVER_DIR / "nvidia-smi"
    if wsl.exists():
        return wsl
    raise GpuUnavailableError("nvidia-smi not found: is the NVIDIA driver installed?")


def gpu_status() -> str:
    """One line per GPU: name, memory used/total, utilisation (from nvidia-smi)."""
    result = subprocess.run(
        [
            str(nvidia_smi()),
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.decoder",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GpuUnavailableError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout.strip()
