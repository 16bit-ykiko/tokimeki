"""`tokimeki gpu-check`: prove that decoding and every model run on the GPU."""

import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from tokimeki.media.decode import nvdec_selftest
from tokimeki.models.faces import FaceDetector
from tokimeki.models.gpu import gpu_status, loaded, torch_cuda
from tokimeki.models.ort import cuda_available


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _ffmpeg_has_cuda_filters() -> str:
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=True
    ).stdout
    if "scale_cuda" not in filters:
        raise RuntimeError("this ffmpeg build has no scale_cuda filter (not a CUDA build)")
    return "ffmpeg has scale_cuda"


def _nvdec() -> str:
    with tempfile.TemporaryDirectory(prefix="tokimeki-check-") as tmp:
        frames = nvdec_selftest(Path(tmp))
    if frames != 24:
        raise RuntimeError(f"decoded {frames} frames of a 24-frame clip")
    return "decoded a synthetic H.264 clip with NVDEC + scale_cuda"


def _torch() -> str:
    device = torch_cuda()
    x = torch.randn(1024, 1024, device=device)
    value = float((x @ x).sum().item())
    if value != value:
        raise RuntimeError("matmul on the GPU returned NaN")
    return f"PyTorch {torch.__version__} on {torch.cuda.get_device_name(device)}"


def _onnxruntime() -> str:
    if not cuda_available():
        raise RuntimeError("onnxruntime has no CUDAExecutionProvider")
    with loaded("face detector", FaceDetector) as detector:
        detector.detect([Image.new("RGB", (640, 360))])
    return "onnxruntime ran the face detector on CUDAExecutionProvider"


def run_checks() -> list[Check]:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("driver", gpu_status),
        ("ffmpeg", _ffmpeg_has_cuda_filters),
        ("nvdec", _nvdec),
        ("torch", _torch),
        ("onnxruntime", _onnxruntime),
    ]
    results: list[Check] = []
    for name, check in checks:
        start = time.monotonic()
        try:
            detail = check()
            ok = True
        except Exception as error:
            detail, ok = f"{type(error).__name__}: {error}", False
        results.append(Check(name, ok, f"{detail} ({time.monotonic() - start:.1f}s)"))
    return results
