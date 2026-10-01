import shutil
import subprocess

import pytest
import torch

from tokimeki.models.ort import cuda_available


def _has_gpu() -> bool:
    if shutil.which("ffmpeg") is None or not torch.cuda.is_available() or not cuda_available():
        return False
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False
    ).stdout
    return "scale_cuda" in filters


HAS_GPU = _has_gpu()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(reason="needs an NVIDIA GPU (NVDEC, CUDA PyTorch, CUDA onnxruntime)")
    for item in items:
        if "gpu" in item.keywords and not HAS_GPU:
            item.add_marker(skip)
