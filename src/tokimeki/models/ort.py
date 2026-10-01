"""onnxruntime behind a typed wrapper that only ever runs on CUDA."""

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from tokimeki.models import CPU_THREADS
from tokimeki.models.gpu import GpuUnavailableError

CUDA_PROVIDER = "CUDAExecutionProvider"

type Shape = list[int | str | None]


class _NodeArg(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> Shape: ...


class _ModelMeta(Protocol):
    @property
    def custom_metadata_map(self) -> dict[str, str]: ...


class _Session(Protocol):
    def get_providers(self) -> list[str]: ...
    def get_inputs(self) -> list[_NodeArg]: ...
    def get_modelmeta(self) -> _ModelMeta: ...
    def run(
        self, output_names: None, input_feed: dict[str, NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]: ...


class _Options(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int
    log_severity_level: int
    graph_optimization_level: object


_ort = importlib.import_module("onnxruntime")
_available_providers = cast(Callable[[], list[str]], _ort.get_available_providers)
_new_options = cast(Callable[[], _Options], _ort.SessionOptions)
_new_session = cast(Callable[..., _Session], _ort.InferenceSession)
_OPTIMIZE_ALL = cast(object, _ort.GraphOptimizationLevel.ORT_ENABLE_ALL)


def cuda_available() -> bool:
    return CUDA_PROVIDER in _available_providers()


class OnnxModel:
    """An ONNX model on the GPU.

    onnxruntime quietly runs on the CPU when the CUDA provider fails to load; that is
    checked after the session starts and turned into an error.
    """

    def __init__(self, path: Path) -> None:
        if not cuda_available():
            raise GpuUnavailableError(
                f"onnxruntime has no {CUDA_PROVIDER}; install the CUDA build of onnxruntime"
            )
        options = _new_options()
        options.intra_op_num_threads = CPU_THREADS
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = _OPTIMIZE_ALL
        provider_options = {"cudnn_conv_algo_search": "HEURISTIC"}
        session = _new_session(
            str(path), sess_options=options, providers=[(CUDA_PROVIDER, provider_options)]
        )
        providers = session.get_providers()
        if not providers or providers[0] != CUDA_PROVIDER:
            raise GpuUnavailableError(f"{path.name} did not start on CUDA (got {providers})")
        self._session: _Session | None = session
        self.path = path

    @property
    def session(self) -> _Session:
        if self._session is None:
            raise RuntimeError(f"{self.path.name} is closed")
        return self._session

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self.session.get_inputs()]

    def input_shape(self, index: int = 0) -> Shape:
        return list(self.session.get_inputs()[index].shape)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.session.get_modelmeta().custom_metadata_map)

    def run(self, feeds: Mapping[str, NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        return self.session.run(None, dict(feeds))

    def close(self) -> None:
        self._session = None
