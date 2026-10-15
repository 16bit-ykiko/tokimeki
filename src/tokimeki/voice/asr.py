"""Japanese speech recognition with Whisper models (transformers, fp16 on CUDA).

Anime Whisper (kotoba-whisper-v2.0 fine-tuned on 5,300 hours of visual-novel lines) keeps
stammers, laughs and breaths and punctuates like a script, which is what a TTS bank wants;
it must run without an initial prompt.
"""

import importlib
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from tokimeki.models.gpu import torch_cuda

MODEL_REPO = "litagin/anime-whisper"
SAMPLE_RATE = 16000
BATCH_SIZE = 8


class _Features(Protocol):
    @property
    def input_features(self) -> torch.Tensor: ...


class _Processor(Protocol):
    def __call__(
        self, audio: list[NDArray[np.float32]], *, sampling_rate: int, return_tensors: str
    ) -> _Features: ...

    def batch_decode(self, ids: torch.Tensor, *, skip_special_tokens: bool) -> list[str]: ...


class _Model(Protocol):
    def to(self, device: torch.device) -> "_Model": ...
    def eval(self) -> "_Model": ...
    def generate(self, input_features: torch.Tensor, **kwargs: object) -> torch.Tensor: ...


class Transcriber:
    def __init__(self, repo: str = MODEL_REPO) -> None:
        transformers = importlib.import_module("transformers")
        load_processor = cast(
            Callable[..., _Processor], transformers.WhisperProcessor.from_pretrained
        )
        load_model = cast(
            Callable[..., _Model], transformers.WhisperForConditionalGeneration.from_pretrained
        )
        self.repo = repo
        self._device = torch_cuda()
        self._processor = load_processor(repo)
        self._model = load_model(repo, torch_dtype=torch.float16).to(self._device).eval()

    def transcribe(self, clips: Sequence[NDArray[np.float32]]) -> list[str]:
        """The text of each 16 kHz mono clip (at most 30 s each)."""
        out: list[str] = []
        for start in range(0, len(clips), BATCH_SIZE):
            batch = [np.asarray(c, dtype=np.float32) for c in clips[start : start + BATCH_SIZE]]
            features = self._processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = features.input_features.to(self._device, torch.float16)
            with torch.inference_mode():
                ids = self._model.generate(
                    inputs,
                    language="ja",
                    task="transcribe",
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.0,
                )
            out += [t.strip() for t in self._processor.batch_decode(ids, skip_special_tokens=True)]
        return out

    def close(self) -> None:
        del self._model
