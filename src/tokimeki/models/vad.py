"""Silero VAD (the batched "sequence" ONNX export shipped in the `silero-vad` package)."""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.models.ort import OnnxModel

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
FRAMES_PER_CALL = 512
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


def _model_path() -> Path:
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("the silero-vad package is not installed")
    return Path(spec.origin).parent / "data" / "silero_vad_16k_sequence.onnx"


def frame_blocks(audio: NDArray[np.float32]) -> list[NDArray[np.float32]]:
    """Blocks of `(frames, 64 + 512)`: each 32 ms frame led by the last 64 samples before it."""
    count = -(-len(audio) // FRAME_SAMPLES)
    frames = np.zeros((count, FRAME_SAMPLES), dtype=np.float32)
    frames.reshape(-1)[: len(audio)] = audio
    context = np.zeros((count, CONTEXT_SAMPLES), dtype=np.float32)
    context[1:] = frames[:-1, -CONTEXT_SAMPLES:]
    joined = np.concatenate([context, frames], axis=1)
    return [joined[i : i + FRAMES_PER_CALL] for i in range(0, count, FRAMES_PER_CALL)]


class SileroVad:
    def __init__(self) -> None:
        self._model = OnnxModel(_model_path())

    def speech_probabilities(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Probability of speech per 32 ms frame of 16 kHz mono audio."""
        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        cell = np.zeros((1, 1, 128), dtype=np.float32)
        out: list[NDArray[np.float32]] = []
        for block in frame_blocks(audio):
            result = self._model.run_named({"input": block, "h": hidden, "c": cell})
            out.append(result["speech_probs"].reshape(-1))
            hidden, cell = result["hn"], result["cn"]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self._model.close()
