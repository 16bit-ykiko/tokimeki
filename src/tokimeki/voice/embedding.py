"""Speaker embeddings for anime voices (ECAPA-TDNN trained on visual-novel speech, ONNX on CUDA).

litagin's anime speaker embedding tells anime characters apart where VoxCeleb models hear
only "young woman"; its ONNX export takes 16 kHz mono samples and gives an L2-normalised
192-dimensional vector (the filterbank is inside the graph).
"""

import numpy as np
from numpy.typing import NDArray

from tokimeki.models.hub import fetch
from tokimeki.models.ort import OnnxModel

MODEL_REPO = "aloxaf/anime-speaker-embedding-char-onnx"
MODEL_FILE = "anime_char.onnx"
SAMPLE_RATE = 16000
DIMENSIONS = 192


def cosine_distances(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """1 - cosine similarity between every row of `a` and every row of `b`."""
    na = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    nb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return (1.0 - na @ nb.T).astype(np.float32)


class SpeakerEncoder:
    def __init__(self) -> None:
        self._model = OnnxModel(fetch(MODEL_REPO, MODEL_FILE))
        self._input = self._model.input_names[0]

    def embed(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """The embedding of one voice, 16 kHz mono."""
        out = self._model.run({self._input: samples.astype(np.float32)[None]})[0]
        return out[0].astype(np.float32)

    def close(self) -> None:
        self._model.close()
