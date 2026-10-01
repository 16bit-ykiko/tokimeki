"""CCIP character embeddings (deepghs' contrastive anime character image pretraining)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.models.hub import fetch
from tokimeki.models.ort import OnnxModel

CCIP_REPO = "deepghs/ccip_onnx"
CCIP_MODEL = "ccip-caformer-24-randaug-pruned"
CCIP_SIZE = 384
CCIP_BATCH_SIZE = 16
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)


def ccip_preprocess(image: Image.Image) -> NDArray[np.float32]:
    resized = image.convert("RGB").resize((CCIP_SIZE, CCIP_SIZE), Image.Resampling.BILINEAR)
    data = (np.asarray(resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return (data - _CLIP_MEAN) / _CLIP_STD


class CcipEncoder:
    def __init__(self, model: str = CCIP_MODEL) -> None:
        self._model = OnnxModel(fetch(CCIP_REPO, f"{model}/model_feat.onnx"))
        self.batch_size = CCIP_BATCH_SIZE

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """CPU side: one input batch."""
        return np.stack([ccip_preprocess(im) for im in images]).astype(np.float32)

    def infer(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """GPU side: one embedding row per image of a batch from `prepare`."""
        return self._model.run({self._model.input_names[0]: batch})[0]

    def embed(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        chunks = [
            self.infer(self.prepare(images[start : start + self.batch_size]))
            for start in range(0, len(images), self.batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)

    def close(self) -> None:
        self._model.close()


# From the model's metrics.json and cluster.json on the Hub.
CCIP_SAME_THRESHOLD = 0.17847511429108218
CCIP_DBSCAN_EPS = 0.12921094122454668
CCIP_DBSCAN_MIN_SAMPLES = 2


def ccip_differences(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """CCIP difference between every row of `a` and of `b`, in [0, 1]; lower means more alike.

    This is what the model's `model_metrics.onnx` computes: (1 - cosine similarity) / 2.
    """
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.clip((1.0 - an @ bn.T) / 2.0, 0.0, 1.0).astype(np.float32)
