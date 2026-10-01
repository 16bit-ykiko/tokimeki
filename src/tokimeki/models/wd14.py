"""WD14 tagger (SmilingWolf's v3 ONNX models): content rating plus general and character tags."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.models.hub import fetch
from tokimeki.models.ort import OnnxModel

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
BATCH_SIZE = 8

RATINGS = ("general", "sensitive", "questionable", "explicit")
_RATING, _GENERAL, _CHARACTER = 9, 0, 4


@dataclass(frozen=True, slots=True)
class Prediction:
    rating: dict[str, float]
    general: dict[str, float]
    character: dict[str, float]


@dataclass(frozen=True, slots=True)
class Labels:
    names: list[str]
    rating: NDArray[np.intp]
    general: NDArray[np.intp]
    character: NDArray[np.intp]


def load_labels(path: Path) -> Labels:
    names: list[str] = []
    categories: list[int] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            categories.append(int(row["category"]))
    cats = np.array(categories)
    return Labels(
        names,
        np.flatnonzero(cats == _RATING),
        np.flatnonzero(cats == _GENERAL),
        np.flatnonzero(cats == _CHARACTER),
    )


def preprocess(image: Image.Image, size: int) -> NDArray[np.float32]:
    """Fit into a white `size` square (centred) and lay out as BGR 0-255 floats (HWC)."""
    image = image.convert("RGB")
    scale = size / max(image.size)
    width, height = round(image.width * scale), round(image.height * scale)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas = np.full((size, size, 3), 255, dtype=np.float32)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top : top + height, left : left + width] = np.asarray(image)[:, :, ::-1]
    return canvas


def decode(
    probs: NDArray[np.float32],
    labels: Labels,
    general_threshold: float = GENERAL_THRESHOLD,
    character_threshold: float = CHARACTER_THRESHOLD,
) -> Prediction:
    def above(indices: NDArray[np.intp], threshold: float) -> dict[str, float]:
        return {labels.names[i]: float(probs[i]) for i in indices if probs[i] >= threshold}

    return Prediction(
        rating={labels.names[i]: float(probs[i]) for i in labels.rating},
        general=above(labels.general, general_threshold),
        character=above(labels.character, character_threshold),
    )


class Wd14Tagger:
    def __init__(self, repo: str = MODEL_REPO) -> None:
        self.repo = repo
        self.labels = load_labels(fetch(repo, "selected_tags.csv"))
        if [self.labels.names[i] for i in self.labels.rating] != list(RATINGS):
            raise ValueError(f"{repo}: unexpected rating labels")
        self._model = OnnxModel(fetch(repo, "model.onnx"))
        size = self._model.input_shape()[1]
        if not isinstance(size, int):
            raise ValueError(f"{repo}: input size is not fixed")
        self.size = size

        self.batch_size = BATCH_SIZE

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """CPU side: one input batch."""
        return np.stack([preprocess(im, self.size) for im in images])

    def infer(self, batch: NDArray[np.float32]) -> list[Prediction]:
        """GPU side: predictions for a batch from `prepare`."""
        probs = self._model.run({self._model.input_names[0]: batch})[0]
        return [decode(row, self.labels) for row in probs]

    def predict(self, images: Sequence[Image.Image]) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(images), self.batch_size):
            out += self.infer(self.prepare(images[start : start + self.batch_size]))
        return out

    def close(self) -> None:
        self._model.close()
