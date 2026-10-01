"""Anime face detection (deepghs' YOLOv8 face models)."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.models.hub import fetch
from tokimeki.models.ort import OnnxModel

DETECTOR_REPO = "deepghs/anime_face_detection"
DETECTOR_MODEL = "face_detect_v1.4_s"
NMS_IOU = 0.7
BATCH_SIZE = 16
INPUT_LONG_SIDE = 640
STRIDE = 32


@dataclass(frozen=True, slots=True)
class Detection:
    """A face box normalised to the image: (0, 0) top left, (1, 1) bottom right."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(
            np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]),
            0,
            None,
        )
        h = np.clip(
            np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]),
            0,
            None,
        )
        inter = w * h
        overlap = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[overlap <= iou]
    return keep


def decode_yolo(
    output: NDArray[np.float32], input_w: int, input_h: int, threshold: float, iou: float = NMS_IOU
) -> list[Detection]:
    """YOLOv8 output `(4 + classes, anchors)` with centre-size boxes in input pixels."""
    scores = output[4:].max(axis=0)
    candidates = output[:, scores > threshold].T
    if not len(candidates):
        return []
    cx, cy, w, h = candidates[:, 0], candidates[:, 1], candidates[:, 2], candidates[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
    kept_scores = candidates[:, 4:].max(axis=1)
    out: list[Detection] = []
    for i in nms(boxes, kept_scores, iou):
        x0, y0, x1, y1 = np.clip(boxes[i] / [input_w, input_h, input_w, input_h], 0.0, 1.0)
        out.append(Detection(float(x0), float(y0), float(x1), float(y1), float(kept_scores[i])))
    return out


def input_size(width: int, height: int, long_side: int = INPUT_LONG_SIDE) -> tuple[int, int]:
    """Keep the aspect ratio, long side `long_side`, both sides a multiple of the YOLO stride."""
    scale = long_side / max(width, height)
    return (
        math.ceil(width * scale / STRIDE) * STRIDE,
        math.ceil(height * scale / STRIDE) * STRIDE,
    )


class FaceDetector:
    def __init__(self, model: str = DETECTOR_MODEL) -> None:
        self._model = OnnxModel(fetch(DETECTOR_REPO, f"{model}/model.onnx"))
        with fetch(DETECTOR_REPO, f"{model}/threshold.json").open() as f:
            self.threshold = float(cast(dict[str, float], json.load(f))["threshold"])
        self.batch_size = BATCH_SIZE

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """CPU side: resized as YOLO's rectangular inference does, as NCHW 0-1 floats.

        All images in a batch must share one size (frames of an episode do).
        """
        size = input_size(*images[0].size)
        pixels = np.stack([np.asarray(im.convert("RGB").resize(size)) for im in images])
        batch = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2)).astype(np.float32)
        batch *= np.float32(1 / 255)
        return batch

    def infer(self, batch: NDArray[np.float32]) -> list[list[Detection]]:
        """GPU side: the faces in each image of a batch from `prepare`."""
        height, width = batch.shape[2], batch.shape[3]
        outputs = self._model.run({self._model.input_names[0]: batch})[0]
        return [decode_yolo(o, width, height, self.threshold) for o in outputs]

    def detect(self, images: Sequence[Image.Image]) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for start in range(0, len(images), self.batch_size):
            out += self.infer(self.prepare(images[start : start + self.batch_size]))
        return out

    def close(self) -> None:
        self._model.close()
