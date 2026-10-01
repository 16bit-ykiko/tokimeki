"""Stand-ins for the GPU models and NVDEC, so stage logic is tested anywhere."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from tokimeki.models.wd14 import Prediction
from tokimeki.stages import content_filter, frames, shots

FRAMES = 48
CUT = 24
RED, BLUE = (220, 30, 30), (30, 30, 220)


def frame_color(index: int) -> tuple[int, int, int]:
    """The first shot is red, the second blue."""
    return RED if index < CUT else BLUE


class FakeTransNet:
    def predict(self, frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        p = np.zeros(len(frames), dtype=np.float32)
        p[CUT - 1] = 1.0
        return p

    def close(self) -> None:
        pass


def fake_decode(path: Path, width: int, height: int) -> NDArray[np.uint8]:
    return np.zeros((FRAMES, 27, 48, 3), dtype=np.uint8)


def fake_extract(
    path: Path, width: int, height: int, indices: Sequence[int], out_paths: Sequence[Path]
) -> None:
    for index, out in zip(indices, out_paths, strict=True):
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), frame_color(index)).save(out)


class FakeTagger:
    """Calls a red frame questionable."""

    repo = "fake/wd14"
    size = 64

    def predict(self, images: Sequence[Image.Image]) -> list[Prediction]:
        out: list[Prediction] = []
        for image in images:
            pixel = np.asarray(image)[0, 0]
            red = int(pixel[0]) > int(pixel[2])
            rating = {
                "general": 0.2 if red else 0.9,
                "sensitive": 0.2,
                "questionable": 0.6 if red else 0.01,
                "explicit": 0.0,
            }
            out.append(Prediction(rating, {"smile": 0.8}, {"momo_velia_deviluke": 0.9}))
        return out

    def close(self) -> None:
        pass


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shots, "TransNet", FakeTransNet)
    monkeypatch.setattr(shots, "decode_for_transnet", fake_decode)
    monkeypatch.setattr(frames, "extract_frames", fake_extract)
    monkeypatch.setattr(content_filter, "Wd14Tagger", FakeTagger)
