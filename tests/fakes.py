"""Stand-ins for the GPU models and NVDEC, so stage logic is tested anywhere."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from tokimeki.media.audio import AudioStream
from tokimeki.models.faces import Detection
from tokimeki.models.wd14 import Prediction
from tokimeki.stages import cast, content_filter, frames, lines, shots, voice

FRAMES = 72
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


def _corners(images: Sequence[Image.Image]) -> NDArray[np.float32]:
    return np.stack([np.asarray(im, dtype=np.float32)[0, 0] for im in images])


class FakeTagger:
    """Calls a red frame questionable."""

    repo = "fake/wd14"
    size = 64
    batch_size = 2

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return _corners(images)

    def infer(self, batch: NDArray[np.float32]) -> list[Prediction]:
        out: list[Prediction] = []
        for pixel in batch:
            red = bool(pixel[0] > pixel[2])
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


class FakeDetector:
    """One face in the middle of every frame."""

    batch_size = 3

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return _corners(images)

    def infer(self, batch: NDArray[np.float32]) -> list[list[Detection]]:
        return [[Detection(0.4, 0.3, 0.6, 0.7, 0.9)] for _ in batch]

    def close(self) -> None:
        pass


class FakeEncoder:
    """The same character everywhere."""

    batch_size = 3

    def prepare(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return _corners(images)

    def infer(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ones((len(batch), 8), dtype=np.float32)

    def close(self) -> None:
        pass


class FakeVad:
    """Speech wherever the test subtitles have a line."""

    def speech_probabilities(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.full(len(audio) // 512 + 1, 0.2, dtype=np.float32)

    def close(self) -> None:
        pass


def fake_audio(
    path: Path, sample_rate: int, stream_index: int | None = None
) -> NDArray[np.float32]:
    return np.zeros(sample_rate * 40, dtype=np.float32)


class FakeSeparator:
    """Half of the mix is voice."""

    def separate(self, stereo: NDArray[np.float32]) -> NDArray[np.float32]:
        return stereo * 0.5

    def close(self) -> None:
        pass


def fake_stereo(
    path: Path, sample_rate: int, stream_index: int | None = None, channels: int = 2
) -> NDArray[np.float32]:
    return np.full((channels, sample_rate * 3), 0.2, dtype=np.float32)


def fake_main_audio(path: Path) -> AudioStream:
    return AudioStream(1, "flac", "jpn", "", True, False)


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shots, "TransNet", FakeTransNet)
    monkeypatch.setattr(shots, "decode_for_transnet", fake_decode)
    monkeypatch.setattr(frames, "extract_frames", fake_extract)
    monkeypatch.setattr(content_filter, "Wd14Tagger", FakeTagger)
    monkeypatch.setattr(cast, "FaceDetector", FakeDetector)
    monkeypatch.setattr(cast, "CcipEncoder", FakeEncoder)
    monkeypatch.setattr(lines, "SileroVad", FakeVad)
    monkeypatch.setattr(lines, "decode_audio", fake_audio)
    monkeypatch.setattr(lines, "main_audio", fake_main_audio)
    monkeypatch.setattr(voice, "VocalSeparator", FakeSeparator)
    monkeypatch.setattr(voice, "decode_audio", fake_stereo)
    monkeypatch.setattr(voice, "main_audio", fake_main_audio)


def unsafe_everything(self: FakeTagger, batch: NDArray[np.float32]) -> list[Prediction]:
    unsafe = {"general": 0.1, "sensitive": 0.3, "questionable": 0.7, "explicit": 0.1}
    return [Prediction(unsafe, {}, {}) for _ in batch]
