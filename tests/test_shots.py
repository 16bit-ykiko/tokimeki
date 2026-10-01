from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from clips import make_clip
from numpy.typing import NDArray
from PIL import Image

from tokimeki.library.episodes import get_episode, stage_done
from tokimeki.library.shots import list_shots
from tokimeki.media.decode import decode_for_transnet
from tokimeki.media.sampling import sample_frames
from tokimeki.models.transnet import TransNet, shots_from_predictions
from tokimeki.stages import pipeline, shots
from tokimeki.stages.base import open_series, register_episodes


def test_shots_from_predictions() -> None:
    p = np.zeros(20, dtype=np.float32)
    p[4] = 0.9
    p[10:13] = 0.8
    assert shots_from_predictions(p) == [(0, 5), (5, 11), (13, 20)]
    p[0] = 0.9
    assert shots_from_predictions(p)[0] == (1, 5)
    assert shots_from_predictions(np.ones(5, dtype=np.float32)) == [(0, 5)]


def test_sampling_every_half_second_and_at_least_three() -> None:
    assert sample_frames(0, 240, Fraction(24)) == [
        6,
        18,
        30,
        42,
        54,
        66,
        78,
        90,
        102,
        114,
        126,
        138,
        150,
        162,
        174,
        186,
        198,
        210,
        222,
        234,
    ]
    assert sample_frames(100, 106, Fraction(24)) == [101, 103, 105]
    assert sample_frames(0, 2, Fraction(24)) == [0, 1]
    assert sample_frames(5, 5, Fraction(24)) == []


class FakeTransNet:
    def predict(self, frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        p = np.zeros(len(frames), dtype=np.float32)
        p[23] = 1.0
        return p

    def close(self) -> None:
        pass


def fake_decode(path: Path, width: int, height: int) -> NDArray[np.uint8]:
    return np.zeros((48, 27, 48, 3), dtype=np.uint8)


def fake_extract(
    path: Path, width: int, height: int, indices: list[int], out_paths: list[Path]
) -> None:
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 18)).save(out)


def test_stage_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    monkeypatch.setattr(shots, "TransNet", FakeTransNet)
    monkeypatch.setattr(shots, "decode_for_transnet", fake_decode)
    monkeypatch.setattr("tokimeki.stages.frames.extract_frames", fake_extract)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes, until="shots")
    (episode,) = episodes
    assert stage_done(ctx.conn, episode.id, "shots")
    assert [(s.start_frame, s.end_frame) for s in list_shots(ctx.conn, episode.id)] == [
        (0, 24),
        (24, 48),
    ]
    assert len(list(ctx.paths.frames_dir(episode.id).glob("*.jpg"))) == 6
    assert get_episode(ctx.conn, episode.id).frame_count == 48

    monkeypatch.setattr(shots, "TransNet", None)
    pipeline.run(ctx, episodes, until="shots")

    monkeypatch.setattr(shots, "TransNet", FakeTransNet)
    pipeline.redo(ctx, episodes, "shots")
    assert not ctx.paths.frames_dir(episode.id).exists()
    assert list_shots(ctx.conn, episode.id) == []


@pytest.mark.gpu
def test_transnet_finds_hard_cuts(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "cuts.mkv", ["testsrc2", "mandelbrot", "color=c=orange"], seconds=2)
    frames = decode_for_transnet(clip, 320, 180)
    network = TransNet()
    try:
        spans = shots_from_predictions(network.predict(frames))
    finally:
        network.close()
    assert len(spans) == 3
    assert [start for start, _ in spans][1:] == pytest.approx([48, 96], abs=1)
