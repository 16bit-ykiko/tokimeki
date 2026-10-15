import subprocess
from pathlib import Path
from typing import cast

import fakes
import numpy as np
import pytest

from tokimeki import api
from tokimeki.anime import pipeline
from tokimeki.anime.pipeline import register_episodes
from tokimeki.anime.proxy import inside
from tokimeki.library.series import open_series


def test_inside_ranges_drives_ffmpeg_timeline() -> None:
    blank = f"not({inside('n', [(3.0, 5.0), (8.0, 9.0), (12.0, 20.0)])})"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=white:s=16x16:r=24",
         "-frames:v", "24", "-vf",
         f"format=gray,drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{blank}'",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    lit = np.frombuffer(raw, np.uint8).reshape(24, -1).mean(axis=1) > 128
    assert np.flatnonzero(lit).tolist() == [3, 4, 8, *range(12, 20)]


@pytest.mark.gpu
def test_proxy_is_black_and_silent_outside_kept_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "show"
    root.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=24:d=1",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "flac", str(root / "ep01.mkv")],
        check=True,
    )  # fmt: skip
    fakes.install(monkeypatch)
    ctx = open_series(root)
    pipeline.run(ctx, register_episodes(ctx))
    (made,) = cast(list[str], api.make_proxies(str(root))["proxies"])
    video = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(made), "-vf", "scale=16:9,format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    luma = np.frombuffer(video, np.uint8).reshape(-1, 81).mean(axis=1)
    assert luma[:24].max() < 20 and luma[26:].min() > 20
    audio = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(made), "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.abs(np.frombuffer(audio, np.float32))
    assert samples[200:7800].max() < 0.01 and samples[8200:15800].mean() > 0.1
    assert api.make_proxies(str(root))["proxies"] == [made]
