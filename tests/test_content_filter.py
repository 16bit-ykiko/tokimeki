from pathlib import Path

import fakes
import numpy as np
import pytest
from clips import make_clip
from PIL import Image

from tokimeki.library.episodes import stage_done, stage_params
from tokimeki.library.records import Rating, ShotStatus, TagCategory
from tokimeki.library.shots import frame_tags, list_frames, list_shots
from tokimeki.models.wd14 import Labels, Wd14Tagger, decode, preprocess
from tokimeki.stages import content_filter, pipeline
from tokimeki.stages.base import Context, open_series, register_episodes
from tokimeki.stages.content_filter import UNSAFE_THRESHOLD, is_unsafe


def test_unsafe_threshold_is_conservative() -> None:
    assert UNSAFE_THRESHOLD <= 0.25
    assert not is_unsafe(Rating(0.9, 0.3, 0.1, 0.05))
    assert is_unsafe(Rating(0.6, 0.3, 0.15, 0.05))
    assert is_unsafe(Rating(0.1, 0.1, 0.0, 0.3))


def test_preprocess_pads_white_and_swaps_to_bgr() -> None:
    data = preprocess(Image.new("RGB", (40, 20), (255, 0, 0)), 40)
    assert data.shape == (40, 40, 3)
    assert data[0, 0].tolist() == [255, 255, 255]
    assert data[20, 20].tolist() == [0, 0, 255]


def test_decode_keeps_tags_above_thresholds() -> None:
    labels = Labels(
        ["general", "sensitive", "questionable", "explicit", "smile", "blush", "momo"],
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
        np.array([6]),
    )
    probs = np.array([0.9, 0.1, 0.02, 0.01, 0.7, 0.2, 0.95], dtype=np.float32)
    prediction = decode(probs, labels)
    assert prediction.rating["questionable"] == pytest.approx(0.02)
    assert prediction.general == {"smile": pytest.approx(0.7)}
    assert prediction.character == {"momo": pytest.approx(0.95)}


def _ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Context:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    pipeline.run(ctx, register_episodes(ctx), until="filter")
    return ctx


def test_unsafe_shots_leave_no_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ingest(tmp_path, monkeypatch)
    red, blue = list_shots(ctx.conn, 1)
    assert (red.status, blue.status) == (ShotStatus.DROPPED, ShotStatus.KEPT)
    assert (red.start_frame, red.end_frame) == (0, fakes.CUT)
    assert list_frames(ctx.conn, red.id) == []
    kept = list_frames(ctx.conn, blue.id)
    assert len(kept) == 3
    cached = sorted(int(p.stem) for p in ctx.paths.frames_dir(1).glob("*.jpg"))
    assert cached == [f.frame_index for f in kept]
    tags = frame_tags(ctx.conn, kept[0].id)
    assert {(t.name, t.category) for t in tags} == {
        ("smile", TagCategory.GENERAL),
        ("momo_velia_deviluke", TagCategory.CHARACTER),
    }
    assert stage_params(ctx.conn, 1, "filter") == {
        "model": "fake/wd14",
        "unsafe_threshold": UNSAFE_THRESHOLD,
    }


def test_rerun_skips_and_redo_rejudges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ingest(tmp_path, monkeypatch)
    episodes = register_episodes(ctx)
    monkeypatch.setattr(content_filter, "Wd14Tagger", None)
    pipeline.run(ctx, episodes, until="filter")

    monkeypatch.setattr(content_filter, "Wd14Tagger", fakes.FakeTagger)
    pipeline.redo(ctx, episodes, "filter")
    assert not stage_done(ctx.conn, 1, "filter")
    assert {s.status for s in list_shots(ctx.conn, 1)} == {ShotStatus.PENDING}
    pipeline.run(ctx, episodes, until="filter")
    statuses = [s.status for s in list_shots(ctx.conn, 1)]
    assert statuses == [ShotStatus.DROPPED, ShotStatus.KEPT]
    assert len(list(ctx.paths.frames_dir(1).glob("*.jpg"))) == 3


@pytest.mark.gpu
def test_wd14_rates_on_the_gpu() -> None:
    tagger = Wd14Tagger()
    try:
        (prediction,) = tagger.predict([Image.new("RGB", (640, 360), (240, 240, 240))])
    finally:
        tagger.close()
    assert set(prediction.rating) == {"general", "sensitive", "questionable", "explicit"}
    assert not is_unsafe(Rating(**prediction.rating))
