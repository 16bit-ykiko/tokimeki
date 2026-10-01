from pathlib import Path

import fakes
import pytest
from clips import make_clip

from tokimeki.library.shots import list_shots
from tokimeki.report import build_report
from tokimeki.report.render import clock, framing
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes


def test_formatting() -> None:
    assert clock(83.25) == "01:23.2"
    assert [framing(h) for h in (0.5, 0.2, 0.05)] == ["close-up", "medium", "wide"]


def test_report_shows_kept_shots_and_no_dropped_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    pipeline.run(ctx, register_episodes(ctx))
    dropped, kept = list_shots(ctx.conn, 1)

    index = build_report(ctx.conn, ctx.paths)
    assert index == tmp_path / ".tokimeki" / "report" / "index.html"
    html = index.read_text()
    images = sorted(p.name for p in (index.parent / "img").iterdir())
    shot_images = [name for name in images if name.startswith("shot-")]
    assert shot_images == [f"shot-{kept.id}.jpg"]
    assert f"shot-{dropped.id}.jpg" not in html
    assert any(name.startswith("face-") for name in images)
    assert "50.0%" in html
    assert "00:00.0&ndash;00:01.0" in html
    assert f"tokimeki cast name {tmp_path} 1 &quot;NAME&quot;" in html
    assert "momo_velia_deviluke 100%" in html


def test_report_forgets_shots_dropped_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    index = build_report(ctx.conn, ctx.paths)
    assert any(p.name.startswith("shot-") for p in (index.parent / "img").iterdir())

    monkeypatch.setattr(fakes, "RED", fakes.BLUE)
    monkeypatch.setattr(fakes.FakeTagger, "predict", fakes.unsafe_everything)
    pipeline.redo(ctx, episodes, "filter")
    pipeline.run(ctx, episodes)
    build_report(ctx.conn, ctx.paths)
    assert not any(p.name.startswith("shot-") for p in (index.parent / "img").iterdir())
    assert not any(ctx.paths.frames_dir(1).iterdir())
