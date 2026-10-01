import importlib
import subprocess
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Protocol, cast

import fakes
import pytest
from clips import make_clip
from test_plan import song

from tokimeki import api
from tokimeki.mad.fonts import Fonts
from tokimeki.mad.plan import Plan, SlotPlan
from tokimeki.mad.render import PREVIEW, RenderSettings, clip_key, slot_frames
from tokimeki.mad.subtitles import write_lyrics
from tokimeki.models.timeline import TimelineClip, write_timeline
from tokimeki.song.analysis import SongSource, save_analysis
from tokimeki.song.lyrics import LyricLine, pair
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes


def _plan(bounds: list[float]) -> Plan:
    slots = [SlotPlan(a, b, i) for i, (a, b) in enumerate(pairwise(bounds))]
    return Plan("t", "/s", "梦梦", "test-song", slots)


def test_slot_frames_add_up_from_the_excerpt_start() -> None:
    plan = _plan([10.0, 10.37, 11.11, 11.49, 12.98, 14.01])
    assert sum(slot_frames(plan, s) for s in plan.slots) == round(4.01 * plan.frame_rate)


def test_clip_key_changes_with_what_is_rendered(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x")
    clip = SlotPlan(0.0, 1.0, 1, "", 10.0, 11.0)
    key = clip_key(source, clip, 24, PREVIEW, "24000/1001")
    assert key == clip_key(source, replace(clip, why="other"), 24, PREVIEW, "24000/1001")
    assert key != clip_key(source, replace(clip, source_out=10.95), 24, PREVIEW, "24000/1001")
    assert key != clip_key(source, clip, 25, PREVIEW, "24000/1001")
    big = RenderSettings("final", 1920, 1080, 19, "p6")
    assert key != clip_key(source, clip, 24, big, "24000/1001")


class _Duration(Protocol):
    def to_seconds(self) -> float: ...


class _Timeline(Protocol):
    def duration(self) -> _Duration: ...


def test_timeline_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "t.otio"
    video = [
        TimelineClip("0 intro", tmp_path / "ep.mkv", 10.0, 48),
        TimelineClip("1 chorus", tmp_path / "ep.mkv", 20.0, 24, 0.95),
    ]
    audio = [TimelineClip("song", tmp_path / "song.flac", 0.0, 72)]
    write_timeline(path, "mad", 24.0, video, audio)
    adapters = importlib.import_module("opentimelineio.adapters")
    read = cast(Callable[[str], _Timeline], adapters.read_from_file)
    assert read(str(path)).duration().to_seconds() == pytest.approx(3.0)
    assert '"LinearTimeWarp.1"' in path.read_text()


@pytest.mark.gpu
def test_render_joins_clips_on_the_beat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKIMEKI_HOME", str(tmp_path / "home"))
    root = tmp_path / "series"
    root.mkdir()
    make_clip(root / "ep01.mkv", ["testsrc2", "color=c=blue"], seconds=3)
    fakes.install(monkeypatch)
    ctx = open_series(root)
    pipeline.run(ctx, register_episodes(ctx))
    audio = tmp_path / "song.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=3", str(audio)], check=True
    )
    lyrics = [LyricLine(0.2, 1.0, "信じてね", "ja"), LyricLine(0.2, 1.0, "相信我", "zh")]
    save_analysis(
        replace(song(0.25, 3.0), source=SongSource(audio, 0.0, 3.0, "Test"), lyrics=lyrics)
    )
    plan = Plan("t", str(root), "", "test-song", [SlotPlan(0.0, 1.25, 1), SlotPlan(1.25, 2.5, 2)])
    plan.slots[0].shot = 2
    plan.slots[1].shot = None
    with pytest.raises(api.ApiError, match="errors"):
        api.render_plan(str(_write(tmp_path, plan)), "preview")
    plan = Plan("t", str(root), "", "test-song", [SlotPlan(0.0, 1.25, 2)])
    result = api.render_plan(str(_write(tmp_path, plan)), "preview", otio=True)
    outputs = cast(list[str], result["outputs"])
    video = next(o for o in outputs if o.endswith("preview.mp4"))
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert int(frames) == round(1.25 * Fraction(24000, 1001))
    assert any(o.endswith("timeline.otio") for o in outputs)
    assert any(o.endswith("lyrics.ass") for o in outputs)
    assert (root / ".tokimeki/mads/t/report.html").exists()


def _write(tmp_path: Path, plan: Plan) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(plan.to_json(), encoding="utf-8")
    return path


def test_lyrics_become_bilingual_ass_and_srt(tmp_path: Path) -> None:
    pairs = pair(
        [LyricLine(8.1, 10.8, "信じてね", "ja"), LyricLine(8.1, 10.8, "相信我", "zh"),
         LyricLine(30.0, 33.0, "外", "ja")]
    )  # fmt: skip
    fonts = Fonts("Yu Gothic", "Microsoft YaHei", tmp_path)
    ass = write_lyrics(tmp_path, pairs, 5.0, 20.0, fonts, "t")
    assert ass is not None
    text = ass.read_text(encoding="utf-8-sig")
    assert "Style: ja,Yu Gothic,60" in text and "Style: zh,Microsoft YaHei,42" in text
    assert "Dialogue: 0,0:00:03.10,0:00:05.80,ja,,0,0,0,,{\\fad(150,220)}信じてね" in text
    assert "外" not in text
    assert (
        (tmp_path / "lyrics.srt")
        .read_text()
        .startswith("1\n00:00:03,100 --> 00:00:05,800\n信じてね\n相信我")
    )
    assert write_lyrics(tmp_path, pairs, 40.0, 50.0, fonts, "t") is None
