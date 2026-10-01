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
from test_arrange import candidate

from tokimeki.mad import make as mad
from tokimeki.mad.candidates import FramePoint
from tokimeki.mad.place import place, snap, window
from tokimeki.mad.plan import Clip, Plan, PlanSlot, Preferences, SongRef
from tokimeki.mad.render import PREVIEW, RenderSettings, clip_key, render, slot_frames
from tokimeki.models.timeline import TimelineClip, write_timeline
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes


def test_window_centres_the_peak_and_keeps_off_the_edges() -> None:
    shot = candidate(1, 6.0, {"smile": 0.9})
    c = replace(shot, frames=(FramePoint(shot.start + 3.0, 1.0, 0.4, {}),))
    start, speed = window(c, 2.0)
    assert speed == 1.0 and start == pytest.approx(c.start + 2.0)
    early, _ = window(shot, 2.0)
    assert early == pytest.approx(shot.start + 2 / 24)
    late = replace(c, frames=(FramePoint(c.end - 0.1, 1.0, 0.4, {}),))
    start, _ = window(late, 2.0)
    assert start + 2.0 <= late.end - 2 / 24 + 1e-9
    short = candidate(2, 1.0, {})
    start, speed = window(short, 1.0)
    assert speed == pytest.approx((1.0 - 4 / 24) / 1.0) or speed == pytest.approx(0.9)
    assert snap(10.02, Fraction(24)) == pytest.approx(10.0)


def _plan(slots: list[PlanSlot], clips: list[Clip], song: str = "song.flac") -> Plan:
    ref = SongRef("s", "Song", song, 0.0, 0.0, slots[-1].end, 120.0)
    return Plan("t", Preferences("梦梦", song), ref, slots, clips, "heuristic")


def test_place_fills_only_unplaced_clips() -> None:
    c = candidate(3, 5.0, {"blush": 0.8})
    slots = [PlanSlot(0, 0.0, 1.5, "chorus", 4), PlanSlot(1, 1.5, 3.0, "chorus", 4)]
    kept = Clip(1, "ep01.mkv", 4, "kept", 99.0, 1.0, 99.5)
    placed = place(_plan(slots, [Clip(0, "ep01.mkv", 3, "why"), kept]), {3: c})
    first = placed.clips[0]
    assert first.speed == 1.0 and first.peak == pytest.approx(c.peak.time, abs=1e-3)
    assert first.source_start is not None and c.start <= first.source_start <= c.end - 1.5
    assert placed.clips[1] == kept


def test_slot_frames_add_up_to_the_excerpt() -> None:
    bounds = [0.0, 0.37, 1.11, 1.49, 2.98, 4.01]
    slots = [PlanSlot(i, a, b, "verse", 4) for i, (a, b) in enumerate(pairwise(bounds))]
    plan = _plan(slots, [])
    assert sum(slot_frames(plan, s) for s in slots) == round(4.01 * plan.fps)


def test_clip_key_changes_with_what_is_rendered(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x")
    clip = Clip(0, "ep.mkv", 1, "", 10.0, 1.0, 10.5)
    key = clip_key(source, clip, 24, PREVIEW, "24000/1001")
    assert key == clip_key(source, replace(clip, reason="other"), 24, PREVIEW, "24000/1001")
    assert key != clip_key(source, replace(clip, speed=0.95), 24, PREVIEW, "24000/1001")
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


def test_plans_cannot_use_dropped_shots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    pipeline.run(ctx, register_episodes(ctx))
    slots = [PlanSlot(0, 0.0, 1.0, "chorus", 2)]
    dropped = _plan(slots, [Clip(0, "ep01.mkv", 1, "no")])
    with pytest.raises(mad.PlanError, match="not kept shots"):
        mad.plan_candidates(ctx, dropped)
    assert set(mad.plan_candidates(ctx, _plan(slots, [Clip(0, "ep01.mkv", 2, "yes")]))) == {2}


@pytest.mark.gpu
def test_render_joins_clips_on_the_beat(tmp_path: Path) -> None:
    episode = make_clip(tmp_path / "ep01.mkv", ["testsrc2", "color=c=blue"], seconds=3)
    song = tmp_path / "song.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=3", str(song)], check=True
    )
    slots = [PlanSlot(0, 0.0, 1.25, "verse", 4), PlanSlot(1, 1.25, 2.5, "chorus", 4)]
    clips = [Clip(0, "ep01.mkv", 1, "", 0.5, 1.0), Clip(1, "ep01.mkv", 2, "", 3.5, 0.9)]
    plan = _plan(slots, clips, str(song))
    ctx = open_series(tmp_path)
    out = render(ctx.paths, plan, tmp_path / "out", PREVIEW)
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert int(frames) == round(2.5 * Fraction(24000, 1001))
    assert len(list((tmp_path / ".tokimeki/cache/clips").glob("*.mp4"))) == 2
    assert episode.exists()
