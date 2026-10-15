import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from test_plan import EPISODE, candidate, song

from tokimeki.library.db import open_library
from tokimeki.mad.candidates import Candidate, FramePoint
from tokimeki.mad.fx import ClipFx, clip_fx, suggest, zoom_filter
from tokimeki.mad.motion import Onset, ShotMotion
from tokimeki.mad.plan import Fx, Plan, PlanFormatError, SlotPlan
from tokimeki.mad.refine import refine
from tokimeki.mad.render import clip_key
from tokimeki.mad.song.analysis import SongAnalysis
from tokimeki.mad.validate import validate
from tokimeki.render.video import PREVIEW

SIZE = (320, 180)


def _zoomed(fx: ClipFx, frames: int) -> NDArray[np.uint8]:
    """Gray frames with a white square around (0.25, 0.5), through the clip's zoom filter."""
    zoom = zoom_filter(fx, frames, Fraction(24))
    assert zoom is not None
    width, height = SIZE
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=gray:s={width}x{height}:r=24",
         "-frames:v", str(frames), "-threads", "1",
         "-vf", f"drawbox=x=72:y=82:w=16:h=16:color=white:t=fill,format=gray,{zoom}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(frames, height, width)


def _square(frame: NDArray[np.uint8]) -> tuple[float, float, float]:
    """Centre and area of the white square."""
    ys, xs = np.nonzero(frame > 200)
    return float(xs.mean()), float(ys.mean()), float(len(xs))


def test_drift_pushes_in_toward_the_focus_which_stays_put() -> None:
    frames = _zoomed(ClipFx(drift=(1.2, (0.25, 0.5))), 24)
    x0, y0, area0 = _square(frames[0])
    x1, y1, area1 = _square(frames[-1])
    assert area0 == 256
    assert area1 == pytest.approx(256 * 1.2**2, rel=0.1)
    assert (x1, y1) == pytest.approx((x0, y0), abs=1.0)
    areas = [_square(f)[2] for f in frames]
    assert areas == sorted(areas)


def test_punch_peaks_on_its_frame_and_eases_back() -> None:
    frames = _zoomed(ClipFx(punch=(1.2, (0.25, 0.5), 10)), 30)
    areas = [_square(f)[2] for f in frames]
    assert areas[8] == 256 and areas[20] == 256
    assert max(areas) == areas[10] == pytest.approx(256 * 1.44, rel=0.1)
    assert areas[9] > areas[8] and areas[11] < areas[10]
    both = _zoomed(ClipFx(drift=(1.1, (0.25, 0.5)), punch=(1.2, (0.25, 0.5), 10)), 30)
    assert _square(both[10])[2] > areas[10]


def _accented(analysis: SongAnalysis, strong: dict[float, float]) -> SongAnalysis:
    assert analysis.accents is not None
    accents = [
        strong.get(round(b, 3), a) for b, a in zip(analysis.beats, analysis.accents, strict=True)
    ]
    return replace(analysis, accents=accents)


def test_suggest_flashes_chorus_entries_and_hard_downbeats_and_drifts_still_shots() -> None:
    analysis = _accented(song(0.5, 20.0), {12.0: 0.9, 15.5: 0.85})
    sections = ["verse", "verse", "chorus", "chorus", "chorus", "chorus", "chorus"]
    bounds = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 19.0]
    slots = [
        SlotPlan(a, b, i, source_in=100.0 * i, source_out=100.0 * i + b - a, section=section)
        for i, (a, b, section) in enumerate(zip(bounds, bounds[1:], sections, strict=False))
    ]
    still = ShotMotion((), 0.0, 0.0)
    blink = ShotMotion((Onset(101.0, 0.2),), 0.05, 0.0)
    motion = {1: blink, 3: still, 4: still, 5: still, 6: still}
    plan, notes = suggest(Plan("p", "s", "c", "x", slots), analysis, motion)
    kinds = [[f.kind for f in s.fx] for s in plan.slots]
    assert kinds == [[], [], ["flash"], ["flash", "drift"], ["punch"], ["drift"], []]
    assert plan.slots[4].fx[0].at == 15.5
    assert len(notes) == 5


def test_refine_fills_the_punch_beat_and_the_focus_on_her_face() -> None:
    c = candidate(5, 6.0, {"smile": 0.9})
    frames = (FramePoint(c.start + 3.0, 1.0, 0.4, {"smile": 0.9}, (0.7, 0.3)),)
    c = Candidate(c.episode, c.shot, c.scene, c.presence, c.face_height, c.others, c.expressions,
                  frames, c.lines)  # fmt: skip
    analysis = _accented(song(), {1.0: 0.9})
    slot = SlotPlan(0.0, 2.0, 5, fx=[Fx("punch"), Fx("drift"), Fx("flash")])
    refined, notes = refine(Plan("p", "s", "梦梦", "test-song", [slot]), analysis, {5: c})
    punch, drift, flash = refined.slots[0].fx
    assert (punch.at, punch.zoom, punch.focus) == (1.0, 1.12, (0.7, 0.3))
    assert (drift.zoom, drift.focus) == (1.08, (0.7, 0.3))
    assert (flash.frames, flash.strength) == (3, 0.85)
    assert any("punch on the beat at 1.00s" in n for n in notes)
    again = Plan.from_json(refined.to_json())
    assert again.slots[0].fx == refined.slots[0].fx


def test_clip_fx_counts_the_accent_from_the_cut_and_joins_the_cache_key(tmp_path: Path) -> None:
    plan = Plan("p", "s", "c", "x", [SlotPlan(10.0, 12.0, 1), SlotPlan(12.0, 14.0, 2)])
    slot = replace(plan.slots[1], source_in=100.0, source_out=102.0)
    assert clip_fx(plan, slot) is None
    slot.fx = [Fx("punch", at=13.0, zoom=1.1, focus=(0.4, 0.4)), Fx("flash")]
    fx = clip_fx(plan, slot)
    assert fx is not None and fx == ClipFx((3, 0.85), None, (1.1, (0.4, 0.4), 24))
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x")
    plain = clip_key(source, slot, 48, PREVIEW, "24000/1001")
    assert plain == clip_key(source, slot, 48, PREVIEW, "24000/1001", None)
    assert plain != clip_key(source, slot, 48, PREVIEW, "24000/1001", fx)
    other = replace(fx, punch=(1.2, (0.4, 0.4), 24))
    assert clip_key(source, slot, 48, PREVIEW, "24000/1001", other) != clip_key(
        source, slot, 48, PREVIEW, "24000/1001", fx
    )


def test_fx_parse_from_names_and_objects() -> None:
    plan = Plan.from_dict(
        {"name": "n", "series": "s", "character": "c", "song": "x",
         "slots": [{"start": 0, "end": 1, "shot": 1,
                    "fx": ["flash", {"kind": "punch", "at": 0.5, "focus": [0.2, 0.3]}]}]}
    )  # fmt: skip
    assert plan.slots[0].fx == [Fx("flash"), Fx("punch", at=0.5, focus=(0.2, 0.3))]
    with pytest.raises(PlanFormatError, match=r"fx\[0\]\.focus"):
        Plan.from_dict(
            {"name": "n", "series": "s", "character": "c", "song": "x",
             "slots": [{"start": 0, "end": 1, "shot": 1, "fx": [{"kind": "drift", "focus": 1}]}]}
        )  # fmt: skip
    assert EPISODE.fps == 24


def test_validate_knows_fx_and_punches_only_on_accents() -> None:
    analysis = _accented(song(0.5, 20.0), {11.0: 0.9})
    slot = SlotPlan(10.0, 12.0, 1, fx=[Fx("sparkle"), Fx("flash", frames=9, at=10.5)])
    punches = [
        SlotPlan(12.0, 14.0, 2, fx=[Fx("punch", at=13.0), Fx("punch", at=13.02, zoom=2.0)]),
        SlotPlan(14.0, 16.0, 3, accent=15.0, fx=[Fx("punch", at=15.0), Fx("drift")]),
        SlotPlan(16.0, 16.5, 4, fx=[Fx("punch", at=17.0), Fx("drift", focus=(1.2, 0.5))]),
    ]
    plan = Plan("p", "s", "c", "x", [slot, *punches])
    with open_library(":memory:") as conn:
        issues = [i for i in validate(conn, plan, analysis) if i.code.startswith("fx.")]
    found = {(i.slot, i.code, i.level) for i in issues}
    assert found == {
        (0, "fx.unknown", "error"), (0, "fx.frames", "error"), (0, "fx.ignored", "warning"),
        (1, "fx.accent", "error"), (1, "fx.repeat", "error"), (1, "fx.zoom", "error"),
        (3, "fx.at", "error"), (3, "fx.focus", "error"), (3, "fx.short", "warning"),
    }  # fmt: skip
    good = replace(slot, fx=[Fx("flash"), Fx("punch", at=11.0)])
    with open_library(":memory:") as conn:
        assert not [
            i for i in validate(conn, replace(plan, slots=[good]), analysis) if "fx" in i.code
        ]
