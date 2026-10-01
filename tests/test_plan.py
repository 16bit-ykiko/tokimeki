import json
from fractions import Fraction
from pathlib import Path
from typing import cast

import fakes
import numpy as np
import pytest
from clips import make_clip

from tokimeki import api
from tokimeki.library.records import Episode, Shot, ShotStatus, TagStat
from tokimeki.mad.arrange import arrange, fits
from tokimeki.mad.candidates import Candidate, FramePoint, cuteness
from tokimeki.mad.plan import PLAN_JSON_SCHEMA, Plan, PlanFormatError, SlotPlan
from tokimeki.mad.refine import frame_ceil, frame_floor, refine, window
from tokimeki.song.analysis import SongAnalysis, SongSource, save_analysis, slots_over
from tokimeki.song.slots import Slot
from tokimeki.song.structure import Section
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes

EPISODE = Episode(1, "ep01.mkv", 1920, 1080, Fraction(24), 24 * 600)


def candidate(shot_id: int, seconds: float, tags: dict[str, float], face: float = 0.4) -> Candidate:
    start = shot_id * 24 * 10
    shot = Shot(shot_id, 1, shot_id, start, start + round(seconds * 24), ShotStatus.KEPT)
    frames = (FramePoint(EPISODE.seconds(start + 12), cuteness(tags, {}), face, tags),)
    stats = tuple(TagStat(t, s, 1.0) for t, s in tags.items())
    return Candidate(EPISODE, shot, shot_id, 1.0, face, (), stats, frames, ("你好",))


def slots(*spec: tuple[str, float]) -> list[Slot]:
    out: list[Slot] = []
    t = 0.0
    for i, (section, length) in enumerate(spec):
        out.append(Slot(i, t, t + length, section, 4))
        t += length
    return out


def song(beat: float = 0.5, length: float = 20.0, *, chorus_only: bool = False) -> SongAnalysis:
    beats = [float(b) for b in np.arange(0, length, beat)]
    sections = [
        Section("verse", 0, 0, 4, 0.0, length / 2, 0.5, 1.0),
        Section("chorus", 1, 4, 8, length / 2, length, 0.5, 2.0),
    ]
    if chorus_only:
        sections = [Section("chorus", 1, 0, 1, 0.0, length, 0.5, 2.0)]
    source = SongSource(Path("song.flac"), 0.0, length, "Test")
    return SongAnalysis("test-song", "Test", source, 60 / beat, beats, beats[::4], [], None, "none",
                        sections, [], (0.0, length), [])  # fmt: skip


def test_cuteness_and_fit() -> None:
    assert cuteness({"blush": 0.8}, {}) > cuteness({"closed_eyes": 0.9}, {})
    assert cuteness({"blush": 0.5}, {"blush": 2.0}) == pytest.approx(1.0)
    (slot,) = slots(("verse", 2.0))
    assert fits(candidate(1, 2.0, {}), slot)
    assert not fits(candidate(1, 1.9, {}), slot)


def test_chorus_gets_the_cutest_and_the_end_the_best_smile() -> None:
    pool = [
        candidate(1, 3, {"smile": 0.3}, face=0.1),
        candidate(2, 3, {"blush": 0.9, "pout": 0.8}),
        candidate(3, 3, {"smile": 0.95, ":d": 0.8}),
        candidate(4, 3, {}, face=0.1),
    ]
    result = arrange(slots(("verse", 2.0), ("verse", 2.0), ("chorus", 1.0), ("chorus", 1.0)), pool)
    by_slot = {a.slot: a.candidate.shot.id for a in result}
    assert (by_slot[2], by_slot[3]) == (2, 3)
    assert sorted(by_slot.values()) == [1, 2, 3, 4]
    assert "best smile" in result[3].reason


def test_story_order_never_puts_a_shot_in_a_slot_it_cannot_fill() -> None:
    long_late, short_early = candidate(9, 4.0, {"smile": 0.9}), candidate(2, 1.2, {"smile": 0.1})
    result = arrange(slots(("verse", 3.5), ("verse", 1.0)), [short_early, long_late])
    assert [a.candidate.shot.id for a in result] == [9, 2]


def test_plan_parse_reports_every_problem_by_path() -> None:
    with pytest.raises(PlanFormatError) as error:
        Plan.from_dict({"name": 1, "series": "s", "character": "c", "song": "x",
                        "slots": [{"start": "a", "end": 1, "shot": 1.5}, 3]})  # fmt: skip
    assert error.value.problems == [
        "name: expected a string",
        "slots[0].shot: expected a shot id (integer)",
        "slots[0].start: expected a number",
        "slots[1]: expected an object",
    ]
    plan = Plan("p", "s", "梦梦", "x", [SlotPlan(0.0, 1.0, 3, "why", 10.0, 11.0, 1.0)])
    assert Plan.from_json(plan.to_json()) == plan
    assert "slots" in json.dumps(PLAN_JSON_SCHEMA)


def test_refine_snaps_to_beats_and_fills_windows() -> None:
    c = candidate(5, 6.0, {"smile": 0.9})
    c = Candidate(c.episode, c.shot, c.scene, c.presence, c.face_height, c.others, c.expressions,
                  (FramePoint(c.start + 3.0, 1.0, 0.4, {"smile": 0.9}),), c.lines)  # fmt: skip
    plan = Plan("p", "s", "梦梦", "test-song", [SlotPlan(0.02, 1.98, 5, "x")])
    refined, notes = refine(plan, song(), {5: c})
    (slot,) = refined.slots
    assert (slot.start, slot.end) == (0.0, 2.0)
    assert slot.source_in == pytest.approx(c.start + 2.0) and slot.speed == 1.0
    assert slot.section == "verse" and slot.episode == "ep01.mkv"
    assert any("moved onto beats" in n for n in notes)
    start, _ = window(candidate(6, 6.0, {}), 2.0)
    assert start >= candidate(6, 6.0, {}).start + 2 / 24 - 1e-9
    assert frame_floor(10.02, Fraction(24)) == 10.0 and frame_ceil(
        10.01, Fraction(24)
    ) == pytest.approx(10 + 1 / 24)


def test_parse_range() -> None:
    assert api.parse_range("0:58-1:21.5") == (58.0, 81.5)
    assert api.parse_range("58.2 - 81.1") == (58.2, 81.1)
    with pytest.raises(api.ApiError):
        api.parse_range("chorus")


@pytest.fixture
def series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TOKIMEKI_HOME", str(tmp_path / "home"))
    root = tmp_path / "series"
    root.mkdir()
    make_clip(root / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(root)
    pipeline.run(ctx, register_episodes(ctx))
    ctx.conn.execute("UPDATE clusters SET name = '梦梦'")
    save_analysis(song(beat=0.25, length=1.0, chorus_only=True))
    return root


def test_context_auto_validate_refine(series: Path) -> None:
    context = api.plan_context(str(series), "梦梦", song="test-song")
    assert len(cast(list[object], context["scenes"])) == 1
    text = json.dumps(context, ensure_ascii=False)
    assert '"shot": 2' in text and '"shot": 1,' not in text
    assert "keyframe" in text and "slots" in text
    result = api.plan_auto(str(series), "梦梦", "test-song", "draft")
    path = Path(str(result["path"]))
    assert path == series / ".tokimeki/mads/draft/plan.json"
    report = api.plan_validate(str(path))
    assert report["ok"], report
    plan = cast(dict[str, object], json.loads(path.read_text()))
    first = cast(list[dict[str, object]], plan["slots"])[0]
    first["shot"], first["end"] = 1, 0.6
    report = api.plan_validate(plan)
    issues = cast(list[dict[str, object]], report["issues"])
    assert {"shot.not_kept", "slot.off_beat"} <= {i["code"] for i in issues}
    assert not report["ok"]


def test_ranges_and_slot_caps() -> None:
    analysis = song(0.5, 20.0)
    excerpt, cut = slots_over(analysis, (10.3, 18.2), budget=100)
    assert (excerpt.start, excerpt.end) == (10.0, 18.0)
    assert [(s.section, s.beats) for s in cut] == [("chorus", 2)] * 8
    _, capped = slots_over(analysis, (10.3, 18.2), budget=3)
    assert [(s.start, s.end) for s in capped] == [(10.0, 14.0), (14.0, 18.0)]
    _, fixed = slots_over(analysis, None, budget=100, fixed={"chorus": 4})
    assert [s.beats for s in fixed if s.section == "chorus"] == [4] * 5


def test_auto_takes_a_range_and_a_cap(series: Path) -> None:
    context = api.plan_context(str(series), "梦梦", song="test-song", span="0-1", max_slots=1)
    slots = cast(list[object], cast(dict[str, object], context["song"])["slots"])
    assert len(slots) == 1
    result = api.plan_auto(str(series), "梦梦", "test-song", "short", span="0-1", max_slots=1)
    assert cast(dict[str, object], result["validation"])["ok"]
