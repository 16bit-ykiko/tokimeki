from fractions import Fraction

import pytest

from tokimeki.library.records import Episode, Shot, ShotStatus, TagStat
from tokimeki.mad.arrange import ArrangementError, arrange, fits, to_clips
from tokimeki.mad.candidates import Candidate, FramePoint, cuteness
from tokimeki.mad.model_arranger import arrange_with_model, build_prompt, parse_answer
from tokimeki.mad.plan import Clip, Plan, PlanSlot, Preferences, SongRef, problems

EPISODE = Episode(1, "ep01.mkv", 1920, 1080, Fraction(24), 24 * 600)


def candidate(shot_id: int, seconds: float, tags: dict[str, float], face: float = 0.4) -> Candidate:
    start = shot_id * 24 * 10
    shot = Shot(shot_id, 1, shot_id, start, start + round(seconds * 24), ShotStatus.KEPT)
    frames = (FramePoint(EPISODE.seconds(start + 12), cuteness(tags, {}), face, tags),)
    stats = tuple(TagStat(t, s, 1.0) for t, s in tags.items())
    return Candidate(EPISODE, shot, shot_id, 1.0, face, (), stats, frames, ("你好",))


def slots(*spec: tuple[str, float]) -> list[PlanSlot]:
    out: list[PlanSlot] = []
    t = 0.0
    for i, (section, length) in enumerate(spec):
        out.append(PlanSlot(i, t, t + length, section, 4))
        t += length
    return out


def test_cuteness_weighs_blush_over_closed_eyes() -> None:
    assert cuteness({"blush": 0.8}, {}) > cuteness({"closed_eyes": 0.9}, {})
    assert cuteness({"blush": 0.5}, {"blush": 2.0}) == pytest.approx(1.0)


def test_fits_allows_slowing_to_ninety_percent() -> None:
    (slot,) = slots(("verse", 2.0))
    assert fits(candidate(1, 2.0, {}), slot)
    assert not fits(candidate(1, 1.8, {}), slot)


def test_chorus_gets_the_cutest_and_the_end_the_best_smile() -> None:
    pool = [
        candidate(1, 3, {"smile": 0.3}, face=0.1),
        candidate(2, 3, {"blush": 0.9, "pout": 0.8}),
        candidate(3, 3, {"smile": 0.95, ":d": 0.8}),
        candidate(4, 3, {}, face=0.1),
    ]
    plan_slots = slots(("verse", 2.0), ("verse", 2.0), ("chorus", 1.0), ("chorus", 1.0))
    result = arrange(plan_slots, pool)
    by_slot = {a.slot: a.candidate.shot.id for a in result}
    assert by_slot[3] == 3
    assert by_slot[2] == 2
    assert sorted(by_slot.values()) == [1, 2, 3, 4]
    assert "best smile" in result[3].reason


def test_arrange_refuses_when_shots_are_too_short() -> None:
    with pytest.raises(ArrangementError, match="long enough"):
        arrange(slots(("verse", 3.0)), [candidate(1, 1.0, {})])


def _plan(clips: list[Clip]) -> Plan:
    song = SongRef("s", "Song", "s.flac", 0.0, 0.0, 4.0, 120.0)
    plan_slots = slots(("verse", 2.0), ("chorus", 2.0))
    return Plan("t", Preferences("梦梦", "s.flac"), song, plan_slots, clips, "heuristic")


def test_plan_round_trips_and_reports_problems() -> None:
    plan = _plan([Clip(0, "ep01.mkv", 1, "a", 10.0, 1.0, 10.5), Clip(1, "ep01.mkv", 2, "b")])
    assert Plan.from_json(plan.to_json()) == plan
    assert problems(plan) == []
    bad = _plan([Clip(0, "ep01.mkv", 1, "a", speed=1.3), Clip(0, "ep01.mkv", 1, "b")])
    found = " ".join(problems(bad))
    assert "slot 0 has 2 clips" in found and "slot 1 has 0" in found
    assert "used more than once" in found and "speed 1.300" in found


class FakeModel:
    name = "fake"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.answers.pop(0)


def test_model_answers_are_checked_and_retried() -> None:
    pool = [candidate(1, 3, {"smile": 0.5}), candidate(2, 3, {"blush": 0.9})]
    plan_slots = slots(("verse", 2.0), ("chorus", 1.0))
    first = '{"slot": 0, "shot": 1, "reason": "x"}'
    bad = f'{{"assignments": [{first}, {{"slot": 1, "shot": 1, "reason": "y"}}]}}'
    good = f'Sure: {{"assignments": [{first}, {{"slot": 1, "shot": 2, "reason": "y"}}]}}'
    model = FakeModel([bad, good])
    result = arrange_with_model(model, plan_slots, pool, "梦梦", "Song")
    assert [(a.slot, a.candidate.shot.id) for a in result] == [(0, 1), (1, 2)]
    assert "used twice" in model.prompts[1]
    assert to_clips(result)[1] == Clip(1, "ep01.mkv", 2, "y")
    with pytest.raises(ArrangementError, match="slots left empty"):
        parse_answer('{"assignments": []}', plan_slots, pool)


def test_prompt_lists_slots_and_shots() -> None:
    system, user = build_prompt(
        slots(("chorus", 1.0)), [candidate(7, 2, {"blush": 0.9})], "梦梦", "MORE&MORE", "more blush"
    )
    assert '"assignments"' in system
    assert "slot 0: chorus" in user and "shot 7:" in user and "blush 0.90" in user
    assert "Editor's request: more blush" in user and "你好" in user
