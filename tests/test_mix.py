from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_plan import song

from tokimeki.library.records import Episode, Line, LineKind
from tokimeki.mad.mix import (
    DUCK_ATTACK,
    DUCK_RELEASE,
    RATE,
    TARGET_PEAK,
    VOICE_ABOVE,
    VoiceClip,
    duck_gain,
    level_db,
    mix,
)
from tokimeki.mad.plan import Plan, SlotPlan
from tokimeki.mad.voices import VoiceLine, pick_voice, voiced
from tokimeki.song.gaps import gaps, quiet_bars, sung_overlap, sung_spans
from tokimeki.song.lyrics import LyricLine

EPISODE = Episode(1, "ep01.mkv", 1920, 1080, Fraction(24), 24 * 600)


def test_ducking_eases_down_before_a_line_and_back_after() -> None:
    gain = duck_gain(RATE * 4, [(1.0, 2.0, 9.0)])
    deep = 10 ** (-9 / 20)
    assert gain[RATE // 2] == 1.0 and gain[round(1.5 * RATE)] == pytest.approx(deep)
    assert gain[round((1.0 - DUCK_ATTACK / 2) * RATE)] == pytest.approx(10 ** (-4.5 / 20), rel=1e-3)
    assert gain[round((2.0 + DUCK_RELEASE) * RATE) + 10] == 1.0
    assert np.all(np.diff(gain[:RATE]) <= 1e-12)


def test_voice_sits_above_the_ducked_song_and_under_the_peak() -> None:
    t = np.arange(RATE * 4) / RATE
    music = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    speech = np.stack([0.05 * np.sin(2 * np.pi * 300 * t[:RATE])] * 2).astype(np.float32)
    out = mix(music, [VoiceClip(1.5, speech, gain=0.0, duck=9.0)])
    assert np.abs(out).max() == pytest.approx(TARGET_PEAK)
    head = slice(RATE // 2, RATE)
    scale = np.abs(out[:, head]).max() / np.abs(music[:, head]).max()
    voice_part = out[:, round(1.7 * RATE) : round(2.3 * RATE)] / scale
    ducked = music[:, round(1.7 * RATE) : round(2.3 * RATE)] * 10 ** (-9 / 20)
    voice_db = level_db((voice_part - ducked).astype(np.float32))
    assert voice_db == pytest.approx(level_db(music) - 9 + VOICE_ABOVE, abs=0.5)
    assert np.abs(out[:, head] / scale - music[:, head]).max() < 1e-5
    loud = mix(music, [VoiceClip(1.5, speech * 20, gain=12.0, duck=0.0)])
    assert np.abs(loud).max() == pytest.approx(TARGET_PEAK)


def _sung() -> list[LyricLine]:
    return [LyricLine(3.0, 8.0, "a", "ja"), LyricLine(3.0, 8.0, "甲", "zh"),
            LyricLine(12.0, 18.0, "b", "ja")]  # fmt: skip


def test_gaps_are_where_nobody_sings() -> None:
    analysis = replace(song(0.5, 20.0), lyrics=_sung())
    assert sung_spans(analysis) == [(3.0, 8.0), (12.0, 18.0)]
    found = [(g.start, g.end, g.where) for g in gaps(analysis, 0.0, 20.0)]
    assert found == [(0.0, 3.0, "intro"), (8.0, 12.0, "between"), (18.0, 20.0, "outro")]
    assert [g.where for g in gaps(analysis, 4.0, 15.0)] == ["between"]
    assert sung_overlap(analysis, 7.0, 13.0) == pytest.approx(2.0)
    analysis.energy = [0.5, 0.1, 0.9, 0.8, 0.7, 0.15, 0.95, 0.85, 0.75, 0.2]
    assert quiet_bars(analysis, 0.0, 20.0) == [0.0, 10.0, 18.0]


def _line(i: int, start: float, end: float) -> Line:
    return Line(i, 1, start, end, LineKind.DIALOGUE, "Default", f"line {i}")


def test_the_draft_puts_her_line_in_the_intro_gap() -> None:
    analysis = replace(song(0.5, 20.0), lyrics=_sung())
    plan = Plan("t", "/s", "梦梦", "test-song", [SlotPlan(0.0, 4.0, 7), SlotPlan(4.0, 20.0, 8)])
    lines = [
        VoiceLine(_line(1, 100.0, 102.0), EPISODE, [5], ["梦梦"]),
        VoiceLine(_line(2, 200.0, 201.8), EPISODE, [7], ["梦梦", "娜娜"]),
        VoiceLine(_line(3, 300.0, 302.2), EPISODE, [9], ["娜娜"]),
        VoiceLine(_line(4, 400.0, 405.0), EPISODE, [7], ["梦梦"]),
    ]
    picked = pick_voice(plan, analysis, lines, "梦梦", lambda x: (x.line.start, x.line.end))
    assert picked is not None
    assert (picked.line, picked.at) == (2, 0.3)
    longer = pick_voice(plan, analysis, lines, "梦梦", lambda x: (x.line.start - 1, x.line.end))
    assert longer is not None and longer.line == 1


def test_voiced_trims_to_the_voice() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    samples[4000:9000] = 0.3
    found = voiced(samples, 16000)
    assert found == pytest.approx((0.25, 0.57))
    assert voiced(np.zeros(1600, dtype=np.float32), 16000) is None


def test_voiced_keeps_to_the_runs_under_the_subtitle() -> None:
    samples = np.zeros(32000, dtype=np.float32)
    samples[0:3200] = 0.3
    samples[8000:20000] = 0.3
    samples[20800:24000] = 0.3
    assert voiced(samples, 16000, (0.55, 1.4)) == pytest.approx((0.5, 1.5))
    assert voiced(samples, 16000) == pytest.approx((0.0, 1.5))
