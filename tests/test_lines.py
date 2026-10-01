import sqlite3
import subprocess
from fractions import Fraction
from pathlib import Path

import fakes
import numpy as np
import pytest
from clips import make_clip
from numpy.typing import NDArray

from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.db import open_library
from tokimeki.library.lines import (
    NewLine,
    lines_by_shot,
    list_lines,
    list_parts,
    replace_lines,
    shots_in_parts,
)
from tokimeki.library.records import LineKind, Part, PartKind
from tokimeki.media.audio import main_audio
from tokimeki.media.subtitles import ass_time, parse_ass
from tokimeki.models.vad import FRAME_SECONDS, frame_blocks
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes
from tokimeki.stages.lines import classify, speech_mask, timing_check

ASS = """﻿[Script Info]
Title: test

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:01.00,0:00:00.00,*Default,NTP,0000,0000,0000,,------------
Dialogue: 0,0:00:01.50,0:00:03.00,*Default,NTP,0000,0000,0000,,果酱可以吗, 茉茉
Dialogue: 0,0:00:04.00,0:00:05.25,*Default,NTP,0000,0000,0000,,{\\an8}第一行\\N第二行
Dialogue: 0,0:01:00.00,0:01:04.00,opjp,NTP,0000,0000,0000,,{\\fad(100,100)}信じてね
Dialogue: 0,0:01:00.00,0:01:04.00,opcn,NTP,0000,0000,0000,,相信我
Dialogue: 0,0:01:10.00,0:01:20.00,edcn,NTP,0000,0000,0000,,黄牌
Dialogue: 0,0:00:00.50,0:00:01.00,title,NTP,0000,0000,0000,,{\\pos(1,2)}第1话
"""


def test_parse_ass() -> None:
    events = parse_ass(ASS)
    assert len(events) == 6
    assert (events[0].start, events[0].end, events[0].style) == (1.5, 3.0, "*Default")
    assert events[0].text == "果酱可以吗, 茉茉"
    assert events[1].text == "第一行\n第二行"
    assert events[2].text == "信じてね"
    assert ass_time("1:02:03.45") == pytest.approx(3723.45)


def test_classify_and_parts() -> None:
    lines, lyrics = classify(parse_ass(ASS))
    kinds = [line.kind for line in lines]
    assert kinds.count(LineKind.DIALOGUE) == 2
    assert kinds.count(LineKind.LYRICS) == 3
    assert kinds.count(LineKind.OTHER) == 1
    assert len(lyrics[PartKind.OPENING]) == 2
    assert len(lyrics[PartKind.ENDING]) == 1


def _speech(lines: list[NewLine], shift: float, frames: int) -> NDArray[np.float32]:
    shifted = [NewLine(x.start + shift, x.end + shift, x.kind, x.style, x.text) for x in lines]
    return speech_mask(shifted, frames) * 0.8 + 0.1


def _dialogue(count: int) -> list[NewLine]:
    rng = np.random.default_rng(1)
    starts = np.cumsum(rng.uniform(2.0, 6.0, count))
    return [
        NewLine(float(s), float(s + rng.uniform(0.8, 2.5)), LineKind.DIALOGUE, "Default", "x")
        for s in starts
    ]


def test_timing_check_finds_and_applies_a_real_offset() -> None:
    dialogue = _dialogue(60)
    frames = int((dialogue[-1].end + 10) / FRAME_SECONDS)
    check = timing_check(dialogue, _speech(dialogue, 0.8, frames))
    assert check.offset == pytest.approx(0.8, abs=FRAME_SECONDS)
    assert check.applied
    assert check.onsets_matched > 0.9


def test_timing_check_leaves_well_timed_subtitles_alone() -> None:
    dialogue = _dialogue(60)
    frames = int((dialogue[-1].end + 10) / FRAME_SECONDS)
    check = timing_check(dialogue, _speech(dialogue, 0.03, frames))
    assert abs(check.offset) < 0.1
    assert not check.applied


def test_vad_frames_carry_context() -> None:
    audio = np.arange(1100, dtype=np.float32)
    (block,) = frame_blocks(audio)
    assert block.shape == (3, 576)
    assert block[1, :64].tolist() == audio[448:512].tolist()
    assert block[2, 64 + 75] == 1099.0
    assert not block[2, 64 + 76 :].any()


def test_lines_attach_to_overlapping_shots() -> None:
    conn = open_library(":memory:")
    episode = episode_db.register_episode(conn, "ep.mkv", 1280, 720, Fraction(24), 240)
    a, b, c = shot_db.replace_shots(conn, episode.id, [(0, 48), (48, 120), (120, 240)])
    line = NewLine(1.5, 2.5, LineKind.DIALOGUE, "Default", "hi")
    opening = Part(episode.id, PartKind.OPENING, 5.5, 9.0)
    replace_lines(conn, episode.id, [line], [opening])
    by_shot = lines_by_shot(conn, episode.id)
    assert set(by_shot) == {a.id, b.id}
    assert by_shot[a.id][0].text == "hi"
    assert shots_in_parts(conn, episode.id) == {c.id: PartKind.OPENING}
    assert list_parts(conn, episode.id) == [opening]


def test_main_audio_skips_commentary(tmp_path: Path) -> None:
    clip = tmp_path / "audio.mkv"
    sine = "sine=frequency=440:duration=1"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", sine,
         "-f", "lavfi", "-i", sine, "-map", "0", "-map", "1", "-c:a", "flac",
         "-metadata:s:a:0", "title=Commentary", "-disposition:a:0", "default",
         "-disposition:a:1", "0", str(clip)],
        check=True,
    )  # fmt: skip
    assert main_audio(clip).index == 1


def test_stage_imports_subtitles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"], seconds=40)
    (tmp_path / "ep01.ass").write_text(ASS, encoding="utf-8")
    make_clip(tmp_path / "ep02.mkv", ["color=c=red"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes, until="lines")
    first, second = episodes
    assert len(list_lines(ctx.conn, first.id, LineKind.DIALOGUE)) == 2
    assert [p.kind for p in list_parts(ctx.conn, first.id)] == [PartKind.OPENING, PartKind.ENDING]
    assert list_lines(ctx.conn, second.id) == []
    assert isinstance(ctx.conn, sqlite3.Connection)
