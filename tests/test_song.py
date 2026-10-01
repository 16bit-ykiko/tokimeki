from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from tokimeki.media.cue import cue_time, parse_cue
from tokimeki.models.beats import chunk_starts, log_mel, mel_filters, pick_peaks, snap_downbeats
from tokimeki.song.analysis import SongSource, lyrics_share, song_id
from tokimeki.song.lyrics import LyricLine, parse_lrc
from tokimeki.song.slots import Excerpt, first_chorus, make_slots
from tokimeki.song.structure import (
    Section,
    bar_grid,
    best_lag,
    boundaries,
    label_sections,
    merge_runs,
    novelty,
    vocal_share,
)

CUE = """﻿REM GENRE Anime
PERFORMER "Momo"
TITLE "Single"
FILE "CD.flac" WAVE
  TRACK 01 AUDIO
    TITLE "MORE&MORE"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "B-side"
    INDEX 00 03:49:71
    INDEX 01 03:52:09
  TRACK 03 AUDIO
    TITLE "MORE&MORE (Instrumental)"
    INDEX 01 08:12:62
"""


def test_cue_tracks_end_at_the_next_pregap(tmp_path: Path) -> None:
    sheet = parse_cue(CUE, tmp_path)
    assert sheet.file == tmp_path / "CD.flac"
    first, second, third = sheet.tracks
    assert (first.title, first.performer, first.start) == ("MORE&MORE", "Momo", 0.0)
    assert first.end == pytest.approx(cue_time("03:49:71"))
    assert second.start == pytest.approx(232.12)
    assert third.end is None


def test_log_mel_matches_torchaudio() -> None:
    t = np.arange(22050 * 5) / 22050
    rng = np.random.default_rng(0)
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 3000 * t) * (t % 0.5 < 0.1)
        + 0.05 * rng.normal(size=t.size)
    ).astype(np.float32)
    reference = np.load(Path(__file__).parent / "data" / "torchaudio_logmel.npz")["ref"]
    out = log_mel(torch.tensor(signal), torch.tensor(mel_filters(), dtype=torch.float32))
    assert np.abs(out.numpy() - reference).max() < 1e-3


def test_peaks_and_downbeats() -> None:
    logits = np.full(200, -5.0, dtype=np.float32)
    logits[[10, 11, 60, 110, 160]] = [2.0, 2.0, 1.0, 3.0, 0.5]
    beats = pick_peaks(logits)
    assert beats.tolist() == pytest.approx([10.5 / 50, 60 / 50, 110 / 50, 160 / 50])
    assert snap_downbeats(beats, np.array([1.25, 2.1])).tolist() == pytest.approx([1.2, 2.2])


def test_chunks_cover_long_pieces() -> None:
    assert chunk_starts(1000) == [-6]
    starts = chunk_starts(4000)
    assert starts[0] == -6 and starts[-1] == 4000 - 1494


def test_bar_grid_follows_the_majority_phase() -> None:
    beats = np.arange(32, dtype=np.float64) * 0.5
    downbeats = np.array([1.0, 3.0, 5.0, 6.5, 9.0])
    assert bar_grid(beats, downbeats)[:3].tolist() == [1.0, 3.0, 5.0]


def test_vocal_share_and_alignment() -> None:
    backing = np.ones((10, 4))
    mix = backing.copy()
    mix[5:, 1] += 3.0
    band = np.array([False, True, True, False])
    share = vocal_share(mix, backing, band)
    assert share[:5].tolist() == [0.0] * 5
    assert share[5] == pytest.approx(0.6)
    a = np.zeros(100)
    a[[20, 50, 70]] = 1.0
    assert best_lag(a, np.roll(a, -3), 10) == 3


def _song_like() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    blocks = [
        ("intro", 8, 0.0, 1.0),
        ("verse", 16, 0.5, 2.0),
        ("chorus", 16, 0.5, 4.0),
        ("verse", 16, 0.5, 2.0),
        ("chorus", 16, 0.5, 4.0),
        ("outro", 8, 0.0, 1.0),
    ]
    centres = {"intro": 0, "verse": 1, "chorus": 2, "outro": 0}
    features: list[np.ndarray] = []
    vocal: list[float] = []
    loud: list[float] = []
    for name, bars, sung, level in blocks:
        for _ in range(bars):
            f = rng.normal(0, 0.1, 6)
            f[centres[name]] += 3.0
            features.append(f)
            vocal.append(sung)
            loud.append(level)
    return np.array(features), np.array(vocal), np.array(loud)


def test_sections_are_found_and_named() -> None:
    features, vocal, loud = _song_like()
    bars = np.arange(len(features), dtype=np.float64) * 2.0
    starts = boundaries(novelty(features), vocal >= 0.2)
    assert starts == [0, 8, 24, 40, 56, 72]
    sections = merge_runs(label_sections(starts, features, vocal, loud, bars, len(bars) * 2.0))
    assert [s.label for s in sections] == ["intro", "verse", "chorus", "verse", "chorus", "outro"]
    assert sections[2].group == sections[4].group


def _section(label: str, start: float, end: float) -> Section:
    return Section(label, 0, int(start / 2), int(end / 2), start, end, 0.5, 1.0)


def test_excerpt_runs_to_the_first_chorus_within_limits() -> None:
    sections = [_section("intro", 0, 16), _section("verse", 16, 48), _section("chorus", 48, 80),
                _section("verse", 80, 112), _section("chorus", 112, 144)]  # fmt: skip
    excerpt = first_chorus(sections)
    assert (excerpt.start, excerpt.end) == (0, 80)
    long = [_section("intro", 0, 30), _section("verse", 30, 70), _section("chorus", 70, 110)]
    assert first_chorus(long).start == 30


def test_slots_slow_down_to_fit_the_shots() -> None:
    sections = [_section("intro", 0, 8), _section("verse", 8, 24), _section("chorus", 24, 40)]
    beats = [float(b) for b in np.arange(0, 40, 0.5)]
    excerpt = Excerpt(0, 40, sections)
    fast = make_slots(beats, 120.0, excerpt, 100)
    assert len(fast) == 4 + 8 + 16
    fitted = make_slots(beats, 120.0, excerpt, 22)
    assert {s.section: s.beats for s in fitted} == {"intro": 8, "verse": 8, "chorus": 2}
    assert len(fitted) == 22
    tight = make_slots(beats, 120.0, excerpt, 20)
    assert {s.section: s.beats for s in tight}["chorus"] == 4
    assert tight[0].start == 0 and tight[-1].end == 40
    assert all(b.start == a.end for a, b in pairwise(tight))


def test_lrc_lines_end_where_the_next_begins() -> None:
    text = (
        "[ti:MORE&MORE]\n[offset:+500]\n[00:12.00]もっと<00:12.50>もっと\n"
        "[00:15.30][01:02.00]好きよ\n[00:20.00]\n"
    )
    lines = parse_lrc(text)
    assert [(x.start, x.end, x.text) for x in lines] == [
        (11.5, 14.8, "もっともっと"),
        (14.8, 19.5, "好きよ"),
        (61.5, 66.5, "好きよ"),
    ]


def test_lyrics_give_the_vocal_line() -> None:
    lines = [LyricLine(2.0, 4.0, "a")]
    share = lyrics_share(lines, np.array([0.0, 2.0, 4.0]), 6.0)
    assert share.tolist() == [0.0, 1.0, 0.0]


def test_songs_without_a_vocal_line_get_an_intro_from_loudness() -> None:
    features, _, loud = _song_like()
    bars = np.arange(len(features), dtype=np.float64) * 2.0
    ones = np.ones(len(features))
    starts = boundaries(novelty(features), ones >= 0.2)
    sections = merge_runs(
        label_sections(starts, features, ones, loud, bars, len(bars) * 2.0, vocal_known=False)
    )
    assert sections[0].label == "intro" and sections[-1].label == "outro"
    assert "chorus" in [s.label for s in sections]


def test_song_ids_are_readable_and_stable(tmp_path: Path) -> None:
    source = SongSource(tmp_path / "a.flac", 0.0, 100.0, "MORE&MORE", 1)
    assert song_id(source).startswith("more-more-")
    assert song_id(source) == song_id(source)
    assert song_id(SongSource(tmp_path / "a.flac", 232.1, 100.0, "そして君と", 2)).startswith(
        "song-"
    )
