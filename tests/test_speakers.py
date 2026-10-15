"""Speakers on a synthetic episode: two tones stand for two voices, faces for the cast."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tokimeki.anime.stems import stem_path
from tokimeki.cli import main
from tokimeki.library import cast as cast_db
from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.lines import NewLine, list_lines, replace_lines
from tokimeki.library.records import Box, Line, LineKind, Part, PartKind, Rating
from tokimeki.library.series import Context, open_series
from tokimeki.media.audio import encode_audio
from tokimeki.voice import speakers
from tokimeki.voice.clips import CLIP_RATE, cut_lines
from tokimeki.voice.embedding import DIMENSIONS, SpeakerEncoder
from tokimeki.voice.page import render_page
from tokimeki.voice.review import review, write_clips
from tokimeki.voice.speakers import Evidence, Match
from tokimeki.voice.store import (
    VoiceStatus,
    clean_voices,
    create_speaker,
    episode_voices,
    find_speaker,
    list_speakers,
)

FPS = 24
LOW, HIGH = 220.0, 880.0
MOMO_LINES = [(1.0, 2.5), (3.0, 4.6), (5.0, 6.2), (7.0, 8.4)]
NANA_LINES = [(11.0, 12.5), (13.0, 14.4), (15.0, 16.3), (17.0, 18.2)]
OVERLAPPING = [(21.0, 22.5), (22.0, 23.5)]
SHORT = (25.0, 25.3)
IN_OPENING = (27.0, 28.0)


class ToneEncoder:
    """Hears which of the two tones a clip holds."""

    def embed(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        spectrum = np.abs(np.fft.rfft(samples))
        peak = np.argmax(spectrum) * CLIP_RATE / len(samples)
        rng = np.random.default_rng(len(samples))
        base = np.array([1.0, 0.0, 0.0]) if abs(peak - LOW) < abs(peak - HIGH) else np.eye(3)[1]
        return (base + rng.normal(0, 0.05, 3)).astype(np.float32)

    def close(self) -> None:
        pass


def _tone(
    rate: int, seconds: float, spans: list[tuple[tuple[float, float], float]]
) -> NDArray[np.float32]:
    audio = np.zeros(round(rate * seconds), dtype=np.float32)
    for (a, b), freq in spans:
        t = np.arange(round(a * rate), round(b * rate))
        audio[t] = 0.3 * np.sin(2 * np.pi * freq * t / rate)
    return audio


def _faces(ctx: Context, shot_id: int, frames: int, cluster: int, height: float) -> None:
    safe = Rating(1, 0, 0, 0)
    shot_db.keep_shot(ctx.conn, shot_id, [shot_db.RatedFrame(i, safe, []) for i in range(frames)])
    box = Box(0.4, 0.5 - height / 2, 0.6, 0.5 + height / 2)
    new = [
        cast_db.NewFace(f.id, box, 0.9, np.ones(4, dtype=np.float32))
        for f in shot_db.list_frames(ctx.conn, shot_id)
    ]
    cast_db.assign_faces(ctx.conn, cast_db.add_faces(ctx.conn, new), cluster)


def _series(tmp_path: Path) -> Context:
    """Momo alone in close-up for 0-10 s, Nana for 10-20 s, nobody named after that."""
    ctx = open_series(tmp_path)
    seconds = 30
    episode = episode_db.register_episode(
        ctx.conn, "ep01.mkv", 1280, 720, Fraction(FPS), FPS * seconds
    )
    spans = [(0, 10 * FPS), (10 * FPS, 20 * FPS), (20 * FPS, 30 * FPS)]
    momo_shot, nana_shot, rest = shot_db.replace_shots(ctx.conn, episode.id, spans)
    momo = cast_db.create_cluster(ctx.conn, "梦梦")
    nana = cast_db.create_cluster(ctx.conn, "娜娜")
    _faces(ctx, momo_shot.id, 4, momo, 0.5)
    _faces(ctx, nana_shot.id, 4, nana, 0.5)
    shot_db.keep_shot(ctx.conn, rest.id, [])
    timed = [*MOMO_LINES, *NANA_LINES, *OVERLAPPING, SHORT, IN_OPENING]
    lines = [
        NewLine(a, b, LineKind.DIALOGUE, "Default", f"line {i}") for i, (a, b) in enumerate(timed)
    ]
    opening = Part(episode.id, PartKind.OPENING, 26.5, 29.0)
    replace_lines(ctx.conn, episode.id, lines, [opening])
    voices = [(s, LOW) for s in MOMO_LINES] + [(s, HIGH) for s in NANA_LINES]
    voices += [(s, LOW) for s in (*OVERLAPPING, SHORT, IN_OPENING)]
    stem = _tone(44100, seconds, voices)
    encode_audio(np.stack([stem, stem]), 44100, stem_path(ctx.paths, episode.id))
    for stage in ("lines", "voice"):
        episode_db.mark_stage_done(ctx.conn, episode.id, stage, {})
    return ctx


def test_lines_are_cut_to_their_voice_with_a_status() -> None:
    rate = 16000
    stem = _tone(
        rate, 10, [((1.0, 2.0), LOW), ((3.0, 3.2), LOW), ((5.0, 6.5), LOW), ((6.0, 7.0), HIGH)]
    )
    timed = [(1.0, 2.0), (3.0, 3.2), (5.0, 6.5), (6.0, 7.0), (7.5, 8.2), (8.5, 9.5)]
    lines = [_line(i, a, b) for i, (a, b) in enumerate(timed)]
    cuts = cut_lines(lines, [(0.0, 10.0)], [(8.4, 10.0)], stem, rate)
    statuses = [c.status for c in cuts]
    assert statuses == [
        VoiceStatus.CLEAN,
        VoiceStatus.SHORT,
        VoiceStatus.OVERLAP,
        VoiceStatus.OVERLAP,
        VoiceStatus.SILENT,
    ]
    assert cuts[0].start == pytest.approx(0.94, abs=0.02)
    assert cuts[0].end == pytest.approx(2.12, abs=0.02)


def _line(i: int, a: float, b: float) -> Line:
    return Line(i + 1, 1, a, b, LineKind.DIALOGUE, "Default", "x")


def test_stage_clusters_voices_and_names_them_from_close_ups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    episodes = episode_db.list_episodes(ctx.conn)
    assert speakers.run(ctx, episodes) == 2
    statuses = [v.status for v in episode_voices(ctx.conn, 1)]
    assert statuses.count(VoiceStatus.CLEAN) == 8
    assert statuses.count(VoiceStatus.OVERLAP) == 2
    assert statuses.count(VoiceStatus.SHORT) == 1
    assert len(statuses) == 11
    found = speakers.evidence(ctx)
    assert sorted(e.suggestion or "" for e in found.values()) == ["娜娜", "梦梦"]
    assert sorted(speakers.accept(ctx))[0].endswith("娜娜")
    momo = find_speaker(ctx.conn, "梦梦")
    assert (momo.lines, round(momo.seconds, 1)) == (4, 6.4)
    assert speakers.run(ctx, episodes) == 0


def test_names_and_moves_survive_a_redo_and_reclustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    episodes = episode_db.list_episodes(ctx.conn)
    speakers.run(ctx, episodes)
    speakers.accept(ctx)
    moved = clean_voices(ctx.conn, find_speaker(ctx.conn, "梦梦").id)[0].line_id
    assert speakers.move(ctx, [moved], "娜娜") == [f"line {moved}: 梦梦 -> 娜娜"]
    speakers.recluster(ctx)
    episode_db.mark_stage_done(ctx.conn, 1, "voice", {})
    speakers.run(ctx, episodes)
    counts = {s.label: s.lines for s in list_speakers(ctx.conn)}
    assert counts == {"梦梦": 3, "娜娜": 5}
    unnamed = create_speaker(ctx.conn)
    with pytest.raises(ValueError, match="no name"):
        speakers.move(ctx, [moved], f"#{unnamed}")


def test_suggestion_trusts_seed_majority_then_on_screen_lift() -> None:
    assert Evidence(5, {"梦梦": 3, "梨斗": 1}).suggestion == "梦梦"
    assert Evidence(5, {"梦梦": 1}, [Match("娜娜", 0.6, 3.0)]).suggestion == "娜娜"
    split = Evidence(9, {"梦梦": 2, "梨斗": 2}, [Match("梨斗", 0.5, 2.0)])
    assert split.suggestion == "梨斗"
    assert Evidence(2, {}, [Match("娜娜", 0.9, 5.0)]).suggestion is None
    assert Evidence(9, {}, [Match("娜娜", 0.9, 1.1)]).suggestion is None


def test_matches_weigh_against_how_often_a_character_is_on_screen() -> None:
    lines = {1: {"a": 1.0, "b": 1.0}, 2: {"a": 1.0}, 3: {"a": 1.0}, 4: {"b": 1.0}}
    found = speakers.matches(lines, {7: [1, 4]})[7]
    assert [m.name for m in found] == ["b", "a"]
    assert found[0].lift == pytest.approx(2.0)


def test_review_and_page_list_speakers_with_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    speakers.run(ctx, episode_db.list_episodes(ctx.conn))
    data = review(ctx, typical=2, doubtful=1)
    assert data.statuses["clean"] == 8
    assert sorted(s.suggestion or "" for s in data.speakers) == ["娜娜", "梦梦"]
    assert all(len(s.typical) == 2 and len(s.doubtful) == 1 for s in data.speakers)
    assert write_clips(ctx, data) == 6
    assert write_clips(ctx, data) == 0
    page = render_page("~/series", data)
    assert page.count("<audio") == 6 and "梦梦" in page
    assert len(list_lines(ctx.conn, 1)) == 12


def test_a_newer_stem_or_lines_run_makes_speakers_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    speakers.run(ctx, episode_db.list_episodes(ctx.conn))
    assert speakers.fresh(ctx.conn, 1, speakers.NAME, speakers.AFTER)
    episode_db.mark_stage_done(ctx.conn, 1, "lines", {})
    assert not speakers.fresh(ctx.conn, 1, speakers.NAME, speakers.AFTER)


def test_cli_lists_accepts_and_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ctx = _series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    series = str(tmp_path)
    assert main(["voice", "speakers", series]) == 0
    assert "looks like 梦梦" in capsys.readouterr().out
    assert main(["voice", "accept", series]) == 0
    line = clean_voices(ctx.conn, find_speaker(ctx.conn, "梦梦").id)[0].line_id
    assert main(["voice", "move", series, "--line", str(line), "--nobody"]) == 0
    assert main(["voice", "fixes", series]) == 0
    assert f"line {line}: nobody" in capsys.readouterr().out
    assert main(["voice", "move", series, "--line", "999", "--to", "梦梦"]) == 1
    assert main(["voice", "page", series]) == 0
    assert (tmp_path / ".tokimeki/voice/page/index.html").exists()


@pytest.mark.gpu
def test_speaker_encoder_runs_on_the_gpu() -> None:
    encoder = SpeakerEncoder()
    try:
        t = np.arange(CLIP_RATE * 2) / CLIP_RATE
        embedding = encoder.embed((0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
    finally:
        encoder.close()
    assert embedding.shape == (DIMENSIONS,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-3)
