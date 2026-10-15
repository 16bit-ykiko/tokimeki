"""Speakers on a synthetic episode (see `voices`)."""

from pathlib import Path

import numpy as np
import pytest
from voices import HIGH, LOW, NANA_LINES, ToneEncoder, series, tone

from tokimeki.anime import repeats
from tokimeki.anime.repeats import Repeat
from tokimeki.cli import main
from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.lines import list_lines
from tokimeki.library.records import Line, LineKind, ShotStatus
from tokimeki.library.series import Context
from tokimeki.voice import speakers
from tokimeki.voice.clips import CLIP_RATE, cut_lines, episode_cuts
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


def test_lines_are_cut_to_their_voice_with_a_status() -> None:
    rate = 16000
    stem = tone(
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


def test_lines_in_recurring_shots_are_not_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = series(tmp_path)
    episode = episode_db.list_episodes(ctx.conn)[0]
    nana_shot = shot_db.list_shots(ctx.conn, episode.id, ShotStatus.KEPT)[1]
    before = {c.line.id for c in episode_cuts(ctx, episode)[0]}

    def marked(_: Context) -> dict[int, Repeat]:
        return {nana_shot.id: Repeat("preview", (99,))}

    monkeypatch.setattr(repeats, "update", marked)
    after = {c.line.id for c in episode_cuts(ctx, episode)[0]}
    assert after < before and len(before - after) == len(NANA_LINES)


def test_stage_clusters_voices_and_names_them_from_close_ups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = series(tmp_path)
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
    ctx = series(tmp_path)
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
    ctx = series(tmp_path)
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
    ctx = series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    speakers.run(ctx, episode_db.list_episodes(ctx.conn))
    assert speakers.fresh(ctx.conn, 1, speakers.NAME, speakers.AFTER)
    episode_db.mark_stage_done(ctx.conn, 1, "lines", {})
    assert not speakers.fresh(ctx.conn, 1, speakers.NAME, speakers.AFTER)


def test_cli_lists_accepts_and_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ctx = series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    root = str(tmp_path)
    assert main(["voice", "speakers", root]) == 0
    assert "looks like 梦梦" in capsys.readouterr().out
    assert main(["voice", "accept", root]) == 0
    line = clean_voices(ctx.conn, find_speaker(ctx.conn, "梦梦").id)[0].line_id
    assert main(["voice", "move", root, "--line", str(line), "--nobody"]) == 0
    assert main(["voice", "fixes", root]) == 0
    assert f"line {line}: nobody" in capsys.readouterr().out
    assert main(["voice", "move", root, "--line", "999", "--to", "梦梦"]) == 1
    assert main(["voice", "page", root]) == 0
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
