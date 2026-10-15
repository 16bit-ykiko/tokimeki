"""Japanese text for every line with a voice, from the voice stem (the subtitles are Chinese)."""

import logging
from collections.abc import Sequence

from tokimeki.anime.stems import stem_path
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_params
from tokimeki.library.records import Episode
from tokimeki.library.series import Context
from tokimeki.media.audio import decode_audio
from tokimeki.models.gpu import loaded
from tokimeki.voice import speakers
from tokimeki.voice.asr import MODEL_REPO, SAMPLE_RATE, Transcriber
from tokimeki.voice.clips import samples
from tokimeki.voice.store import (
    VoiceStatus,
    clear_transcripts,
    episode_voices,
    set_transcripts,
)

NAME = "transcripts"
AFTER = (speakers.NAME,)
MAX_SECONDS = 30.0

log = logging.getLogger("tokimeki")


def done(ctx: Context, episode: Episode, repo: str) -> bool:
    params = stage_params(ctx.conn, episode.id, NAME)
    return (
        params is not None
        and params.get("model") == repo
        and speakers.fresh(ctx.conn, episode.id, NAME, AFTER)
    )


def run(ctx: Context, episodes: Sequence[Episode], repo: str = MODEL_REPO) -> int:
    """Transcribe the voiced lines of episodes not done with this model; returns lines done."""
    todo = [
        e
        for e in episodes
        if speakers.fresh(ctx.conn, e.id, speakers.NAME, speakers.AFTER) and not done(ctx, e, repo)
    ]
    if not todo:
        return 0
    count = 0
    with loaded(repo, lambda: Transcriber(repo)) as transcriber:
        for episode in todo:
            voices = [
                v for v in episode_voices(ctx.conn, episode.id) if v.status != VoiceStatus.SILENT
            ]
            stem = decode_audio(stem_path(ctx.paths, episode.id), SAMPLE_RATE)
            clips = [
                samples(stem, SAMPLE_RATE, v.start, min(v.end, v.start + MAX_SECONDS))
                for v in voices
            ]
            texts = transcriber.transcribe(clips)
            with transaction(ctx.conn):
                clear_transcripts(ctx.conn, episode.id)
                set_transcripts(
                    ctx.conn,
                    repo,
                    [(v.line_id, t) for v, t in zip(voices, texts, strict=True) if t],
                )
                mark_stage_done(ctx.conn, episode.id, NAME, {"model": repo})
            count += len(voices)
            log.info("%s: %d lines transcribed with %s", episode.path, len(voices), repo)
    return count


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        clear_transcripts(ctx.conn, episode.id)
        clear_stage(ctx.conn, episode.id, NAME)
