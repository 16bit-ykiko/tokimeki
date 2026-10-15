"""What the user needs to name speakers by ear: per speaker its most typical and most doubtful
lines, with their text and who is on screen, and the cast it seems to match.

The data is plain dataclasses (the JSON API returns them as they are); `page` turns them
into a static listening page and `write_clips` cuts the audio it plays.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tokimeki.library.episodes import get_episode, list_episodes
from tokimeki.library.lines import get_line
from tokimeki.library.series import Context
from tokimeki.media.audio import decode_audio, encode_audio
from tokimeki.stages.voice import stem_path
from tokimeki.voice.embedding import cosine_distances
from tokimeki.voice.paths import page_dir
from tokimeki.voice.speakers import Evidence, Match, evidence
from tokimeki.voice.store import (
    LineVoice,
    VoiceStatus,
    clean_voices,
    episode_voices,
    list_speakers,
    transcripts,
    voice_embeddings,
)

AUDIO_RATE = 44100


@dataclass(frozen=True, slots=True)
class LineClip:
    line: int
    episode: int
    start: float
    end: float
    subtitle: str
    transcript: str | None
    distance: float
    """Cosine distance to the mean voice of its speaker (0 when it has none)."""
    audio: str
    """The clip's path under the page directory."""


@dataclass(frozen=True, slots=True)
class SpeakerReview:
    id: int
    name: str | None
    lines: int
    seconds: float
    suggestion: str | None
    seeds: dict[str, int]
    """Lines said while one named character was alone on screen in a close-up."""
    matches: list[Match]
    typical: list[LineClip]
    doubtful: list[LineClip]


@dataclass(frozen=True, slots=True)
class VoiceReview:
    statuses: dict[str, int]
    clean_seconds: float
    speakers: list[SpeakerReview]
    unassigned: list[LineClip]


def clip_name(voice: LineVoice) -> str:
    return f"clips/{voice.line_id}-{round(voice.start * 1000)}-{round(voice.end * 1000)}.m4a"


def _clips(
    conn: sqlite3.Connection, voices: Sequence[LineVoice], texts: dict[int, str]
) -> list[LineClip]:
    if not voices:
        return []
    embeddings = voice_embeddings(conn, [v.line_id for v in voices])
    centre = embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(embeddings, centre)[:, 0]
    return [
        LineClip(
            v.line_id,
            v.episode_id,
            round(v.start, 3),
            round(v.end, 3),
            " ".join(get_line(conn, v.line_id).text.split()),
            texts.get(v.line_id),
            round(float(d), 3),
            clip_name(v),
        )
        for v, d in zip(voices, distances, strict=True)
    ]


def review(ctx: Context, typical: int = 6, doubtful: int = 3, unassigned: int = 12) -> VoiceReview:
    conn = ctx.conn
    texts = transcripts(conn)
    found = evidence(ctx)
    statuses: dict[str, int] = {s.value: 0 for s in VoiceStatus}
    for episode in list_episodes(conn):
        for v in episode_voices(conn, episode.id):
            statuses[v.status.value] += 1
    clean = clean_voices(conn)
    speakers: list[SpeakerReview] = []
    for s in list_speakers(conn):
        clips = sorted(_clips(conn, clean_voices(conn, s.id), texts), key=lambda c: c.distance)
        far = clips[typical:][-doubtful:] if doubtful else []
        proof = found.get(s.id, Evidence())
        speakers.append(
            SpeakerReview(
                s.id,
                s.name,
                s.lines,
                round(s.seconds, 1),
                proof.suggestion,
                proof.seeds,
                proof.matches[:3],
                clips[:typical],
                far[::-1],
            )
        )
    lone = [v for v in clean if v.speaker_id is None]
    lone = sorted(lone, key=lambda v: -v.duration)[:unassigned]
    return VoiceReview(
        statuses,
        round(sum(v.duration for v in clean), 1),
        speakers,
        _clips(conn, lone, texts),
    )


def write_clips(ctx: Context, data: VoiceReview) -> int:
    """Cut every clip the review plays that is not on disk yet; returns how many were cut."""
    root = page_dir(ctx.paths)
    wanted = [
        c for s in data.speakers for c in (*s.typical, *s.doubtful) if not (root / c.audio).exists()
    ] + [c for c in data.unassigned if not (root / c.audio).exists()]
    for episode_id in sorted({c.episode for c in wanted}):
        episode = get_episode(ctx.conn, episode_id)
        stem = decode_audio(stem_path(ctx.paths, episode.id), AUDIO_RATE)
        for c in (c for c in wanted if c.episode == episode_id):
            part = stem[round(c.start * AUDIO_RATE) : round(c.end * AUDIO_RATE)]
            encode_audio(np.ascontiguousarray(part[None]), AUDIO_RATE, root / c.audio)
    return len(wanted)
