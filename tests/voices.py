"""A synthetic episode for the voice tests: two tones stand for two voices, faces for the cast."""

from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.anime.stems import stem_path
from tokimeki.library import cast as cast_db
from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.lines import NewLine, replace_lines
from tokimeki.library.records import Box, LineKind, Part, PartKind, Rating
from tokimeki.library.series import Context, open_series
from tokimeki.media.audio import encode_audio
from tokimeki.voice.clips import CLIP_RATE

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


def tone(
    rate: int, seconds: float, spans: list[tuple[tuple[float, float], float]]
) -> NDArray[np.float32]:
    audio = np.zeros(round(rate * seconds), dtype=np.float32)
    for (a, b), freq in spans:
        t = np.arange(round(a * rate), round(b * rate))
        audio[t] = 0.3 * np.sin(2 * np.pi * freq * t / rate)
    return audio


def faces(ctx: Context, shot_id: int, frames: int, cluster: int, height: float) -> None:
    safe = Rating(1, 0, 0, 0)
    shot_db.keep_shot(ctx.conn, shot_id, [shot_db.RatedFrame(i, safe, []) for i in range(frames)])
    box = Box(0.4, 0.5 - height / 2, 0.6, 0.5 + height / 2)
    new = [
        cast_db.NewFace(f.id, box, 0.9, np.ones(4, dtype=np.float32))
        for f in shot_db.list_frames(ctx.conn, shot_id)
    ]
    cast_db.assign_faces(ctx.conn, cast_db.add_faces(ctx.conn, new), cluster)


def series(tmp_path: Path) -> Context:
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
    faces(ctx, momo_shot.id, 4, momo, 0.5)
    faces(ctx, nana_shot.id, 4, nana, 0.5)
    shot_db.keep_shot(ctx.conn, rest.id, [])
    timed = [*MOMO_LINES, *NANA_LINES, *OVERLAPPING, SHORT, IN_OPENING]
    lines = [
        NewLine(a, b, LineKind.DIALOGUE, "Default", f"line {i}") for i, (a, b) in enumerate(timed)
    ]
    opening = Part(episode.id, PartKind.OPENING, 26.5, 29.0)
    replace_lines(ctx.conn, episode.id, lines, [opening])
    voices = [(s, LOW) for s in MOMO_LINES] + [(s, HIGH) for s in NANA_LINES]
    voices += [(s, LOW) for s in (*OVERLAPPING, SHORT, IN_OPENING)]
    stem = tone(44100, seconds, voices)
    encode_audio(np.stack([stem, stem]), 44100, stem_path(ctx.paths, episode.id))
    for stage in ("lines", "voice"):
        episode_db.mark_stage_done(ctx.conn, episode.id, stage, {})
    return ctx
