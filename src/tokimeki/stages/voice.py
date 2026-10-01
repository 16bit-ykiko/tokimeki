"""Stage 6: the voices of an episode, apart from its music, for lines laid over a MAD.

MDX-Net (on the GPU) separates the voice from the main audio track; the stem is cached as
FLAC and is silent outside kept shots, so a line from a dropped shot cannot be used.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Episode
from tokimeki.library.shots import kept_ranges
from tokimeki.media.audio import decode_audio, encode_audio, main_audio
from tokimeki.models.gpu import loaded
from tokimeki.models.separation import MODEL_FILE, SAMPLE_RATE, VocalSeparator
from tokimeki.paths import SeriesPaths
from tokimeki.stages.base import Context

NAME = "voice"
EDGE_RAMP = 0.01

log = logging.getLogger("tokimeki")


def stem_path(paths: SeriesPaths, episode_id: int) -> Path:
    return paths.cache / "voice" / f"{episode_id}.flac"


def kept_gain(ranges: Sequence[tuple[float, float]], length: int, rate: int) -> NDArray[np.float32]:
    """1 inside the ranges and 0 outside, with short ramps inside each range's edges."""
    gain = np.zeros(length, dtype=np.float32)
    ramp = max(1, round(EDGE_RAMP * rate))
    for start, end in ranges:
        a, b = max(0, round(start * rate)), min(length, round(end * rate))
        if b <= a:
            continue
        gain[a:b] = 1.0
        n = min(ramp, (b - a) // 2)
        if n:
            rise = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
            gain[a : a + n] = rise
            gain[b - n : b] = rise[::-1]
    return gain


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [
        e
        for e in episodes
        if stage_done(ctx.conn, e.id, "lines") and not stage_done(ctx.conn, e.id, NAME)
    ]
    if not todo:
        return
    with loaded("MDX-Net", VocalSeparator) as separator:
        for episode in todo:
            source = ctx.paths.episode_file(episode.path)
            stream = main_audio(source).index
            mix = decode_audio(source, SAMPLE_RATE, stream, channels=2)
            voice = separator.separate(mix)
            voice *= kept_gain(kept_ranges(ctx.conn, episode), voice.shape[1], SAMPLE_RATE)
            encode_audio(voice, SAMPLE_RATE, stem_path(ctx.paths, episode.id))
            with transaction(ctx.conn):
                mark_stage_done(ctx.conn, episode.id, NAME, {"model": MODEL_FILE})
            log.info("%s: voice separated (%.0f s)", episode.path, voice.shape[1] / SAMPLE_RATE)


def reset(ctx: Context, episode: Episode) -> None:
    stem_path(ctx.paths, episode.id).unlink(missing_ok=True)
    with transaction(ctx.conn):
        clear_stage(ctx.conn, episode.id, NAME)
