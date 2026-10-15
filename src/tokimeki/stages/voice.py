"""Stage 6: the voices of an episode, apart from its music, for lines laid over a MAD.

MDX-Net (on the GPU) separates the voice from the main audio track; the stem is cached as
FLAC and is silent outside kept shots, so a line from a dropped shot cannot be used.
"""

import logging
from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.paths import SeriesPaths
from tokimeki.library.records import Episode, Line
from tokimeki.library.series import Context
from tokimeki.library.shots import kept_ranges
from tokimeki.media.audio import decode_audio, encode_audio, main_audio
from tokimeki.models.gpu import loaded
from tokimeki.models.separation import MODEL_FILE, SAMPLE_RATE, VocalSeparator

NAME = "voice"
EDGE_RAMP = 0.01
TRIM_SEARCH = 0.3
"""Seconds searched for the voice beyond the line's subtitle times."""
TRIM_FLOOR = 30.0
"""dB below the loudest 10 ms of the line that still count as voice."""
TRIM_LEAD, TRIM_TAIL = 0.06, 0.12
TRIM_RATE = 16000
PAUSE = 0.06
"""Shorter silences inside a line do not split it."""

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


def search_window(
    line: Line, neighbours: Sequence[Line], kept: Sequence[tuple[float, float]]
) -> tuple[float, float] | None:
    """Where to look for the line's voice: its subtitle times widened by `TRIM_SEARCH`, never
    into the previous or next of `neighbours` (the episode's dialogue, by start) nor out of
    the kept range it starts in; None when it starts outside kept shots."""
    starts = [x.start for x in neighbours]
    i = bisect_left(starts, line.start)
    before = max((x.end for x in neighbours[:i] if x.id != line.id), default=0.0)
    after = min((x.start for x in neighbours[i:] if x.id != line.id), default=float("inf"))
    room = next(((a, b) for a, b in kept if a <= line.start < b), None)
    if room is None:
        return None
    lo = max(line.start - TRIM_SEARCH, min(before, line.start), room[0])
    hi = min(line.end + TRIM_SEARCH, max(after, line.end), room[1])
    return lo, hi


def heard(
    samples: NDArray[np.float32], rate: int, window: tuple[float, float], line: Line
) -> tuple[float, float] | None:
    """Episode seconds of the line's voice in `samples`, the stem over `window`; None when
    it is silent there."""
    lo, hi = window
    found = voiced(samples, rate, (line.start - lo, line.end - lo))
    if found is None:
        return None
    a, b = found
    return max(lo, lo + a - TRIM_LEAD), min(hi, lo + b + TRIM_TAIL)


def voice_bounds(
    ctx: Context, episode: Episode, line: Line, neighbours: list[Line]
) -> tuple[float, float]:
    """Where the line's voice is in the stem (`search_window` trimmed to the voice); the
    subtitle times when the stem is silent there or not separated yet."""
    path = stem_path(ctx.paths, episode.id)
    window = search_window(line, neighbours, kept_ranges(ctx.conn, episode))
    if not path.exists() or window is None:
        return line.start, line.end
    lo, hi = window
    samples = decode_audio(path, TRIM_RATE, start=lo, duration=hi - lo)
    return heard(samples, TRIM_RATE, window, line) or (line.start, line.end)


def voiced(
    samples: NDArray[np.float32], rate: int, core: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    """Start and end (seconds) of the voice: runs of 10 ms frames within `TRIM_FLOOR` dB of
    the loudest, pauses under `PAUSE` bridged, and of those only the runs that overlap
    `core` (the subtitle's own times) when given."""
    hop = rate // 100
    frames = len(samples) // hop
    if frames == 0:
        return None
    rms = np.sqrt((samples[: frames * hop].reshape(frames, hop).astype(np.float64) ** 2).mean(1))
    db = 20 * np.log10(rms + 1e-9)
    if db.max() < -60:
        return None
    loud = np.flatnonzero(db >= db.max() - TRIM_FLOOR)
    runs: list[tuple[int, int]] = []
    for i in loud:
        if runs and i - runs[-1][1] <= round(PAUSE * 100):
            runs[-1] = (runs[-1][0], int(i) + 1)
        else:
            runs.append((int(i), int(i) + 1))
    if core is not None:
        a, b = core[0] * 100, core[1] * 100
        runs = [r for r in runs if r[0] < b and a < r[1]] or runs
    return runs[0][0] * hop / rate, runs[-1][1] * hop / rate
