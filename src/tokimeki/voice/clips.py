"""Each dialogue line's voice in the episode's voice stem, and whether it is clean.

A clean voice is one speaker with enough voice to tell who: lines said over another line,
too short, too long or silent in the stem (off-screen speech, dropped shots) are kept with
their status but never get a speaker. Two voices one after the other inside one subtitle
line are caught later, when the halves of the line sound like different speakers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.lines import list_lines, list_parts
from tokimeki.library.records import Episode, Line, LineKind
from tokimeki.library.series import Context
from tokimeki.library.shots import kept_ranges
from tokimeki.media.audio import decode_audio
from tokimeki.stages.voice import heard, search_window, stem_path
from tokimeki.voice.store import VoiceStatus

CLIP_RATE = 16000
MIN_VOICE = 0.6
"""Seconds of voice below which a speaker embedding is mostly noise."""
MAX_VOICE = 12.0
OVERLAP = 0.15
"""Seconds two subtitle lines may overlap at a handover before both count as said at once."""


@dataclass(frozen=True, slots=True)
class Cut:
    line: Line
    start: float
    end: float
    status: VoiceStatus

    @property
    def duration(self) -> float:
        return self.end - self.start


def inside(ranges: Sequence[tuple[float, float]], start: float, end: float) -> bool:
    return any(a <= start + 1e-6 and end <= b + 1e-6 for a, b in ranges)


def overlapped(line: Line, dialogue: Sequence[Line]) -> bool:
    return any(
        x.id != line.id and min(x.end, line.end) - max(x.start, line.start) > OVERLAP
        for x in dialogue
    )


def status_of(seconds: float) -> VoiceStatus:
    if seconds < MIN_VOICE:
        return VoiceStatus.SHORT
    if seconds > MAX_VOICE:
        return VoiceStatus.LONG
    return VoiceStatus.CLEAN


def cut_lines(
    dialogue: Sequence[Line],
    kept: Sequence[tuple[float, float]],
    parts: Sequence[tuple[float, float]],
    stem: NDArray[np.float32],
    rate: int,
) -> list[Cut]:
    """Where each line inside kept shots and outside the OP/ED is heard in `stem` (mono,
    `rate`), and its status. `dialogue` is the episode's dialogue, by start."""
    cuts: list[Cut] = []
    for line in dialogue:
        if not inside(kept, line.start, line.end):
            continue
        if any(a < line.end and line.start < b for a, b in parts):
            continue
        window = search_window(line, dialogue, kept)
        if window is None:
            continue
        lo, hi = window
        found = heard(stem[round(lo * rate) : round(hi * rate)], rate, window, line)
        if found is None:
            cuts.append(Cut(line, line.start, line.end, VoiceStatus.SILENT))
            continue
        a, b = found
        status = VoiceStatus.OVERLAP if overlapped(line, dialogue) else status_of(b - a)
        cuts.append(Cut(line, a, b, status))
    return cuts


def episode_cuts(ctx: Context, episode: Episode) -> tuple[list[Cut], NDArray[np.float32]]:
    """The episode's cuts, and its voice stem as mono `CLIP_RATE` audio to take them from."""
    stem = decode_audio(stem_path(ctx.paths, episode.id), CLIP_RATE)
    dialogue = list_lines(ctx.conn, episode.id, LineKind.DIALOGUE)
    parts = [(p.start, p.end) for p in list_parts(ctx.conn, episode.id)]
    return cut_lines(dialogue, kept_ranges(ctx.conn, episode), parts, stem, CLIP_RATE), stem


def samples(stem: NDArray[np.float32], rate: int, start: float, end: float) -> NDArray[np.float32]:
    return np.ascontiguousarray(stem[round(start * rate) : round(end * rate)])
