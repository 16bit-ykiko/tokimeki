"""Where a spoken line fits in a song: the stretches nobody sings, and the quiet bars."""

from dataclasses import dataclass

import numpy as np

from tokimeki.song.analysis import SUNG, SongAnalysis
from tokimeki.song.structure import VOCAL_ON

MIN_GAP = 0.8
JOIN = 0.05
"""Sung stretches closer than this are one."""
QUIET_SHARE = 1 / 3
"""Quiet bars are the unsung ones among the quietest third of the excerpt's bars."""
UNSUNG_BAR = 0.25


@dataclass(frozen=True, slots=True)
class Gap:
    start: float
    end: float
    where: str
    """"intro" (at the excerpt's start), "outro" (at its end) or "between"."""

    @property
    def length(self) -> float:
        return self.end - self.start


def sung_spans(analysis: SongAnalysis) -> list[tuple[float, float]]:
    """When the song is sung: its lyric lines, else its sung bars, joined."""
    if analysis.lyrics:
        spans = sorted((x.start, x.end) for x in analysis.lyrics)
    elif analysis.vocal is not None:
        threshold = SUNG.get(analysis.vocal_source, VOCAL_ON)
        edges = [*analysis.bars, analysis.source.duration]
        spans = [
            (edges[i], edges[i + 1])
            for i, share in enumerate(analysis.vocal)
            if share >= threshold and i + 1 < len(edges)
        ]
    else:
        return []
    joined: list[tuple[float, float]] = []
    for a, b in spans:
        if joined and a - joined[-1][1] < JOIN:
            joined[-1] = (joined[-1][0], max(joined[-1][1], b))
        else:
            joined.append((a, b))
    return joined


def sung_overlap(analysis: SongAnalysis, start: float, end: float) -> float:
    """Seconds of `start`-`end` that are sung."""
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in sung_spans(analysis))


def gaps(
    analysis: SongAnalysis, start: float, end: float, min_length: float = MIN_GAP
) -> list[Gap]:
    """The unsung stretches of `start`-`end` at least `min_length` long."""
    out: list[Gap] = []
    t = start
    for a, b in [*sung_spans(analysis), (end, end)]:
        a, b = max(a, start), min(b, end)
        if b < start or a > end:
            continue
        if a - t >= min_length:
            where = "intro" if t <= start + 1e-6 else "outro" if a >= end - 1e-6 else "between"
            out.append(Gap(t, a, where))
        t = max(t, b)
    return out


def quiet_bars(analysis: SongAnalysis, start: float, end: float) -> list[float]:
    """Starts of the excerpt's unsung bars among its quietest third."""
    edges = [*analysis.bars, analysis.source.duration]
    inside = [
        i for i, b in enumerate(analysis.bars) if start - 1e-6 <= b and edges[i + 1] <= end + 1e-6
    ]
    if not inside or not analysis.energy:
        return []
    cutoff = float(np.quantile([analysis.energy[i] for i in inside], QUIET_SHARE))
    return [
        analysis.bars[i]
        for i in inside
        if analysis.energy[i] <= cutoff
        and sung_overlap(analysis, edges[i], edges[i + 1]) < UNSUNG_BAR * (edges[i + 1] - edges[i])
    ]
