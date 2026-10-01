"""From a song analysis to an excerpt and the slots a MAD fills, one cut per slot."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from tokimeki.song.analysis import SongAnalysis
from tokimeki.song.structure import Section

MIN_EXCERPT = 45.0
MAX_EXCERPT = 90.0

DENSITY: dict[str, tuple[int, ...]] = {
    "intro": (4, 8, 16),
    "verse": (4, 8, 16),
    "pre-chorus": (2, 4, 8),
    "chorus": (2, 4, 8),
    "interlude": (4, 8, 16),
    "bridge": (4, 8, 16),
    "outro": (4, 8, 16),
}
"""Beats per slot for each kind of section, fastest first: a cut a bar in verses, every two
beats in the chorus, slower when there are not enough shots to fill the slots."""

RELAX_ORDER = ("verse", "intro", "interlude", "outro", "bridge", "pre-chorus", "chorus")
MIN_SLOT_SECONDS = 0.5
MAX_SLOT_SECONDS = 4.0
"""Longer slots are hard to fill: few anime shots last that long."""


@dataclass(frozen=True, slots=True)
class Excerpt:
    start: float
    end: float
    sections: list[Section]

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Slot:
    index: int
    start: float
    end: float
    """Seconds from the start of the excerpt."""
    section: str
    beats: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def first_chorus(analysis: SongAnalysis) -> Excerpt:
    """From the top of the song through its first chorus, trimmed or extended by whole
    sections to stay within `MIN_EXCERPT` to `MAX_EXCERPT` seconds when possible."""
    sections = analysis.sections
    last = next((i for i, s in enumerate(sections) if s.label == "chorus"), len(sections) - 1)
    first = 0
    while first < last and sections[last].end - sections[first].start > MAX_EXCERPT:
        first += 1
    while last + 1 < len(sections) and sections[last].end - sections[first].start < MIN_EXCERPT:
        last += 1
    chosen = sections[first : last + 1]
    return Excerpt(chosen[0].start, chosen[-1].end, chosen)


def _beat_slots(
    beats: list[float], start: float, end: float, per_slot: int
) -> list[tuple[float, float, int]]:
    """Spans of `per_slot` beats from `start` to `end`; a span too short to read (a stray
    beat, the section's remainder) joins its neighbour."""
    inside = [b for b in beats if start <= b < end]
    marks = [start, *[b for b in inside[per_slot::per_slot] if b - start > 1e-6], end]
    spans = [(a, b) for a, b in pairwise(marks) if b > a]
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and (
            b - a < MIN_SLOT_SECONDS or merged[-1][1] - merged[-1][0] < MIN_SLOT_SECONDS
        ):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b, per_slot) for a, b in merged]


def excerpt_between(analysis: SongAnalysis, start: float, end: float) -> Excerpt:
    """The stretch `start`-`end` (song seconds), snapped to the nearest bar lines."""
    edges = [*analysis.bars, analysis.song.duration]
    first = min(edges, key=lambda b: abs(b - start))
    last = min(edges, key=lambda b: abs(b - end))
    if last <= first:
        raise ValueError(f"excerpt {start:.1f}-{end:.1f}s is shorter than a bar")
    sections = [
        Section(
            s.label,
            s.group,
            s.first_bar,
            s.end_bar,
            max(s.start, first),
            min(s.end, last),
            s.vocal,
            s.loudness,
        )
        for s in analysis.sections
        if s.end > first and s.start < last
    ]
    return Excerpt(first, last, sections)


def make_slots(
    analysis: SongAnalysis,
    excerpt: Excerpt,
    budget: int,
    fixed: Mapping[str, int] | None = None,
) -> list[Slot]:
    """Slots on the beat grid, sped up per section kind as far as `budget` shots allow.

    Section kinds in `fixed` keep that many beats a slot whatever the budget.
    """
    fixed = fixed or {}
    level = dict.fromkeys(DENSITY, 0)

    def build() -> list[Slot]:
        spans: list[tuple[float, float, int, str]] = []
        for section in excerpt.sections:
            ladder = DENSITY.get(section.label, (4, 8, 16))
            per_slot = fixed.get(
                section.label, ladder[min(level.get(section.label, 0), len(ladder) - 1)]
            )
            for a, b, n in _beat_slots(analysis.beats, section.start, section.end, per_slot):
                spans.append((a, b, n, section.label))
        return [
            Slot(i, a - excerpt.start, b - excerpt.start, label, n)
            for i, (a, b, n, label) in enumerate(spans)
        ]

    present = {s.label for s in excerpt.sections}
    beat = 60.0 / analysis.bpm if analysis.bpm > 0 else 0.5

    def can_slow(label: str) -> bool:
        ladder = DENSITY[label]
        return (
            label in present
            and label not in fixed
            and level[label] + 1 < len(ladder)
            and ladder[level[label] + 1] * beat <= MAX_SLOT_SECONDS
        )

    slots = build()
    while len(slots) > budget:
        label = next((x for x in RELAX_ORDER if can_slow(x)), None)
        if label is None:
            break
        level[label] += 1
        slots = build()
    return slots
