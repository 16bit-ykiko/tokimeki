"""Refining a plan: slot boundaries onto exact beats, and each clip's window around the
expression peak of its shot.

Windows the plan already has are kept (snapped to whole source frames). A missing window is
centred on the shot's cutest sampled frame, kept off the shot's edges, and slowed to 0.9x at
most when the shot is a little shorter than its slot.
"""

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction

from tokimeki.mad.candidates import Candidate
from tokimeki.mad.plan import MAX_SPEED, MIN_SPEED, Plan, SlotPlan
from tokimeki.song.analysis import SongAnalysis

EDGE_FRAMES = 2
"""Frames kept off each end of a shot, where transitions and stray frames sit."""
PEAK_AT = 0.5
"""Where in the clip the peak lands, as a share of its length."""


def nearest(grid: Sequence[float], t: float) -> float:
    i = bisect_left(grid, t)
    options = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
    return min(options, key=lambda g: abs(g - t))


def beat_grid(analysis: SongAnalysis) -> list[float]:
    return sorted({*analysis.beats, 0.0, analysis.source.duration})


def edge(fps: Fraction) -> float:
    return float(EDGE_FRAMES / fps)


def usable(candidate: Candidate) -> tuple[float, float]:
    """The part of a shot clips may show: all of it but `EDGE_FRAMES` at either end."""
    margin = edge(candidate.episode.fps)
    return candidate.start + margin, candidate.end - margin


def window(candidate: Candidate, seconds: float) -> tuple[float, float]:
    """`(source start, speed)` for a clip that must fill `seconds` of the song."""
    first, last = usable(candidate)
    usable_length = last - first
    speed = 1.0 if usable_length >= seconds else max(MIN_SPEED, usable_length / seconds)
    length = seconds * speed
    start = candidate.peak.time - PEAK_AT * length
    start = min(max(start, first), max(first, last - length))
    return start, speed


def frame_floor(seconds: float, fps: Fraction) -> float:
    """The start of the source frame `seconds` falls in."""
    return float(math.floor(Fraction(seconds) * fps + Fraction(1, 1000)) / fps)


def frame_ceil(seconds: float, fps: Fraction) -> float:
    """The first frame boundary at or after `seconds`."""
    return float(math.ceil(Fraction(seconds) * fps - Fraction(1, 1000)) / fps)


def section_at(analysis: SongAnalysis, t: float) -> str:
    return next((s.label for s in analysis.sections if s.start <= t + 1e-6 < s.end), "")


def refine(
    plan: Plan, analysis: SongAnalysis, candidates: Mapping[int, Candidate]
) -> tuple[Plan, list[str]]:
    """The plan on the beat grid with every clip's window filled; and what was changed."""
    notes: list[str] = []
    grid = beat_grid(analysis)
    bounds = [plan.slots[0].start, *(s.end for s in plan.slots)] if plan.slots else []
    snapped = [nearest(grid, b) for b in bounds]
    slots: list[SlotPlan] = []
    for i, slot in enumerate(plan.slots):
        start, end = snapped[i], snapped[i + 1]
        if abs(start - slot.start) > 1e-3 or abs(end - slot.end) > 1e-3:
            moved = f"{slot.start:.3f}-{slot.end:.3f} -> {start:.3f}-{end:.3f}"
            notes.append(f"slot {i}: moved onto beats, {moved}")
        slot = replace(slot, start=start, end=end)
        candidate = candidates.get(slot.shot) if slot.shot is not None else None
        if candidate is None or slot.duration <= 0:
            slots.append(slot)
            continue
        fps = candidate.episode.fps
        if slot.source_in is not None and slot.source_out is not None:
            first = frame_floor(slot.source_in, fps)
            length = slot.source_out - slot.source_in
        elif slot.source_in is not None:
            first = frame_floor(slot.source_in, fps)
            speed = min(MAX_SPEED, max(MIN_SPEED, slot.speed or 1.0))
            length = slot.duration * speed
            notes.append(f"slot {i}: out set from in at {speed:.2f}x")
        else:
            start_at, speed = window(candidate, slot.duration)
            first = frame_floor(start_at, fps)
            if first < usable(candidate)[0] - 1e-6:
                first = frame_ceil(start_at, fps)
            length = slot.duration * speed
            notes.append(
                f"slot {i}: window around the peak at {candidate.peak.time:.2f}s ({speed:.2f}x)"
            )
        slots.append(
            replace(
                slot,
                source_in=round(first, 4),
                source_out=round(first + length, 4),
                speed=round(length / slot.duration, 4),
                section=slot.section or section_at(analysis, start),
                episode=slot.episode or candidate.episode.path,
            )
        )
    return replace(plan, slots=slots), notes
