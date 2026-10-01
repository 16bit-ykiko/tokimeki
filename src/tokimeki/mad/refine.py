"""Refining a plan: slot boundaries onto exact beats, and each clip's window around the
expression peak of its shot, shifted so a motion onset lands on the beat.

Windows the plan already has are kept (snapped to whole source frames). A missing window is
centred on the shot's cutest sampled frame, kept off the shot's edges, and slowed to 0.9x at
most when the shot is a little shorter than its slot. Then (for windows refine fills, or any
slot with `sync`) the window slides, same length and speed, so the strongest movement that
starts in the shot (a head turn, a blink, a jump) begins on the cut or on an accent beat,
the expression peak staying inside.
"""

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from tokimeki.mad.candidates import Candidate
from tokimeki.mad.motion import Onset, ShotMotion
from tokimeki.mad.plan import MAX_SPEED, MIN_SPEED, Plan, SlotPlan
from tokimeki.song.analysis import SongAnalysis

EDGE_FRAMES = 2
"""Frames kept off each end of a shot, where transitions and stray frames sit."""
PEAK_AT = 0.5
"""Where in the clip the peak lands, as a share of its length."""
MIN_SYNC_ONSET = 0.03
"""Onsets moving fewer pixels than this (a lip flap) are not worth aligning."""
CUT_LEAD = 1
"""Frames after the cut a movement starts, so the eye sees it begin."""
PEAK_MARGIN = 0.15
SHIFT_COST = 0.02
"""Strength an alignment loses per second the window moves, so near ones win ties."""


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


@dataclass(frozen=True, slots=True)
class Sync:
    start: float
    onset: Onset
    target: str


def sync(
    slot: SlotPlan,
    start: float,
    length: float,
    candidate: Candidate,
    motion: ShotMotion,
    accents: Sequence[tuple[float, float]],
) -> Sync | None:
    """Where the window (`start`, `length` source seconds) should start so the best onset
    lands on the cut or on one of the `accents` (song seconds, weight) inside the slot,
    keeping the shot's usable range and the expression peak (or the window's middle, when the
    peak is outside it) inside; None when no onset fits."""
    first, last = usable(candidate)
    speed = length / slot.duration
    peak = candidate.peak.time
    anchor = peak if start <= peak <= start + length else start + length / 2
    margin = min(PEAK_MARGIN, length / 4)
    lead = float(CUT_LEAD / candidate.episode.fps)
    targets = [(slot.start + lead, 1.0, "the cut")] + [
        (t, w, f"the accent at {t:.2f}s")
        for t, w in accents
        if slot.start + lead < t < slot.end - 2 * lead
    ]
    best: tuple[float, Sync] | None = None
    for onset in motion.onsets:
        if onset.strength < MIN_SYNC_ONSET:
            continue
        for at, weight, label in targets:
            new = onset.time - (at - slot.start) * speed
            if new < first - 1e-6 or new + length > last + 1e-6:
                continue
            if not new + margin <= anchor <= new + length - margin:
                continue
            score = onset.strength * weight - SHIFT_COST * abs(new - start)
            if best is None or score > best[0]:
                best = (score, Sync(new, onset, label))
    return best[1] if best else None


def section_at(analysis: SongAnalysis, t: float) -> str:
    return next((s.label for s in analysis.sections if s.start <= t + 1e-6 < s.end), "")


def refine(
    plan: Plan,
    analysis: SongAnalysis,
    candidates: Mapping[int, Candidate],
    motion: Mapping[int, ShotMotion] | None = None,
    accents: Sequence[tuple[float, float]] = (),
) -> tuple[Plan, list[str]]:
    """The plan on the beat grid with every clip's window filled and, where `motion` is
    known, synced to it (onsets on the cut, the slot's `accent` or one of the `accents`,
    song seconds with weights); and what was changed."""
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
        filled = slot.source_in is None
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
        shot_motion = (motion or {}).get(candidate.shot.id)
        wanted = slot.sync if slot.sync is not None else filled
        if wanted and shot_motion is not None:
            marked = [(slot.accent, 1.2)] if slot.accent is not None else []
            found = sync(slot, first, length, candidate, shot_motion, [*marked, *accents])
            if found is not None:
                moved = frame_floor(found.start, fps)
                if moved < usable(candidate)[0] - 1e-6:
                    moved = frame_ceil(found.start, fps)
                o = found.onset
                notes.append(
                    f"slot {i}: motion at {o.time:.2f}s ({o.strength:.2f}) on {found.target},"
                    f" window {moved - first:+.2f}s"
                )
                first = moved
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
