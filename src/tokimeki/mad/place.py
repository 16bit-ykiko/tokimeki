"""Cut placement: inside each chosen shot, the window around its expression peak.

Slots already start and end on beats, so placing a clip means choosing which stretch of
its shot fills the slot: centred on the cutest sampled frame, kept off the shot's edges,
and slowed (to 0.9x at most) when the shot is a little shorter than the slot.
"""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction

from tokimeki.mad.arrange import EDGE
from tokimeki.mad.candidates import Candidate
from tokimeki.mad.plan import MIN_SPEED, Clip, Plan

PEAK_AT = 0.5
"""Where in the clip the peak lands, as a share of its length."""


def window(candidate: Candidate, seconds: float) -> tuple[float, float]:
    """`(source start, speed)` for a clip that must fill `seconds` of the song."""
    first = candidate.start + EDGE
    usable = candidate.end - EDGE - first
    speed = 1.0 if usable >= seconds else max(MIN_SPEED, usable / seconds)
    length = seconds * speed
    start = candidate.peak.time - PEAK_AT * length
    start = min(max(start, first), max(first, candidate.end - EDGE - length))
    return start, speed


def snap(seconds: float, fps: Fraction) -> float:
    """The start of the source frame `seconds` falls in."""
    return float(int(seconds * fps) / fps)


def place(plan: Plan, candidates: Mapping[int, Candidate]) -> Plan:
    """The plan with every clip that has no window yet placed in its shot."""
    clips: list[Clip] = []
    for clip in plan.clips:
        if clip.source_start is not None and clip.speed is not None:
            clips.append(clip)
            continue
        candidate = candidates[clip.shot]
        start, speed = window(candidate, plan.slots[clip.slot].duration)
        clips.append(
            replace(
                clip,
                source_start=snap(start, candidate.episode.fps),
                speed=round(speed, 4),
                peak=round(candidate.peak.time, 3),
            )
        )
    return replace(plan, clips=clips)
