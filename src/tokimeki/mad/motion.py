"""Motion within a shot: where movement starts (a head turn, a blink, a jump) and whether the
shot is still, from the motion stage's per-frame measures.

A frame moves when at least `ACTIVE` of its pixels changed. An onset is a moving frame after
`CALM` frames without any, so the drawings held on twos and threes inside one movement are
not mistaken for new ones; a shot's opening movement, carried over the cut, is not an onset.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.records import Episode, Shot
from tokimeki.paths import SeriesPaths
from tokimeki.stages.motion import load_motion

ACTIVE = 0.015
CALM = 6
BURST = 12
"""Frames after an onset over which its strength (the most pixels changed) is taken."""
STILL_MOVING = 0.15
STILL_PAN = 0.003
"""A still shot moves in at most `STILL_MOVING` of its frames and its median mean difference
(which a slow pan raises and held drawings do not) stays under `STILL_PAN`."""


@dataclass(frozen=True, slots=True)
class Onset:
    time: float
    """Episode seconds of the first moving frame."""
    strength: float
    """The most pixels changed in the burst it starts (share of the frame)."""


@dataclass(frozen=True, slots=True)
class ShotMotion:
    onsets: tuple[Onset, ...]
    moving: float
    """Share of the shot's frames that move."""
    pan: float
    """Median mean difference between frames."""

    @property
    def still(self) -> bool:
        return self.moving <= STILL_MOVING and self.pan <= STILL_PAN


def shot_motion(measures: NDArray[np.float32], start: float, fps: float) -> ShotMotion:
    """Onsets and stillness of a shot from its `(frames, 2)` measures (mean difference,
    share changed); `start` is the episode second of its first frame."""
    clean = np.nan_to_num(measures.astype(np.float64))
    mean, changed = clean[:, 0], clean[:, 1]
    active = changed >= ACTIVE
    onsets = [
        Onset(start + t / fps, float(changed[t : t + BURST].max()))
        for t in range(CALM, len(changed))
        if active[t] and not active[t - CALM : t].any()
    ]
    pan = float(np.median(mean[1:])) if len(mean) > 1 else 0.0
    moving = float(active[1:].mean()) if len(active) > 1 else 0.0
    return ShotMotion(tuple(onsets), moving, pan)


def motions(paths: SeriesPaths, shots: Iterable[tuple[Episode, Shot]]) -> Mapping[int, ShotMotion]:
    """Motion of each shot whose episode the motion stage has measured, by shot id."""
    cache: dict[int, NDArray[np.float32] | None] = {}
    out: dict[int, ShotMotion] = {}
    for episode, shot in shots:
        if episode.id not in cache:
            cache[episode.id] = load_motion(paths, episode.id)
        measures = cache[episode.id]
        if measures is None:
            continue
        span = measures[shot.start_frame : shot.end_frame]
        out[shot.id] = shot_motion(span, episode.seconds(shot.start_frame), float(episode.fps))
    return out
