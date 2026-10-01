"""Where a song really starts and ends inside a longer recording (an episode's OP, say).

Given a rough window, the start is the sharpest attack near the window's start (the OP cuts
in over the last scene's fading music) and the end is where the level falls into silence
near the window's end. Pure numpy over decoded samples.
"""

import numpy as np
from numpy.typing import NDArray

FRAME = 0.01
"""Seconds per loudness frame."""
SEARCH_BEFORE = 10.0
SEARCH_AFTER = 5.0
MIN_ATTACK_DB = 20.0
SILENCE_DB = -60.0


def loudness_db(samples: NDArray[np.float32], rate: int) -> NDArray[np.float64]:
    width = max(1, round(rate * FRAME))
    usable = len(samples) // width * width
    power = (samples[:usable].astype(np.float64).reshape(-1, width) ** 2).mean(axis=1)
    return 10 * np.log10(power + 1e-14)


def find_start(db: NDArray[np.float64], around: float) -> float | None:
    """The frame (in seconds from the start of `db`) of the strongest attack near `around`:
    loudness jumping by at least `MIN_ATTACK_DB` over the quietest of the 0.2 s before it."""
    lo = max(int((around - SEARCH_BEFORE) / FRAME), 20)
    hi = min(int((around + SEARCH_AFTER) / FRAME), len(db) - 2)
    best, best_rise = None, MIN_ATTACK_DB
    for i in range(lo, hi):
        rise = float(db[i : i + 2].max() - db[i - 20 : i].min())
        if rise > best_rise and db[i - 1] < db[i]:
            best, best_rise = i, rise
    return None if best is None else best * FRAME


def find_end(db: NDArray[np.float64], around: float) -> float | None:
    """Where the level first stays below `SILENCE_DB` for 0.2 s near `around`."""
    lo = max(int((around - SEARCH_AFTER) / FRAME), 0)
    hi = min(int((around + SEARCH_BEFORE) / FRAME), len(db) - 20)
    for i in range(lo, hi):
        if (db[i : i + 20] < SILENCE_DB).all():
            return i * FRAME
    return None


def song_bounds(
    samples: NDArray[np.float32], rate: int, offset: float, start: float, end: float
) -> tuple[float, float]:
    """Exact `(start, end)` (same clock as `start`/`end`) of the song roughly in `start`-`end`;
    `samples` begin at `offset` on that clock. Falls back to the rough edges."""
    db = loudness_db(samples, rate)
    found_start = find_start(db, start - offset)
    found_end = find_end(db, end - offset)
    return (
        offset + found_start if found_start is not None else start,
        offset + found_end if found_end is not None else end,
    )
