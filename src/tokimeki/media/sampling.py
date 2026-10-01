import math
from fractions import Fraction

SAMPLE_INTERVAL_SECONDS = 0.5
MIN_SAMPLES_PER_SHOT = 3


def sample_frames(
    start: int,
    end: int,
    fps: Fraction,
    interval: float = SAMPLE_INTERVAL_SECONDS,
    minimum: int = MIN_SAMPLES_PER_SHOT,
) -> list[int]:
    """Frame indices to look at in the shot `[start, end)`.

    One frame per `interval` seconds and at least `minimum`, each at the centre of an equal
    slice of the shot, so the first and last frames (often mid-transition) are avoided.
    """
    length = end - start
    if length <= 0:
        return []
    wanted = max(minimum, math.ceil(length / fps / Fraction(interval)))
    count = min(length, wanted)
    return sorted({start + (2 * i + 1) * length // (2 * count) for i in range(count)})
