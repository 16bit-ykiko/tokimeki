"""How hard each beat hits: spectral flux (new sound arriving) and the jump in loudness at
the beat, from the mix itself (numpy; a few seconds of work for a song)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

RATE = 22050
N_FFT = 2048
HOP = 441
"""50 frames a second."""
BEFORE, AFTER = 0.05, 0.07
"""Seconds around a beat searched for its flux peak (beats sit a little before the attack)."""
RISE_WINDOW = (0.25, 0.05, 0.1)
"""Loudness is compared over 250-50 ms before the beat and 0-100 ms after it."""
RISE_FULL = 6.0
"""dB of rise that counts fully."""
FLUX_WEIGHT = 0.7


def flux_and_loudness(
    samples: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per 10 ms frame of mono `RATE` audio: positive spectral flux of the log magnitude, and
    loudness in dB."""
    frames = max(0, (len(samples) - N_FFT) // HOP + 1)
    if frames < 2:
        return np.zeros(frames), np.full(frames, -120.0)
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    chunks = samples[index].astype(np.float64) * np.hanning(N_FFT)
    spectrum = np.log1p(100.0 * np.abs(np.fft.rfft(chunks, axis=1)))
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(spectrum, axis=0)).sum(axis=1)])
    loudness = 10 * np.log10((chunks**2).mean(axis=1) + 1e-12)
    return flux, loudness


def accent_strengths(samples: NDArray[np.float32], beats: Sequence[float]) -> list[float]:
    """0-1 per beat: its flux peak (as a share of the song's 95th-percentile beat) and its
    loudness rise, weighted `FLUX_WEIGHT` to the rest."""
    if not beats:
        return []
    flux, loudness = flux_and_loudness(samples)
    if len(flux) == 0:
        return [0.0] * len(beats)
    fps = RATE / HOP

    def frame(t: float) -> int:
        return int(np.clip(round(t * fps), 0, len(flux) - 1))

    peaks = np.array([flux[frame(b - BEFORE) : frame(b + AFTER) + 1].max() for b in beats])
    early, late, after = RISE_WINDOW
    rises = np.array(
        [
            loudness[frame(b) : frame(b + after) + 1].mean()
            - loudness[frame(b - early) : max(frame(b - early) + 1, frame(b - late))].mean()
            for b in beats
        ]
    )
    scale = float(np.percentile(peaks, 95)) or 1.0
    strength = FLUX_WEIGHT * np.clip(peaks / scale, 0, 1) + (1 - FLUX_WEIGHT) * np.clip(
        rises / RISE_FULL, 0, 1
    )
    return [round(float(x), 3) for x in strength]
