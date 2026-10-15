"""The MAD's sound: the song excerpt with original lines laid over it, the song ducked
smoothly under each line, the whole brought to a -1 dBFS peak (one gain, no compression:
a song taken from an episode's TV mix peaks far lower than a CD).

Pure numpy on `(2, samples)` float arrays, so it is tested without audio files.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RATE = 48000
DUCK_DB = 9.0
DUCK_ATTACK = 0.15
"""Seconds the song takes to go down, ending where the line starts."""
DUCK_RELEASE = 0.4
VOICE_FADE = 0.02
VOICE_ABOVE = 8.0
"""dB the voice sits above the ducked song around it, before the plan's gain: about the
song's own level, so the line reads like part of the track."""
PEAK = 0.97
TARGET_PEAK = 10 ** (-1 / 20)
SONG_FADE_IN, SONG_FADE_OUT = 0.05, 0.5

type Audio = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VoiceClip:
    at: float
    """Seconds from the start of the mix."""
    samples: Audio
    gain: float = 0.0
    duck: float = DUCK_DB


def _ramp(n: int) -> NDArray[np.float64]:
    """A raised-cosine rise over `n` samples."""
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)) if n > 0 else np.zeros(0)


def duck_gain(length: int, spans: Sequence[tuple[float, float, float]]) -> NDArray[np.float64]:
    """Linear gain for the song: down by each span's dB from its start to its end, easing
    down over `DUCK_ATTACK` before it and back over `DUCK_RELEASE` after; the deepest wins."""
    db = np.zeros(length)
    for start, end, depth in spans:
        a, b = round(start * RATE), round(end * RATE)
        attack, release = round(DUCK_ATTACK * RATE), round(DUCK_RELEASE * RATE)
        curve = np.zeros(length)
        lo, hi = max(0, a), min(length, b)
        curve[lo:hi] = depth
        down = depth * _ramp(attack)
        first = a - attack
        seg = slice(max(0, first), max(0, min(length, a)))
        curve[seg] = down[seg.start - first : seg.stop - first]
        up = depth * _ramp(release)[::-1]
        seg = slice(max(0, b), max(0, min(length, b + release)))
        curve[seg] = up[seg.start - b : seg.stop - b]
        db = np.maximum(db, curve)
    return 10 ** (-db / 20)


def level_db(samples: Audio) -> float:
    """RMS level (dBFS) of the parts within 25 dB of the loudest 20 ms."""
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    hop = RATE // 50
    frames = len(mono) // hop
    if frames == 0:
        return -120.0
    rms = np.sqrt((mono[: frames * hop].reshape(frames, hop).astype(np.float64) ** 2).mean(1))
    db = 20 * np.log10(rms + 1e-12)
    active = rms[db >= db.max() - 25]
    return float(20 * np.log10(np.sqrt((active**2).mean()) + 1e-12))


def faded(samples: Audio, fade_in: float, fade_out: float) -> Audio:
    out = samples.astype(np.float64)
    n = out.shape[1]
    a, b = min(n, round(fade_in * RATE)), min(n, round(fade_out * RATE))
    out[:, :a] *= _ramp(a)
    out[:, n - b :] *= _ramp(b)[::-1]
    return out.astype(np.float32)


def mix(song: Audio, voices: Sequence[VoiceClip]) -> Audio:
    """The song (faded in and out) ducked under every voice, each voice levelled to sit
    `VOICE_ABOVE` dB over the ducked song around it, plus its gain, and kept under `PEAK`;
    then everything scaled to `TARGET_PEAK`."""
    n = song.shape[1]
    base = faded(song, SONG_FADE_IN, SONG_FADE_OUT).astype(np.float64)
    spans = [(v.at, v.at + v.samples.shape[1] / RATE, v.duck) for v in voices]
    out = base * duck_gain(n, spans)
    for v, (start, _, _) in zip(voices, spans, strict=True):
        a = round(start * RATE)
        length = min(v.samples.shape[1], n - a)
        if a < 0 or length <= 0:
            continue
        around = base[:, max(0, a - RATE) : min(n, a + length + RATE)].astype(np.float32)
        target = level_db(around) - v.duck + VOICE_ABOVE + v.gain
        voice = faded(v.samples[:, :length], VOICE_FADE, VOICE_FADE).astype(np.float64)
        voice *= 10 ** ((target - level_db(v.samples)) / 20)
        under = np.abs(out[:, a : a + length]).max(initial=0.0)
        peak = np.abs(voice).max(initial=0.0)
        if peak > 0 and under + peak > PEAK:
            voice *= max(0.0, PEAK - under) / peak
        out[:, a : a + length] += voice
    peak = np.abs(out).max(initial=0.0)
    if peak > 0:
        out *= TARGET_PEAK / peak
    return out.astype(np.float32)
