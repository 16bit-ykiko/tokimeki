"""Song structure from beats and spectrograms: a bar grid, sections and their roles.

Pure numpy over arrays the GPU models produced, so it is tested without audio. The vocal
line comes from the CD's instrumental version when there is one: whatever the full mix
has on top of the instrumental is the voice.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

BEATS_PER_BAR = 4
VOCAL_BAND = (300.0, 4000.0)
VOCAL_ON = 0.2
"""Share of a bar's vocal-band energy that must come from the voice for the bar to count as sung."""
NOVELTY_HALF_WIDTH = 4
MIN_SECTION_BARS = 4
SAME_SECTION = 0.85


@dataclass(frozen=True, slots=True)
class Section:
    label: str
    """intro, verse, pre-chorus, chorus, interlude, bridge or outro."""
    group: int
    """Sections with the same group repeat the same material."""
    first_bar: int
    end_bar: int
    start: float
    end: float
    vocal: float
    loudness: float

    @property
    def bars(self) -> int:
        return self.end_bar - self.first_bar


def bar_grid(beats: NDArray[np.float64], downbeats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bar start times for 4/4: every fourth beat, in the phase most detected downbeats agree on."""
    if len(beats) < BEATS_PER_BAR:
        return beats[:1]
    indices = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
    phases = Counter(int(i) % BEATS_PER_BAR for i in indices)
    phase = phases.most_common(1)[0][0] if phases else 0
    return beats[phase::BEATS_PER_BAR]


def per_bar(
    values: NDArray[np.float64], fps: float, bars: NDArray[np.float64], end: float
) -> NDArray[np.float64]:
    """Mean of a framewise array (frames first) over each bar."""
    edges = np.append(bars, end)
    out: list[NDArray[np.float64]] = []
    for a, b in pairwise(edges):
        lo, hi = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        out.append(values[lo:hi].mean(axis=0))
    return np.array(out, dtype=np.float64)


def vocal_share(
    mix: NDArray[np.float64], instrumental: NDArray[np.float64], band: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Per frame, the share of vocal-band energy in the mix that the instrumental lacks.

    Both are linear mel magnitudes `(frames, bins)` of aligned audio.
    """
    extra = np.maximum(mix[:, band] - instrumental[:, band], 0.0).sum(axis=1)
    return extra / (mix[:, band].sum(axis=1) + 1e-9)


def best_lag(a: NDArray[np.float64], b: NDArray[np.float64], reach: int) -> int:
    """The shift (frames) of `b` that best matches `a`: `a[t]` ~ `b[t - lag]`."""
    a0, b0 = a - a.mean(), b - b.mean()
    n = min(len(a0), len(b0))
    scores = [
        float(np.dot(a0[lag:n], b0[: n - lag]) if lag >= 0 else np.dot(a0[: n + lag], b0[-lag:n]))
        for lag in range(-reach, reach + 1)
    ]
    return int(np.argmax(scores)) - reach


def novelty(features: NDArray[np.float64], half: int = NOVELTY_HALF_WIDTH) -> NDArray[np.float64]:
    """Foote novelty: a checkerboard kernel slid along the bar self-similarity diagonal."""
    z = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    sim = z @ z.T
    sign = np.sign(np.arange(-half, half) + 0.5)
    kernel = np.outer(sign, sign)
    n = len(sim)
    padded = np.pad(sim, half, mode="edge")
    return np.array(
        [float((padded[i : i + 2 * half, i : i + 2 * half] * kernel).sum()) for i in range(n)]
    )


def boundaries(
    novelty_curve: NDArray[np.float64],
    sung: NDArray[np.bool_],
    min_bars: int = MIN_SECTION_BARS,
    refrain: NDArray[np.bool_] | None = None,
) -> list[int]:
    """Section starts (bar indices): where a refrain starts or stops, where singing starts or
    stops, and novelty peaks, in that order of trust."""
    n = len(novelty_curve)
    candidates: list[tuple[float, int]] = []
    for marks, weight in ((refrain, 2 * np.inf), (sung, np.inf)):
        if marks is None:
            continue
        for i in range(1, n):
            if marks[i] != marks[i - 1] and bool((marks[i : i + 2] == marks[i]).all()):
                candidates.append((weight, i))
    threshold = novelty_curve.mean() + 0.5 * novelty_curve.std()
    for i in range(1, n - 1):
        v = novelty_curve[i]
        if v >= threshold and v >= novelty_curve[i - 1] and v >= novelty_curve[i + 1]:
            candidates.append((float(v), i))
    chosen: list[int] = [0]
    for _, i in sorted(candidates, key=lambda c: -c[0]):
        if all(abs(i - j) >= min_bars for j in chosen) and n - i >= min_bars // 2:
            chosen.append(i)
    return sorted(chosen)


def label_sections(
    starts: list[int],
    features: NDArray[np.float64],
    vocal: NDArray[np.float64],
    loudness: NDArray[np.float64],
    bars: NDArray[np.float64],
    end: float,
    vocal_known: bool = True,
    vocal_on: float = VOCAL_ON,
    refrain: NDArray[np.bool_] | None = None,
) -> list[Section]:
    """Name the sections: repeated loud sung material is the chorus, what leads into it the
    pre-chorus, unsung parts intro/interlude/outro, the rest verses (bridge if heard once, late).

    Without a vocal line (`vocal_known` false) only a quieter first or last section becomes
    the intro or outro. With `refrain` (bars whose lyrics are sung more than once) the
    sections mostly made of it are the chorus."""
    n = len(bars)
    spans = list(zip(starts, [*starts[1:], n], strict=True))
    means = np.array([features[a:b].mean(axis=0) for a, b in spans])
    z = (means - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    groups: list[int] = []
    for i in range(len(spans)):
        match = next((groups[j] for j in range(i) if float(z[i] @ z[j]) >= SAME_SECTION), None)
        groups.append(match if match is not None else max(groups, default=-1) + 1)
    sung = [float(vocal[a:b].mean()) >= vocal_on for a, b in spans]
    loud = [float(loudness[a:b].mean()) for a, b in spans]
    counts = Counter(g for g, s in zip(groups, sung, strict=True) if s)
    repeated = [g for g, c in counts.items() if c >= 2] or list(counts)

    def group_loudness(g: int) -> float:
        return float(np.mean([v for v, x in zip(loud, groups, strict=True) if x == g]))

    chorus = max(repeated, key=group_loudness, default=-1)
    if refrain is not None and refrain.any():
        chorus = -1
    first_sung = next((i for i, s in enumerate(sung) if s), len(spans))
    last_sung = max((i for i, s in enumerate(sung) if s), default=-1)
    if not vocal_known and len(spans) > 2:
        quiet = float(np.mean(loud))
        sung[0] = loud[0] >= quiet
        sung[-1] = loud[-1] >= quiet
        first_sung = 0 if sung[0] else 1
        last_sung = len(spans) - 1 if sung[-1] else len(spans) - 2

    def is_chorus(i: int) -> bool:
        a, b = spans[i]
        if refrain is not None and refrain.any():
            return sung[i] and float(refrain[a:b].mean()) >= 0.5
        return sung[i] and groups[i] == chorus

    labels: list[str] = []
    for i, group in enumerate(groups):
        if not sung[i]:
            labels.append("intro" if i < first_sung else "outro" if i > last_sung else "interlude")
        elif group == chorus or (
            refrain is not None and float(refrain[spans[i][0] : spans[i][1]].mean()) >= 0.5
        ):
            labels.append("chorus")
        elif i + 1 < len(groups) and is_chorus(i + 1) and spans[i][1] - spans[i][0] <= 8:
            labels.append("pre-chorus")
        elif counts[group] == 1 and i > len(spans) // 2:
            labels.append("bridge")
        else:
            labels.append("verse")
    edges = np.append(bars, end)
    return [
        Section(
            labels[i],
            groups[i],
            a,
            b,
            float(edges[a]),
            float(edges[b]),
            float(vocal[a:b].mean()),
            loud[i],
        )
        for i, (a, b) in enumerate(spans)
    ]


def merge_runs(sections: list[Section]) -> list[Section]:
    """Join neighbouring sections that got the same label."""
    out: list[Section] = []
    for section in sections:
        last = out[-1] if out else None
        if last is not None and last.label == section.label:
            bars = last.bars + section.bars
            out[-1] = Section(
                last.label,
                last.group,
                last.first_bar,
                section.end_bar,
                last.start,
                section.end,
                (last.vocal * last.bars + section.vocal * section.bars) / bars,
                (last.loudness * last.bars + section.loudness * section.bars) / bars,
            )
        else:
            out.append(section)
    return out
