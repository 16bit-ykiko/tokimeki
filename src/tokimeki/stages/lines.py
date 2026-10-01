"""Stage 4: lines. The subtitle file next to an episode, checked against its speech.

Dialogue lines are stored with their timing corrected by the measured offset; OP/ED lyric
lines mark the opening and ending, which later stages leave out of the candidate pool.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.lines import NewLine, replace_lines
from tokimeki.library.records import Episode, LineKind, Part, PartKind
from tokimeki.media.audio import decode_audio, main_audio
from tokimeki.media.subtitles import SubtitleEvent, read_ass
from tokimeki.models.gpu import loaded
from tokimeki.models.vad import FRAME_SECONDS, SAMPLE_RATE, SileroVad
from tokimeki.stages.base import Context

NAME = "lines"

LYRIC_STYLE = re.compile(r"^(op|ed)", re.IGNORECASE)
DIALOGUE_STYLES = frozenset({"default", "dialogue", "main"})
PART_PAD = 2.0
"""Seconds added around the first and last lyric line of an opening or ending."""

MAX_OFFSET = 5.0
MIN_CORRELATION = 0.4
MIN_SHARPNESS = 0.1
"""The line/speech correlation must fall by this much half a second away from its peak."""
MIN_APPLIED = 0.1
"""Smaller shifts are within subtitle convention (lines lead speech a little) and VAD resolution."""
ONSET_WINDOW = 0.3

log = logging.getLogger("tokimeki")


@dataclass(frozen=True, slots=True)
class TimingCheck:
    offset: float
    correlation: float
    sharpness: float
    applied: bool
    onset_delay: float
    onsets_matched: float
    lines: int


def subtitle_file(video: Path) -> Path | None:
    path = video.with_suffix(".ass")
    return path if path.exists() else None


def classify(
    events: Sequence[SubtitleEvent],
) -> tuple[list[NewLine], dict[PartKind, list[NewLine]]]:
    """All events as lines, and the lyric lines of the opening and ending."""
    lines: list[NewLine] = []
    lyrics: dict[PartKind, list[NewLine]] = {PartKind.OPENING: [], PartKind.ENDING: []}
    for event in events:
        if not event.text or event.end <= event.start:
            continue
        style = event.style.lstrip("*").lower()
        match = LYRIC_STYLE.match(style)
        kind = LineKind.OTHER
        if match is not None:
            kind = LineKind.LYRICS
        elif style in DIALOGUE_STYLES:
            kind = LineKind.DIALOGUE
        line = NewLine(event.start, event.end, kind, event.style, event.text)
        lines.append(line)
        if match is not None:
            part = PartKind.OPENING if match.group(1).lower() == "op" else PartKind.ENDING
            lyrics[part].append(line)
    return lines, lyrics


def shifted(lines: Sequence[NewLine], offset: float) -> list[NewLine]:
    return [NewLine(x.start + offset, x.end + offset, x.kind, x.style, x.text) for x in lines]


def parts(episode_id: int, lyrics: dict[PartKind, list[NewLine]], offset: float) -> list[Part]:
    return [
        Part(
            episode_id,
            kind,
            min(x.start for x in sung) + offset - PART_PAD,
            max(x.end for x in sung) + offset + PART_PAD,
        )
        for kind, sung in lyrics.items()
        if sung
    ]


def speech_mask(lines: Sequence[NewLine], frames: int) -> NDArray[np.float32]:
    mask = np.zeros(frames, dtype=np.float32)
    for line in lines:
        first = max(0, int(line.start / FRAME_SECONDS))
        last = min(frames, int(np.ceil(line.end / FRAME_SECONDS)))
        mask[first:last] = 1.0
    return mask


def speech_onsets(
    probabilities: NDArray[np.float32], threshold: float = 0.5
) -> NDArray[np.float64]:
    speaking = probabilities >= threshold
    rising = np.flatnonzero(speaking[1:] & ~speaking[:-1]) + 1
    return (rising * FRAME_SECONDS).astype(np.float64)


def _correlations(mask: NDArray[np.float32], speech: NDArray[np.float32]) -> NDArray[np.float64]:
    """Normalised correlation of `mask` delayed by each lag (in frames) with `speech`."""
    n = len(speech)
    m, p = mask - mask.mean(), speech - speech.mean()
    norm = float(np.linalg.norm(m) * np.linalg.norm(p)) or 1.0
    reach = int(MAX_OFFSET / FRAME_SECONDS)
    return np.array(
        [
            float(np.dot(m[: n - lag], p[lag:]) if lag >= 0 else np.dot(m[-lag:], p[: n + lag]))
            / norm
            for lag in range(-reach, reach + 1)
        ]
    )


def timing_check(dialogue: Sequence[NewLine], probabilities: NDArray[np.float32]) -> TimingCheck:
    """How far the subtitles sit from the speech.

    The offset (seconds to add to every line) best lines up "a line is showing" with the
    VAD's speech probability; it is applied only when the match is good, has a clear peak
    and is large enough to matter. Also reported: how long after a line appears its speech
    starts (median), and the share of line starts within `ONSET_WINDOW` of a speech onset.
    """
    scores = _correlations(speech_mask(dialogue, len(probabilities)), probabilities)
    reach = len(scores) // 2
    best = int(scores.argmax())
    shift = float(best - reach)
    if 0 < best < len(scores) - 1:
        left, mid, right = scores[best - 1], scores[best], scores[best + 1]
        curvature = left - 2 * mid + right
        if curvature < 0:
            shift += 0.5 * (left - right) / curvature
    offset = shift * FRAME_SECONDS
    half = round(0.5 / FRAME_SECONDS)
    around = [scores[i] for i in (best - half, best + half) if 0 <= i < len(scores)]
    sharpness = float(scores[best] - np.mean(around)) if around else 0.0
    applied = (
        scores[best] >= MIN_CORRELATION
        and sharpness >= MIN_SHARPNESS
        and abs(offset) >= MIN_APPLIED
    )
    used = offset if applied else 0.0
    onsets = speech_onsets(probabilities)
    starts = np.array([x.start + used for x in dialogue])
    delay = matched = 0.0
    if len(onsets) and len(starts):
        nearest = onsets[np.abs(starts[:, None] - onsets[None, :]).argmin(axis=1)] - starts
        close = np.abs(nearest) <= ONSET_WINDOW
        matched = float(close.mean())
        delay = float(np.median(nearest[close])) if close.any() else 0.0
    return TimingCheck(
        offset, float(scores[best]), sharpness, applied, delay, matched, len(dialogue)
    )


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [e for e in episodes if not stage_done(ctx.conn, e.id, NAME)]
    with_subtitles: list[tuple[Episode, Path]] = []
    for episode in todo:
        source = subtitle_file(ctx.paths.episode_file(episode.path))
        if source is None:
            with transaction(ctx.conn):
                replace_lines(ctx.conn, episode.id, [], [])
                mark_stage_done(ctx.conn, episode.id, NAME, {"subtitles": ""})
            log.info("%s: no subtitles next to it", episode.path)
        else:
            with_subtitles.append((episode, source))
    if not with_subtitles:
        return
    with loaded("Silero VAD", SileroVad) as vad:
        for episode, source in with_subtitles:
            _import(ctx, episode, source, vad)


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        replace_lines(ctx.conn, episode.id, [], [])
        clear_stage(ctx.conn, episode.id, NAME)


def _import(ctx: Context, episode: Episode, source: Path, vad: SileroVad) -> None:
    video = ctx.paths.episode_file(episode.path)
    lines, lyrics = classify(read_ass(source))
    dialogue = [x for x in lines if x.kind is LineKind.DIALOGUE]
    stream = main_audio(video)
    audio = decode_audio(video, SAMPLE_RATE, stream.index)
    check = timing_check(dialogue, vad.speech_probabilities(audio))
    offset = check.offset if check.applied else 0.0
    with transaction(ctx.conn):
        replace_lines(
            ctx.conn, episode.id, shifted(lines, offset), parts(episode.id, lyrics, offset)
        )
        mark_stage_done(
            ctx.conn,
            episode.id,
            NAME,
            {
                "subtitles": source.name,
                "audio_stream": stream.index,
                "measured_offset": round(check.offset, 3),
                "correlation": round(check.correlation, 3),
                "sharpness": round(check.sharpness, 3),
                "applied_offset": round(offset, 3),
                "speech_onset_delay": round(check.onset_delay, 3),
                "onsets_within_0.3s": round(check.onsets_matched, 3),
            },
        )
    log.info(
        "%s: %d lines (%d dialogue); best match %+.3fs (correlation %.2f, sharpness %.2f, %s);"
        " speech starts %.3fs after a line, %.0f%% of line starts within %.1fs of an onset",
        episode.path,
        len(lines),
        len(dialogue),
        check.offset,
        check.correlation,
        check.sharpness,
        f"shifted {offset:+.3f}s" if check.applied else "not shifted",
        check.onset_delay,
        100 * check.onsets_matched,
        ONSET_WINDOW,
    )
