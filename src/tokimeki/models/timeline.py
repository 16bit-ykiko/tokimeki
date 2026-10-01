"""OpenTimelineIO (untyped) behind a small typed writer, for handing a MAD to an editor."""

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class TimelineClip:
    name: str
    media: Path
    start: float
    """Seconds into the media where the clip starts."""
    frames: int
    """Length on the timeline, in timeline frames."""
    speed: float = 1.0
    at: int | None = None
    """Timeline frame where it starts (a gap before it if needed); None: after the last."""


class _Children(Protocol):
    def append(self, child: object) -> None: ...


class _Composable(Protocol):
    @property
    def effects(self) -> _Children: ...


class _Track(Protocol):
    def append(self, child: object) -> None: ...


class _Timeline(Protocol):
    @property
    def tracks(self) -> _Track: ...


def write_timeline(
    path: Path,
    name: str,
    rate: float,
    video: Sequence[TimelineClip],
    audio: Sequence[Sequence[TimelineClip]],
) -> None:
    """An `.otio` timeline with one video track, cut by cut, and the given audio tracks."""
    opentime = importlib.import_module("opentimelineio.opentime")
    schema = importlib.import_module("opentimelineio.schema")
    adapters = importlib.import_module("opentimelineio.adapters")
    rational = cast(Callable[[float, float], object], opentime.RationalTime)
    time_range = cast(Callable[[object, object], object], opentime.TimeRange)
    new_timeline = cast(Callable[..., _Timeline], schema.Timeline)
    new_track = cast(Callable[..., _Track], schema.Track)
    new_clip = cast(Callable[..., _Composable], schema.Clip)
    new_reference = cast(Callable[..., object], schema.ExternalReference)
    new_warp = cast(Callable[..., object], schema.LinearTimeWarp)
    new_gap = cast(Callable[..., object], schema.Gap)
    write = cast(Callable[[object, str], None], adapters.write_to_file)
    video_kind = cast(str, schema.TrackKind.Video)
    audio_kind = cast(str, schema.TrackKind.Audio)

    timeline = new_timeline(name=name)
    tracks = [(video_kind, video), *((audio_kind, clips) for clips in audio)]
    for kind, clips in tracks:
        track = new_track(name=kind, kind=kind)
        position = 0
        for c in clips:
            if c.at is not None and c.at > position:
                span = time_range(rational(0, rate), rational(c.at - position, rate))
                track.append(new_gap(source_range=span))
                position = c.at
            clip = new_clip(
                name=c.name,
                media_reference=new_reference(target_url=c.media.resolve().as_uri()),
                source_range=time_range(
                    rational(round(c.start * rate), rate), rational(c.frames, rate)
                ),
            )
            if abs(c.speed - 1.0) > 1e-6:
                clip.effects.append(new_warp(time_scalar=c.speed))
            track.append(clip)
            position += c.frames
        timeline.tracks.append(track)
    write(timeline, str(path))
