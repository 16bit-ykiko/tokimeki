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
    audio: Sequence[TimelineClip],
) -> None:
    """An `.otio` timeline with one video and one audio track, cut by cut."""
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
    write = cast(Callable[[object, str], None], adapters.write_to_file)
    video_kind = cast(str, schema.TrackKind.Video)
    audio_kind = cast(str, schema.TrackKind.Audio)

    timeline = new_timeline(name=name)
    for kind, clips in ((video_kind, video), (audio_kind, audio)):
        track = new_track(name=kind, kind=kind)
        for c in clips:
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
        timeline.tracks.append(track)
    write(timeline, str(path))
