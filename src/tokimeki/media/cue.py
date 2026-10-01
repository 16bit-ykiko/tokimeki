"""CUE sheets: where each track of a CD image starts and ends."""

import re
from dataclasses import dataclass
from pathlib import Path

_FRAMES_PER_SECOND = 75
_QUOTED = re.compile(r'"(.*)"')


@dataclass(frozen=True, slots=True)
class CueTrack:
    number: int
    title: str
    performer: str
    start: float
    """INDEX 01, in seconds into the image."""
    end: float | None
    """Where the next track's pregap (INDEX 00) or INDEX 01 starts; None for the last track."""


@dataclass(frozen=True, slots=True)
class CueSheet:
    file: Path
    title: str
    performer: str
    tracks: list[CueTrack]

    def track(self, number: int) -> CueTrack:
        for track in self.tracks:
            if track.number == number:
                return track
        raise KeyError(f"no track {number} in {self.file}")


def cue_time(value: str) -> float:
    minutes, seconds, frames = (int(x) for x in value.split(":"))
    return minutes * 60 + seconds + frames / _FRAMES_PER_SECOND


def _value(rest: str) -> str:
    match = _QUOTED.search(rest)
    return match.group(1) if match else rest.strip()


def parse_cue(content: str, directory: Path) -> CueSheet:
    file = ""
    album_title = album_performer = ""
    raw: list[dict[str, str | float]] = []
    for line in content.lstrip("﻿").splitlines():
        command, _, rest = line.strip().partition(" ")
        command = command.upper()
        if command == "FILE":
            file = _value(rest.rsplit(" ", 1)[0])
        elif command == "TRACK":
            raw.append({"number": float(rest.split()[0])})
        elif command in ("TITLE", "PERFORMER"):
            key = command.lower()
            if raw:
                raw[-1][key] = _value(rest)
            elif command == "TITLE":
                album_title = _value(rest)
            else:
                album_performer = _value(rest)
        elif command == "INDEX" and raw:
            number, time = rest.split()
            raw[-1][f"index{int(number):02d}"] = cue_time(time)
    tracks: list[CueTrack] = []
    for i, entry in enumerate(raw):
        following = raw[i + 1] if i + 1 < len(raw) else None
        end = None
        if following is not None:
            end = float(following.get("index00", following["index01"]))
        tracks.append(
            CueTrack(
                int(entry["number"]),
                str(entry.get("title", "")),
                str(entry.get("performer", album_performer)),
                float(entry["index01"]),
                end,
            )
        )
    return CueSheet(directory / file, album_title, album_performer, tracks)


def read_cue(path: Path) -> CueSheet:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "shift_jis", "gb18030"):
        try:
            return parse_cue(data.decode(encoding), path.parent)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: cannot tell the text encoding")
