"""Timed lyrics: from LRC files, or from ASS subtitles filtered by style (an OP's opjp/opcn
lines), in one or more languages.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tokimeki.media.subtitles import read_ass

_STAMP = re.compile(r"\[(\d+):(\d{1,2}(?:[.:]\d{1,3})?)\]")
_OFFSET = re.compile(r"\[offset:\s*([+-]?\d+)\]", re.IGNORECASE)
_WORD_STAMP = re.compile(r"<\d+:\d{1,2}(?:[.:]\d{1,3})?>")
MAX_LINE = 10.0
LAST_LINE = 5.0


@dataclass(frozen=True, slots=True)
class LyricLine:
    start: float
    end: float
    text: str
    lang: str = ""


@dataclass(frozen=True, slots=True)
class LyricSource:
    """Where lyrics come from: `PATH[#STYLE,STYLE][@LANG]` on the command line."""

    path: Path
    styles: tuple[str, ...] = ()
    lang: str = ""

    @staticmethod
    def parse(spec: str) -> "LyricSource":
        rest, lang = spec, ""
        if spec.rfind("@") > spec.rfind("/"):
            rest, lang = spec[: spec.rfind("@")], spec[spec.rfind("@") + 1 :]
        path, styles = rest, ""
        if rest.rfind("#") > rest.rfind("/"):
            path, styles = rest[: rest.rfind("#")], rest[rest.rfind("#") + 1 :]
        return LyricSource(
            Path(path).expanduser(),
            tuple(s.strip() for s in styles.split(",") if s.strip()),
            lang.strip(),
        )

    def key(self) -> str:
        stamp = self.path.stat().st_mtime_ns if self.path.exists() else 0
        return f"{self.path.resolve()}@{stamp}#{','.join(self.styles)}@{self.lang}"


def _seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds.replace(":", "."))


def parse_lrc(content: str) -> list[LyricLine]:
    """Timed lines in order; a line ends where the next begins (at most `MAX_LINE` later).

    Lines with several stamps repeat; an empty stamped line only ends the one before it;
    `[offset:ms]` shifts everything (positive means earlier, as in the LRC convention).
    """
    shift = 0.0
    stamped: list[tuple[float, str]] = []
    for raw in content.lstrip("﻿").splitlines():
        offset = _OFFSET.search(raw)
        if offset:
            shift = int(offset.group(1)) / 1000
            continue
        stamps = _STAMP.findall(raw)
        if not stamps:
            continue
        text = _WORD_STAMP.sub("", _STAMP.sub("", raw)).strip()
        stamped += [(_seconds(m, s), text) for m, s in stamps]
    stamped.sort(key=lambda item: item[0])
    lines: list[LyricLine] = []
    for i, (start, text) in enumerate(stamped):
        if not text:
            continue
        following = stamped[i + 1][0] if i + 1 < len(stamped) else start + LAST_LINE
        end = min(following, start + MAX_LINE)
        lines.append(LyricLine(start - shift, end - shift, text))
    return lines


def read_lrc(path: Path) -> list[LyricLine]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "shift_jis"):
        try:
            return parse_lrc(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: cannot tell the text encoding")


def read_ass_lyrics(path: Path, styles: Sequence[str], offset: float) -> list[LyricLine]:
    """ASS lines of the given styles (all when none are given), times moved from the media's
    clock onto the song's by subtracting `offset`."""
    wanted = {s.lower() for s in styles}
    return [
        LyricLine(e.start - offset, e.end - offset, e.text.replace("\n", " "))
        for e in read_ass(path)
        if (not wanted or e.style.lstrip("*").lower() in wanted) and e.text
    ]


def read_lyrics(source: LyricSource, offset: float) -> list[LyricLine]:
    """Lines of a lyrics source on the song's clock. LRC times are already the song's; ASS
    times are the media's, so `offset` (where the song starts in it) is subtracted."""
    if source.path.suffix.lower() in (".ass", ".ssa"):
        lines = read_ass_lyrics(source.path, source.styles, offset)
    else:
        lines = read_lrc(source.path)
    return sorted(
        (LyricLine(x.start, x.end, x.text, source.lang) for x in lines), key=lambda x: x.start
    )


@dataclass(frozen=True, slots=True)
class LyricPair:
    """One sung line in every language given, keyed by language."""

    start: float
    end: float
    texts: dict[str, str]


def pair(lines: Sequence[LyricLine], tolerance: float = 0.3) -> list[LyricPair]:
    """Lines of different languages that start together become one entry."""
    pairs: list[LyricPair] = []
    for line in sorted(lines, key=lambda x: x.start):
        last = pairs[-1] if pairs else None
        if (
            last is not None
            and abs(last.start - line.start) <= tolerance
            and line.lang not in last.texts
        ):
            last.texts[line.lang] = line.text
            pairs[-1] = LyricPair(last.start, max(last.end, line.end), last.texts)
        else:
            pairs.append(LyricPair(line.start, line.end, {line.lang: line.text}))
    return pairs
