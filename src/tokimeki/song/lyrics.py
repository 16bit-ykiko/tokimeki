"""LRC lyrics: timed lines, read from `[mm:ss.xx]text` files."""

import re
from dataclasses import dataclass
from pathlib import Path

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
