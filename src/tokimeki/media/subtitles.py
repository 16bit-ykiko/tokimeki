"""Reading ASS/SSA subtitle files."""

import re
from dataclasses import dataclass
from pathlib import Path

_OVERRIDE = re.compile(r"\{[^}]*\}")
_TIME = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d+))?$")


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    start: float
    end: float
    style: str
    actor: str
    text: str


def ass_time(value: str) -> float:
    match = _TIME.match(value.strip())
    if match is None:
        raise ValueError(f"not an ASS time: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    frac = int(fraction) / 10 ** len(fraction) if fraction else 0.0
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + frac


def plain_text(text: str) -> str:
    """Drop override blocks (`{\\pos(…)}` …) and turn ASS line breaks into real ones."""
    text = _OVERRIDE.sub("", text)
    return text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ").strip()


def parse_ass(content: str) -> list[SubtitleEvent]:
    """The `Dialogue:` events of an ASS script, in file order (comments are skipped)."""
    events: list[SubtitleEvent] = []
    fields: list[str] | None = None
    in_events = False
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        key, _, value = line.partition(":")
        if key == "Format":
            fields = [name.strip().lower() for name in value.split(",")]
        elif key == "Dialogue" and fields is not None:
            parts = value.split(",", len(fields) - 1)
            if len(parts) != len(fields):
                continue
            row = dict(zip(fields, (p.strip() for p in parts), strict=True))
            events.append(
                SubtitleEvent(
                    ass_time(row["start"]),
                    ass_time(row["end"]),
                    row.get("style", ""),
                    row.get("name", row.get("actor", "")),
                    plain_text(row.get("text", "")),
                )
            )
    return events


def read_ass(path: Path) -> list[SubtitleEvent]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return parse_ass(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: cannot tell the text encoding")
