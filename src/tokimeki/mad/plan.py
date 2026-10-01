"""The edit plan: the single source of truth for one MAD, a JSON file an agent writes.

One entry per slot of the song: which shot fills it, the source window (`in`/`out`, episode
seconds) and why; and the original lines laid over the song (`voices`). Rendering only reads
the plan. `in`/`out`/`speed` may be left out; `tokimeki plan refine` (or render) fills them
around the shot's expression peak, and a voice's from its subtitle line.
"""

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import cast

PLAN_SCHEMA = "tokimeki.plan/3"
READABLE_SCHEMAS = ("tokimeki.plan/2", PLAN_SCHEMA)
MIN_SPEED = 0.9
MAX_SPEED = 1.1
DEFAULT_FPS = "24000/1001"

type Json = dict[str, object]


class PlanFormatError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class SlotPlan:
    start: float
    end: float
    """Song seconds, on beats."""
    shot: int | None
    why: str = ""
    source_in: float | None = None
    source_out: float | None = None
    """Episode seconds of the stretch shown; it plays at (out - in) / (end - start) speed."""
    speed: float | None = None
    section: str = ""
    episode: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def derived_speed(self) -> float | None:
        if self.source_in is None or self.source_out is None or self.duration <= 0:
            return None
        return (self.source_out - self.source_in) / self.duration


@dataclass
class VoicePlan:
    at: float
    """Song seconds where the line starts."""
    line: int | None = None
    """A dialogue line id from the context; fills `episode`, `in`, `out` and `text`."""
    episode: str = ""
    source_in: float | None = None
    source_out: float | None = None
    """Episode seconds of the voice heard."""
    gain: float = 0.0
    """dB on top of the automatic level (the voice a little above the ducked song)."""
    duck: float | None = None
    """dB the song is lowered under the line; DUCK_DB when left out."""
    text: str = ""
    sub: bool = True
    """Show `text` as a dialogue subtitle."""
    force: bool = False
    """Allow the line over sung lyrics."""
    why: str = ""

    @property
    def length(self) -> float | None:
        if self.source_in is None or self.source_out is None:
            return None
        return self.source_out - self.source_in

    @property
    def end(self) -> float | None:
        length = self.length
        return None if length is None else self.at + length


@dataclass
class Plan:
    name: str
    series: str
    character: str
    song: str
    slots: list[SlotPlan] = field(default_factory=list[SlotPlan])
    fps: str = DEFAULT_FPS
    voices: list[VoicePlan] = field(default_factory=list[VoicePlan])

    @property
    def frame_rate(self) -> Fraction:
        return Fraction(self.fps)

    @property
    def start(self) -> float:
        return self.slots[0].start if self.slots else 0.0

    @property
    def end(self) -> float:
        return self.slots[-1].end if self.slots else 0.0

    def to_dict(self) -> Json:
        def slot(s: SlotPlan) -> Json:
            out: Json = {"start": round(s.start, 4), "end": round(s.end, 4), "shot": s.shot}
            if s.source_in is not None:
                out["in"] = round(s.source_in, 4)
            if s.source_out is not None:
                out["out"] = round(s.source_out, 4)
            if s.speed is not None:
                out["speed"] = round(s.speed, 4)
            out["why"] = s.why
            if s.section:
                out["section"] = s.section
            if s.episode:
                out["episode"] = s.episode
            return out

        def voice(v: VoicePlan) -> Json:
            out: Json = {"at": round(v.at, 4)}
            if v.line is not None:
                out["line"] = v.line
            if v.episode:
                out["episode"] = v.episode
            if v.source_in is not None:
                out["in"] = round(v.source_in, 4)
            if v.source_out is not None:
                out["out"] = round(v.source_out, 4)
            out["gain"] = round(v.gain, 2)
            if v.duck is not None:
                out["duck"] = round(v.duck, 2)
            out["text"] = v.text
            out["sub"] = v.sub
            if v.force:
                out["force"] = True
            out["why"] = v.why
            return out

        return {
            "schema": PLAN_SCHEMA,
            "name": self.name,
            "series": self.series,
            "character": self.character,
            "song": self.song,
            "fps": self.fps,
            "slots": [slot(s) for s in self.slots],
            "voices": [voice(v) for v in self.voices],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1)

    @staticmethod
    def from_dict(data: object) -> "Plan":
        """Parse a plan, collecting every format problem (with its JSON path) at once."""
        problems: list[str] = []
        if not isinstance(data, dict):
            raise PlanFormatError(["the plan must be a JSON object"])
        d = cast(Json, data)

        def text(obj: Json, key: str, where: str, default: str | None = None) -> str:
            value = obj.get(key, default)
            if not isinstance(value, str):
                problems.append(f"{where}{key}: expected a string")
                return ""
            return value

        def number(obj: Json, key: str, where: str, optional: bool = False) -> float | None:
            value = obj.get(key)
            if value is None and optional:
                return None
            if isinstance(value, bool) or not isinstance(value, int | float):
                problems.append(f"{where}{key}: expected a number")
                return None
            return float(value)

        def flag(obj: Json, key: str, where: str, default: bool) -> bool:
            value = obj.get(key, default)
            if not isinstance(value, bool):
                problems.append(f"{where}{key}: expected true or false")
                return default
            return value

        def integer(obj: Json, key: str, where: str) -> int | None:
            value = obj.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{where}{key}: expected an integer")
                return None
            return value

        if d.get("schema", PLAN_SCHEMA) not in READABLE_SCHEMAS:
            problems.append(f"schema: expected one of {', '.join(READABLE_SCHEMAS)}")
        plan = Plan(
            name=text(d, "name", ""),
            series=text(d, "series", ""),
            character=text(d, "character", ""),
            song=text(d, "song", ""),
            fps=text(d, "fps", "", DEFAULT_FPS),
        )
        try:
            Fraction(plan.fps)
        except (ValueError, ZeroDivisionError):
            problems.append("fps: expected a rate such as 24000/1001")
        raw_slots = d.get("slots")
        if not isinstance(raw_slots, list):
            problems.append("slots: expected a list")
            raw_slots = []
        for i, raw in enumerate(cast(list[object], raw_slots)):
            where = f"slots[{i}]."
            if not isinstance(raw, dict):
                problems.append(f"slots[{i}]: expected an object")
                continue
            s = cast(Json, raw)
            shot = s.get("shot")
            if shot is not None and (isinstance(shot, bool) or not isinstance(shot, int)):
                problems.append(f"{where}shot: expected a shot id (integer)")
                shot = None
            plan.slots.append(
                SlotPlan(
                    start=number(s, "start", where) or 0.0,
                    end=number(s, "end", where) or 0.0,
                    shot=shot,
                    why=text(s, "why", where, ""),
                    source_in=number(s, "in", where, optional=True),
                    source_out=number(s, "out", where, optional=True),
                    speed=number(s, "speed", where, optional=True),
                    section=text(s, "section", where, ""),
                    episode=text(s, "episode", where, ""),
                )
            )
        raw_voices = d.get("voices", [])
        if not isinstance(raw_voices, list):
            problems.append("voices: expected a list")
            raw_voices = []
        for i, raw in enumerate(cast(list[object], raw_voices)):
            where = f"voices[{i}]."
            if not isinstance(raw, dict):
                problems.append(f"voices[{i}]: expected an object")
                continue
            v = cast(Json, raw)
            plan.voices.append(
                VoicePlan(
                    at=number(v, "at", where) or 0.0,
                    line=integer(v, "line", where),
                    episode=text(v, "episode", where, ""),
                    source_in=number(v, "in", where, optional=True),
                    source_out=number(v, "out", where, optional=True),
                    gain=number(v, "gain", where, optional=True) or 0.0,
                    duck=number(v, "duck", where, optional=True),
                    text=text(v, "text", where, ""),
                    sub=flag(v, "sub", where, True),
                    force=flag(v, "force", where, False),
                    why=text(v, "why", where, ""),
                )
            )
        if problems:
            raise PlanFormatError(problems)
        return plan

    @staticmethod
    def from_json(text: str) -> "Plan":
        try:
            data = cast(object, json.loads(text))
        except json.JSONDecodeError as error:
            raise PlanFormatError([f"not valid JSON: {error}"]) from error
        return Plan.from_dict(data)


def read_plan(path: Path) -> Plan:
    return Plan.from_json(path.read_text(encoding="utf-8"))


def write_plan(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.to_json(), encoding="utf-8")


def mad_dir(series: Path, name: str) -> Path:
    return series / ".tokimeki" / "mads" / name


PLAN_JSON_SCHEMA: Json = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": PLAN_SCHEMA,
    "title": "tokimeki edit plan",
    "description": "One MAD: which shot fills each slot of a song excerpt, which stretch of "
    "it is shown, and why. Times in seconds. Check with `tokimeki plan validate`.",
    "type": "object",
    "required": ["name", "series", "character", "song", "slots"],
    "properties": {
        "schema": {"const": PLAN_SCHEMA},
        "name": {"type": "string", "description": "Outputs go to <series>/.tokimeki/mads/<name>/."},
        "series": {"type": "string", "description": "The series directory (absolute path)."},
        "character": {"type": "string", "description": "Named cast cluster featured, e.g. 梦梦."},
        "song": {"type": "string", "description": "Song id from `tokimeki song analyze`."},
        "fps": {"type": "string", "default": DEFAULT_FPS, "description": "Output frame rate."},
        "slots": {
            "type": "array",
            "minItems": 1,
            "description": "In order, back to back: each starts where the previous ends. The "
            "first start and last end are the excerpt. Boundaries sit on beats.",
            "items": {
                "type": "object",
                "required": ["start", "end", "shot"],
                "properties": {
                    "start": {"type": "number", "description": "Song seconds, on a beat."},
                    "end": {"type": "number", "description": "Song seconds, on a beat."},
                    "shot": {
                        "type": "integer",
                        "description": "Shot id from the context; kept shots only, each used once.",
                    },
                    "in": {
                        "type": "number",
                        "description": "Episode seconds where the clip starts, inside the shot;"
                        " optional, refine fills it.",
                    },
                    "out": {
                        "type": "number",
                        "description": "Episode seconds where the clip ends; inside the shot.",
                    },
                    "speed": {
                        "type": "number",
                        "minimum": MIN_SPEED,
                        "maximum": MAX_SPEED,
                        "description": "(out - in) / (end - start); optional, derived from in/out.",
                    },
                    "why": {
                        "type": "string",
                        "description": "One short sentence: why this shot here.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Informational: the song section.",
                    },
                    "episode": {
                        "type": "string",
                        "description": "Informational: the shot's episode.",
                    },
                },
            },
        },
        "voices": {
            "type": "array",
            "description": "Original lines laid over the song, in its gaps (no sung lyrics); "
            "the song is ducked smoothly under each. From the context's voice_lines.",
            "items": {
                "type": "object",
                "required": ["at"],
                "properties": {
                    "at": {"type": "number", "description": "Song seconds where it starts."},
                    "line": {
                        "type": "integer",
                        "description": "A voice_lines id; fills episode, in, out and text.",
                    },
                    "episode": {
                        "type": "string",
                        "description": "The episode, when giving in/out without a line.",
                    },
                    "in": {
                        "type": "number",
                        "description": "Episode seconds of the voice; refine trims a line's.",
                    },
                    "out": {"type": "number", "description": "Episode seconds."},
                    "gain": {
                        "type": "number",
                        "default": 0,
                        "description": "dB over the automatic level (a little above the ducked"
                        " song); -20..12.",
                    },
                    "duck": {
                        "type": "number",
                        "description": "dB the song goes down under it (default 9); 0..24.",
                    },
                    "text": {"type": "string", "description": "Subtitle text; the line's."},
                    "sub": {"type": "boolean", "default": True, "description": "Show text."},
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Allow it over sung lyrics.",
                    },
                    "why": {"type": "string"},
                },
            },
        },
    },
}
