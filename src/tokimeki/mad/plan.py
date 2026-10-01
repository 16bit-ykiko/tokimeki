"""The edit plan: the single source of truth for one MAD, stored as JSON.

Arrangement fills the clips (which shot goes in which slot, and why); cut placement fills
each clip's source window and speed; rendering only reads the plan. Editing the JSON by
hand and rendering again re-renders only the clips that changed.
"""

import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import cast

PLAN_VERSION = 1
MIN_SPEED = 0.9
MAX_SPEED = 1.1


@dataclass
class Preferences:
    """What the MAD was asked to be; the knobs a request in plain words turns."""

    character: str
    song: str
    track: int | None = None
    excerpt: tuple[float, float] | None = None
    """Song seconds; None picks the top of the song through the first chorus."""
    beats_per_slot: dict[str, int] = field(default_factory=dict[str, int])
    """Per section kind ("chorus": 2 …); unset kinds slow down only as far as the shots require."""
    boost: dict[str, float] = field(default_factory=dict[str, float])
    """Multipliers on expression tags ("blush": 2.0 favours blushing shots)."""
    min_presence: float = 0.34
    guidance: str = ""
    """Free text for a model arranger."""


@dataclass
class SongRef:
    id: str
    title: str
    path: str
    offset: float
    start: float
    end: float
    bpm: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PlanSlot:
    index: int
    start: float
    end: float
    section: str
    beats: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Clip:
    slot: int
    episode: str
    shot: int
    reason: str
    source_start: float | None = None
    """Episode seconds where the clip starts; set by cut placement."""
    speed: float | None = None
    """Playback speed: the clip shows `slot duration * speed` seconds of the source."""
    peak: float | None = None
    """Episode second of the expression peak the window is built around."""


@dataclass
class Plan:
    name: str
    preferences: Preferences
    song: SongRef
    slots: list[PlanSlot]
    clips: list[Clip]
    arranger: str
    fps_num: int = 24000
    fps_den: int = 1001
    version: int = PLAN_VERSION

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    def clip_for(self, slot: int) -> Clip:
        return next(c for c in self.clips if c.slot == slot)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> "Plan":
        d = cast(dict[str, object], json.loads(text))
        p = cast(dict[str, object], d["preferences"])
        excerpt = cast(list[float] | None, p.get("excerpt"))
        song = cast(dict[str, object], d["song"])
        return Plan(
            name=str(d["name"]),
            preferences=Preferences(
                character=str(p["character"]),
                song=str(p["song"]),
                track=cast(int | None, p.get("track")),
                excerpt=(float(excerpt[0]), float(excerpt[1])) if excerpt else None,
                beats_per_slot=cast(dict[str, int], p.get("beats_per_slot", {})),
                boost=cast(dict[str, float], p.get("boost", {})),
                min_presence=float(cast(float, p.get("min_presence", 0.34))),
                guidance=str(p.get("guidance", "")),
            ),
            song=SongRef(
                str(song["id"]),
                str(song["title"]),
                str(song["path"]),
                float(cast(float, song["offset"])),
                float(cast(float, song["start"])),
                float(cast(float, song["end"])),
                float(cast(float, song["bpm"])),
            ),
            slots=[
                PlanSlot(
                    int(cast(int, s["index"])),
                    float(cast(float, s["start"])),
                    float(cast(float, s["end"])),
                    str(s["section"]),
                    int(cast(int, s["beats"])),
                )
                for s in cast(list[dict[str, object]], d["slots"])
            ],
            clips=[
                Clip(
                    int(cast(int, c["slot"])),
                    str(c["episode"]),
                    int(cast(int, c["shot"])),
                    str(c["reason"]),
                    cast(float | None, c.get("source_start")),
                    cast(float | None, c.get("speed")),
                    cast(float | None, c.get("peak")),
                )
                for c in cast(list[dict[str, object]], d["clips"])
            ],
            arranger=str(d["arranger"]),
            fps_num=int(cast(int, d.get("fps_num", 24000))),
            fps_den=int(cast(int, d.get("fps_den", 1001))),
            version=int(cast(int, d.get("version", PLAN_VERSION))),
        )


def problems(plan: Plan) -> list[str]:
    """What is wrong with a plan on its own terms (the library is checked elsewhere)."""
    found: list[str] = []
    slot_ids = [s.index for s in plan.slots]
    if slot_ids != list(range(len(plan.slots))):
        found.append("slots must be numbered 0, 1, 2 … in order")
    for a, b in zip(plan.slots, plan.slots[1:], strict=False):
        if abs(a.end - b.start) > 1e-6:
            found.append(f"slot {b.index} does not start where slot {a.index} ends")
    covered = [c.slot for c in plan.clips]
    for index in slot_ids:
        if covered.count(index) != 1:
            found.append(f"slot {index} has {covered.count(index)} clips, not 1")
    unknown = set(covered) - set(slot_ids)
    if unknown:
        found.append(f"clips for slots that do not exist: {sorted(unknown)}")
    shots = [(c.episode, c.shot) for c in plan.clips]
    repeated = {s for s in shots if shots.count(s) > 1}
    if repeated:
        found.append(f"shots used more than once: {sorted(s[1] for s in repeated)}")
    for clip in plan.clips:
        if clip.speed is not None and not MIN_SPEED - 1e-9 <= clip.speed <= MAX_SPEED + 1e-9:
            found.append(
                f"slot {clip.slot}: speed {clip.speed:.3f} outside {MIN_SPEED}-{MAX_SPEED}"
            )
    return found


def plan_dir(data_dir: Path, name: str) -> Path:
    return data_dir / "mads" / name
