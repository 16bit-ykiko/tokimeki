from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


class ShotStatus(StrEnum):
    PENDING = "pending"
    KEPT = "kept"
    DROPPED = "dropped"


class LineKind(StrEnum):
    DIALOGUE = "dialogue"
    LYRICS = "lyrics"
    OTHER = "other"


class PartKind(StrEnum):
    OPENING = "op"
    ENDING = "ed"


class TagCategory(StrEnum):
    GENERAL = "general"
    CHARACTER = "character"


@dataclass(frozen=True, slots=True)
class Episode:
    id: int
    path: str
    width: int
    height: int
    fps: Fraction
    frame_count: int

    def seconds(self, frame: int) -> float:
        return float(frame / self.fps)


@dataclass(frozen=True, slots=True)
class Shot:
    id: int
    episode_id: int
    index: int
    start_frame: int
    end_frame: int
    status: ShotStatus

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class Rating:
    general: float
    sensitive: float
    questionable: float
    explicit: float


@dataclass(frozen=True, slots=True)
class Tag:
    name: str
    category: TagCategory
    score: float


@dataclass(frozen=True, slots=True)
class Frame:
    id: int
    shot_id: int
    frame_index: int
    rating: Rating


@dataclass(frozen=True, slots=True)
class Box:
    """A box in coordinates normalised to the frame: (0, 0) top left, (1, 1) bottom right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True, slots=True)
class Face:
    id: int
    frame_id: int
    box: Box
    score: float
    cluster_id: int | None


@dataclass(frozen=True, slots=True)
class Cluster:
    id: int
    name: str | None
    face_count: int

    @property
    def label(self) -> str:
        return self.name or f"#{self.id}"


@dataclass(frozen=True, slots=True)
class ShotCast:
    """One character in one shot.

    `presence` is the share of the shot's sampled frames the character appears in,
    `face_height` the largest face height and `face_area` the mean face area, both as
    fractions of the frame.
    """

    shot_id: int
    cluster_id: int
    presence: float
    face_height: float
    face_area: float


@dataclass(frozen=True, slots=True)
class Line:
    """A subtitle line; times are seconds into the episode, already corrected for any offset."""

    id: int
    episode_id: int
    start: float
    end: float
    kind: LineKind
    style: str
    text: str


@dataclass(frozen=True, slots=True)
class Part:
    """An opening or ending sequence of an episode, in seconds."""

    episode_id: int
    kind: PartKind
    start: float
    end: float
