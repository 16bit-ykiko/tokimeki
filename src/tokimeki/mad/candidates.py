"""The shots a MAD may use: kept shots of the character, outside the OP/ED, with what makes
each one cute (expression tags per sampled frame) and where in the shot that peaks.

Only kept shots ever become candidates, so only they can reach a cloud model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from tokimeki.library.cast import named_cast_by_shot, named_face_heights
from tokimeki.library.lines import lines_by_shot, shots_in_parts
from tokimeki.library.records import Episode, Shot, ShotStatus, TagStat
from tokimeki.library.scenes import list_scenes
from tokimeki.library.shots import list_shots
from tokimeki.library.tags import EXPRESSION_TAGS, frame_tag_scores, tag_stats
from tokimeki.stages.base import Context

CUTE: dict[str, float] = {
    "blush": 1.0,
    "full-face_blush": 1.0,
    "light_blush": 0.6,
    "nose_blush": 0.6,
    "embarrassed": 0.9,
    "one_eye_closed": 0.9,
    "pout": 0.9,
    ":t": 0.8,
    ">_<": 0.9,
    "smile": 0.7,
    ":d": 0.8,
    "^_^": 0.8,
    "laughing": 0.6,
    "heart": 0.7,
    "spoken_heart": 0.7,
    "wavy_mouth": 0.6,
    ":3": 0.6,
    ":p": 0.6,
    "tongue_out": 0.6,
    "smug": 0.5,
    "grin": 0.4,
    "surprised": 0.3,
    ":o": 0.3,
    "tears": 0.3,
    "flying_sweatdrops": 0.3,
    "closed_eyes": 0.1,
}
"""How much each expression tag counts towards "cute"; WD14 fires `closed_eyes` on most
smiles and blinks, so it barely counts."""

MIN_SHOT_SECONDS = 0.6
CLOSE_UP = 0.35
WIDE = 0.15


def cuteness(scores: Mapping[str, float], boost: Mapping[str, float]) -> float:
    return sum(CUTE.get(tag, 0.0) * boost.get(tag, 1.0) * s for tag, s in scores.items())


def framing(face_height: float) -> str:
    if face_height >= CLOSE_UP:
        return "close-up"
    return "medium" if face_height >= WIDE else "wide"


@dataclass(frozen=True, slots=True)
class FramePoint:
    time: float
    """Episode seconds."""
    cuteness: float
    face: float
    """Height of the character's face in this frame (0 if not seen)."""
    tags: Mapping[str, float]


@dataclass(frozen=True)
class Candidate:
    episode: Episode
    shot: Shot
    scene: int
    presence: float
    face_height: float
    others: tuple[str, ...]
    expressions: tuple[TagStat, ...]
    frames: tuple[FramePoint, ...]
    lines: tuple[str, ...]

    @property
    def start(self) -> float:
        return self.episode.seconds(self.shot.start_frame)

    @property
    def end(self) -> float:
        return self.episode.seconds(self.shot.end_frame)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def framing(self) -> str:
        return framing(self.face_height)

    @cached_property
    def peak(self) -> FramePoint:
        """The cutest sampled frame the character is seen in (any frame if never seen)."""
        seen = [f for f in self.frames if f.face > 0] or list(self.frames)
        return max(seen, key=lambda f: (f.cuteness, f.face, -f.time))

    def describe(self) -> str:
        faces = ", ".join(f"{t.tag} {t.peak:.2f}" for t in self.expressions[:5]) or "neutral"
        with_ = f"; with {', '.join(self.others)}" if self.others else ""
        return f"{self.framing}, {self.presence:.0%} on screen{with_}; {faces}"


def find_candidates(
    ctx: Context,
    character: str,
    episodes: Sequence[Episode],
    boost: Mapping[str, float],
    min_presence: float,
    only: Sequence[int] | None = None,
) -> list[Candidate]:
    """Candidate shots; with `only`, exactly those kept shots (by id), whoever is in them."""
    out: list[Candidate] = []
    for episode in episodes:
        cast = named_cast_by_shot(ctx.conn, episode.id)
        parts = shots_in_parts(ctx.conn, episode.id)
        heights = named_face_heights(ctx.conn, episode.id, character)
        said = lines_by_shot(ctx.conn, episode.id)
        scene_of = {
            shot_id: scene.index
            for scene in list_scenes(ctx.conn, episode.id)
            for shot_id in scene.shot_ids
        }
        for shot in list_shots(ctx.conn, episode.id, ShotStatus.KEPT):
            here = dict(cast.get(shot.id, []))
            target = here.get(character)
            if only is not None:
                if shot.id not in only:
                    continue
            elif target is None or target.presence < min_presence or shot.id in parts:
                continue
            if (
                episode.seconds(shot.end_frame) - episode.seconds(shot.start_frame)
                < MIN_SHOT_SECONDS
            ):
                continue
            seen = heights.get(shot.id, {})
            frames = tuple(
                FramePoint(
                    episode.seconds(index), cuteness(tags, boost), seen.get(index, 0.0), tags
                )
                for index, tags in frame_tag_scores(ctx.conn, shot.id, EXPRESSION_TAGS).items()
            )
            out.append(
                Candidate(
                    episode,
                    shot,
                    scene_of.get(shot.id, -1),
                    target.presence if target else 0.0,
                    target.face_height if target else 0.0,
                    tuple(name for name in here if name != character),
                    tuple(tag_stats(ctx.conn, [shot.id])),
                    frames,
                    tuple(line.text.replace("\n", " ") for line in said.get(shot.id, [])),
                )
            )
    return out
