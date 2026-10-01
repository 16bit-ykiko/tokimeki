"""Stage 5: scenes. Consecutive kept shots merged into one exchange or moment.

In a romantic comedy the cute moment is usually an exchange and a reaction, not one shot.
Dialogue carries most of the continuity; how the frames look only merges shots when the
colours clearly match. Scenes never span a dropped shot or the edge of the OP/ED.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.cast import named_cast_by_shot
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.lines import lines_by_shot, shots_in_parts
from tokimeki.library.records import Episode, Line, PartKind, Scene, ShotStatus, TagStat
from tokimeki.library.scenes import list_scenes, replace_scenes
from tokimeki.library.shots import list_frames, list_shots
from tokimeki.library.tags import tag_stats
from tokimeki.media.images import load_image
from tokimeki.stages.base import Context

NAME = "scenes"

LINE_GAP = 1.5
"""Two shots continue one exchange when the speech pauses at most this long between them."""
SAME_LOOK = 0.75
SAME_CAST_LOOK = 0.6
"""Colour-histogram overlap that merges without dialogue (lower with a named character in both)."""
MAX_SCENE_SECONDS = 90.0
HISTOGRAM_LONG_SIDE = 160

log = logging.getLogger("tokimeki")


@dataclass(frozen=True, slots=True)
class ShotFacts:
    shot_id: int
    index: int
    start: float
    end: float
    kept: bool
    part: PartKind | None
    line_ids: frozenset[int]
    first_line: float | None
    last_line: float | None
    cast: frozenset[str]
    colours: NDArray[np.float64]


def colour_histogram(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    """64-bin RGB histogram (4 levels a channel), normalised."""
    levels = pixels // 64
    bins = levels[..., 0].astype(np.int64) * 16 + levels[..., 1] * 4 + levels[..., 2]
    return np.bincount(bins.ravel(), minlength=64) / bins.size


def continues(a: ShotFacts, b: ShotFacts) -> bool:
    """Whether kept shot `b`, right after `a`, carries on the same scene."""
    if not (a.kept and b.kept) or b.index != a.index + 1 or a.part != b.part:
        return False
    if a.line_ids & b.line_ids:
        return True
    if (
        a.last_line is not None
        and b.first_line is not None
        and b.first_line - a.last_line <= LINE_GAP
    ):
        return True
    overlap = float(np.minimum(a.colours, b.colours).sum())
    return overlap >= SAME_LOOK or (bool(a.cast & b.cast) and overlap >= SAME_CAST_LOOK)


def group(facts: Sequence[ShotFacts]) -> list[list[ShotFacts]]:
    """Kept shots grouped into scenes, in order."""
    scenes: list[list[ShotFacts]] = []
    for shot in facts:
        if not shot.kept:
            continue
        current = scenes[-1] if scenes else None
        if (
            current is not None
            and continues(current[-1], shot)
            and shot.end - current[0].start <= MAX_SCENE_SECONDS
        ):
            current.append(shot)
        else:
            scenes.append([shot])
    return scenes


def _facts(ctx: Context, episode: Episode) -> list[ShotFacts]:
    lines = lines_by_shot(ctx.conn, episode.id)
    parts = shots_in_parts(ctx.conn, episode.id)
    cast = named_cast_by_shot(ctx.conn, episode.id)
    facts: list[ShotFacts] = []
    for shot in list_shots(ctx.conn, episode.id):
        kept = shot.status is ShotStatus.KEPT
        colours = np.zeros(64)
        if kept:
            histograms = [
                colour_histogram(
                    np.asarray(
                        load_image(
                            ctx.paths.frame_path(episode.id, f.frame_index), HISTOGRAM_LONG_SIDE
                        )
                    )
                )
                for f in list_frames(ctx.conn, shot.id)
            ]
            colours = np.mean(histograms, axis=0) if histograms else colours
        said = lines.get(shot.id, [])
        facts.append(
            ShotFacts(
                shot.id,
                shot.index,
                episode.seconds(shot.start_frame),
                episode.seconds(shot.end_frame),
                kept,
                parts.get(shot.id),
                frozenset(x.id for x in said),
                min((x.start for x in said), default=None),
                max((x.end for x in said), default=None),
                frozenset(name for name, _ in cast.get(shot.id, [])),
                colours,
            )
        )
    return facts


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    for episode in episodes:
        ready = all(stage_done(ctx.conn, episode.id, s) for s in ("cast", "lines"))
        if not ready or stage_done(ctx.conn, episode.id, NAME):
            continue
        scenes = group(_facts(ctx, episode))
        with transaction(ctx.conn):
            replace_scenes(ctx.conn, episode.id, [[f.shot_id for f in s] for s in scenes])
            mark_stage_done(
                ctx.conn,
                episode.id,
                NAME,
                {"line_gap": LINE_GAP, "same_look": SAME_LOOK, "same_cast_look": SAME_CAST_LOOK},
            )
        sizes = [len(s) for s in scenes]
        log.info(
            "%s: %d scenes from %d kept shots (longest %d shots)",
            episode.path,
            len(scenes),
            sum(sizes),
            max(sizes, default=0),
        )


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        replace_scenes(ctx.conn, episode.id, [])
        clear_stage(ctx.conn, episode.id, NAME)


@dataclass(frozen=True, slots=True)
class SceneSummary:
    scene: Scene
    start: float
    end: float
    cast: list[tuple[str, float]]
    lines: list[Line]
    expressions: list[TagStat]


def summaries(ctx: Context, episode: Episode) -> list[SceneSummary]:
    """Each scene with its named cast (share of its time on screen), dialogue and expressions."""
    shots = {s.id: s for s in list_shots(ctx.conn, episode.id)}
    cast = named_cast_by_shot(ctx.conn, episode.id)
    said = lines_by_shot(ctx.conn, episode.id)
    out: list[SceneSummary] = []
    for scene in list_scenes(ctx.conn, episode.id):
        members = [shots[i] for i in scene.shot_ids]
        total = sum(s.frame_count for s in members)
        screen: dict[str, float] = {}
        for s in members:
            for name, entry in cast.get(s.id, []):
                screen[name] = screen.get(name, 0.0) + entry.presence * s.frame_count / total
        lines = {x.id: x for s in members for x in said.get(s.id, [])}
        out.append(
            SceneSummary(
                scene,
                episode.seconds(members[0].start_frame),
                episode.seconds(members[-1].end_frame),
                sorted(screen.items(), key=lambda item: -item[1]),
                sorted(lines.values(), key=lambda x: x.start),
                tag_stats(ctx.conn, scene.shot_ids),
            )
        )
    return out
