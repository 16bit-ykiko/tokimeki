from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tokimeki.library.episodes import list_episodes, stage_done
from tokimeki.library.records import Episode, ShotStatus
from tokimeki.library.shots import status_counts
from tokimeki.stages import cast, content_filter, lines, motion, scenes, shots, voice
from tokimeki.stages.base import Context


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[Context, Sequence[Episode]], None]
    reset: Callable[[Context, Episode], None]


STAGES = (
    Stage(shots.NAME, shots.run, shots.reset),
    Stage(content_filter.NAME, content_filter.run, content_filter.reset),
    Stage(cast.NAME, cast.run, cast.reset),
    Stage(lines.NAME, lines.run, lines.reset),
    Stage(scenes.NAME, scenes.run, scenes.reset),
    Stage(voice.NAME, voice.run, voice.reset),
    Stage(motion.NAME, motion.run, motion.reset),
)
STAGE_NAMES = tuple(stage.name for stage in STAGES)


def _position(name: str) -> int:
    if name not in STAGE_NAMES:
        raise ValueError(f"unknown stage {name!r}; stages are {', '.join(STAGE_NAMES)}")
    return STAGE_NAMES.index(name)


def redo(ctx: Context, episodes: Sequence[Episode], stage: str) -> None:
    """Forget `stage` and every later stage for `episodes`, so the next run repeats them."""
    for later in reversed(STAGES[_position(stage) :]):
        for episode in episodes:
            later.reset(ctx, episode)


@dataclass(frozen=True)
class EpisodeStatus:
    episode: Episode
    finished: list[str]
    shots: dict[ShotStatus, int]


def status(ctx: Context) -> list[EpisodeStatus]:
    return [
        EpisodeStatus(
            episode,
            [name for name in STAGE_NAMES if stage_done(ctx.conn, episode.id, name)],
            status_counts(ctx.conn, episode.id),
        )
        for episode in list_episodes(ctx.conn)
    ]


def run(ctx: Context, episodes: Sequence[Episode], until: str = STAGE_NAMES[-1]) -> None:
    """Run every stage up to `until`, skipping the work each stage has already finished."""
    for stage in STAGES[: _position(until) + 1]:
        stage.run(ctx, episodes)
