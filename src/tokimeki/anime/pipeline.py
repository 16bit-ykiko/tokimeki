import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tokimeki.anime import content_filter, faces, lines, motion, scenes, shots, stems
from tokimeki.library.episodes import find_episode, list_episodes, register_episode, stage_done
from tokimeki.library.records import Episode, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import status_counts
from tokimeki.media.probe import probe

log = logging.getLogger("tokimeki")


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[Context, Sequence[Episode]], None]
    reset: Callable[[Context, Episode], None]


STAGES = (
    Stage(shots.NAME, shots.run, shots.reset),
    Stage(content_filter.NAME, content_filter.run, content_filter.reset),
    Stage(faces.NAME, faces.run, faces.reset),
    Stage(lines.NAME, lines.run, lines.reset),
    Stage(scenes.NAME, scenes.run, scenes.reset),
    Stage(stems.NAME, stems.run, stems.reset),
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


def register_episodes(ctx: Context) -> list[Episode]:
    """Every video file of the series, registered in the library (probed on first sight)."""
    episodes: list[Episode] = []
    for relative in ctx.paths.find_episodes():
        found = find_episode(ctx.conn, relative)
        if found is None:
            info = probe(ctx.paths.episode_file(relative))
            found = register_episode(
                ctx.conn,
                relative,
                info.width,
                info.height,
                info.fps,
                info.estimated_frames,
            )
            log.info("registered %s (%dx%d, %.3f fps)", relative, info.width, info.height, info.fps)
        episodes.append(found)
    return episodes
