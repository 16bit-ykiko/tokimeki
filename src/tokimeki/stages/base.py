import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tokimeki.library.db import open_library
from tokimeki.library.episodes import find_episode, register_episode
from tokimeki.library.records import Episode
from tokimeki.media.probe import probe
from tokimeki.paths import SeriesPaths

log = logging.getLogger("tokimeki")


@dataclass(frozen=True)
class Context:
    paths: SeriesPaths
    conn: sqlite3.Connection


def open_series(root: Path) -> Context:
    if not root.is_dir():
        raise FileNotFoundError(f"series directory {root} does not exist")
    paths = SeriesPaths(root.resolve())
    paths.ensure()
    return Context(paths, open_library(paths.db))


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
