import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tokimeki.library.db import open_library
from tokimeki.paths import SeriesPaths


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
