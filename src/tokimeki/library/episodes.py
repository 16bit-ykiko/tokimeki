import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction

from tokimeki.library.records import Episode

type Param = str | int | float | bool

_EPISODE_COLUMNS = "id, path, width, height, fps_num, fps_den, frame_count"


def _episode(row: tuple[int, str, int, int, int, int, int]) -> Episode:
    id_, path, width, height, fps_num, fps_den, frame_count = row
    return Episode(id_, path, width, height, Fraction(fps_num, fps_den), frame_count)


def register_episode(
    conn: sqlite3.Connection, path: str, width: int, height: int, fps: Fraction, frame_count: int
) -> Episode:
    """Insert an episode, or return the existing one with the same path untouched."""
    found = find_episode(conn, path)
    if found is not None:
        return found
    cur = conn.execute(
        "INSERT INTO episodes (path, width, height, fps_num, fps_den, frame_count)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (path, width, height, fps.numerator, fps.denominator, frame_count),
    )
    assert cur.lastrowid is not None
    return get_episode(conn, cur.lastrowid)


def get_episode(conn: sqlite3.Connection, episode_id: int) -> Episode:
    row = conn.execute(
        f"SELECT {_EPISODE_COLUMNS} FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no episode {episode_id}")
    return _episode(row)


def find_episode(conn: sqlite3.Connection, path: str) -> Episode | None:
    row = conn.execute(
        f"SELECT {_EPISODE_COLUMNS} FROM episodes WHERE path = ?", (path,)
    ).fetchone()
    return None if row is None else _episode(row)


def list_episodes(conn: sqlite3.Connection) -> list[Episode]:
    rows = conn.execute(f"SELECT {_EPISODE_COLUMNS} FROM episodes ORDER BY path").fetchall()
    return [_episode(row) for row in rows]


def set_frame_count(conn: sqlite3.Connection, episode_id: int, frame_count: int) -> None:
    conn.execute("UPDATE episodes SET frame_count = ? WHERE id = ?", (frame_count, episode_id))


def stage_done(conn: sqlite3.Connection, episode_id: int, stage: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM stage_runs WHERE episode_id = ? AND stage = ?", (episode_id, stage)
    ).fetchone()
    return row is not None


def mark_stage_done(
    conn: sqlite3.Connection, episode_id: int, stage: str, params: Mapping[str, Param]
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO stage_runs (episode_id, stage, finished_at, params)"
        " VALUES (?, ?, ?, ?)",
        (episode_id, stage, datetime.now(UTC).isoformat(), json.dumps(dict(params))),
    )


def clear_stage(conn: sqlite3.Connection, episode_id: int, stage: str) -> None:
    conn.execute("DELETE FROM stage_runs WHERE episode_id = ? AND stage = ?", (episode_id, stage))


def stage_params(conn: sqlite3.Connection, episode_id: int, stage: str) -> dict[str, Param] | None:
    row: tuple[str] | None = conn.execute(
        "SELECT params FROM stage_runs WHERE episode_id = ? AND stage = ?", (episode_id, stage)
    ).fetchone()
    if row is None:
        return None
    loaded: dict[str, Param] = json.loads(row[0])
    return loaded
