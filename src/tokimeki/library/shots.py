import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tokimeki.library.records import Frame, Rating, Shot, ShotStatus, Tag, TagCategory

_SHOT_COLUMNS = "id, episode_id, idx, start_frame, end_frame, status"
_FRAME_COLUMNS = "id, shot_id, frame_index, general, sensitive, questionable, explicit"
_F_FRAME_COLUMNS = ", ".join(f"f.{column}" for column in _FRAME_COLUMNS.split(", "))


@dataclass(frozen=True, slots=True)
class RatedFrame:
    frame_index: int
    rating: Rating
    tags: Sequence[Tag]


def _shot(row: tuple[int, int, int, int, int, str]) -> Shot:
    id_, episode_id, idx, start, end, status = row
    return Shot(id_, episode_id, idx, start, end, ShotStatus(status))


def _frame(row: tuple[int, int, int, float, float, float, float]) -> Frame:
    id_, shot_id, frame_index, general, sensitive, questionable, explicit = row
    return Frame(id_, shot_id, frame_index, Rating(general, sensitive, questionable, explicit))


def replace_shots(
    conn: sqlite3.Connection, episode_id: int, spans: Iterable[tuple[int, int]]
) -> list[Shot]:
    """Replace an episode's shots with `[start, end)` frame spans, all pending."""
    conn.execute("DELETE FROM shots WHERE episode_id = ?", (episode_id,))
    conn.executemany(
        "INSERT INTO shots (episode_id, idx, start_frame, end_frame) VALUES (?, ?, ?, ?)",
        [(episode_id, idx, start, end) for idx, (start, end) in enumerate(spans)],
    )
    return list_shots(conn, episode_id)


def list_shots(
    conn: sqlite3.Connection, episode_id: int, status: ShotStatus | None = None
) -> list[Shot]:
    if status is None:
        rows = conn.execute(
            f"SELECT {_SHOT_COLUMNS} FROM shots WHERE episode_id = ? ORDER BY idx", (episode_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_SHOT_COLUMNS} FROM shots WHERE episode_id = ? AND status = ? ORDER BY idx",
            (episode_id, status.value),
        ).fetchall()
    return [_shot(row) for row in rows]


def get_shot(conn: sqlite3.Connection, shot_id: int) -> Shot:
    row = conn.execute(f"SELECT {_SHOT_COLUMNS} FROM shots WHERE id = ?", (shot_id,)).fetchone()
    if row is None:
        raise KeyError(f"no shot {shot_id}")
    return _shot(row)


def drop_shot(conn: sqlite3.Connection, shot_id: int) -> None:
    conn.execute("UPDATE shots SET status = ? WHERE id = ?", (ShotStatus.DROPPED.value, shot_id))


def keep_shot(conn: sqlite3.Connection, shot_id: int, frames: Sequence[RatedFrame]) -> None:
    conn.execute("UPDATE shots SET status = ? WHERE id = ?", (ShotStatus.KEPT.value, shot_id))
    for frame in frames:
        r = frame.rating
        cur = conn.execute(
            "INSERT INTO frames (shot_id, frame_index, general, sensitive, questionable, explicit)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (shot_id, frame.frame_index, r.general, r.sensitive, r.questionable, r.explicit),
        )
        conn.executemany(
            "INSERT INTO frame_tags (frame_id, tag, category, score) VALUES (?, ?, ?, ?)",
            [(cur.lastrowid, t.name, t.category.value, t.score) for t in frame.tags],
        )


def reset_filter(conn: sqlite3.Connection, episode_id: int) -> None:
    """Forget the content filter's verdicts for an episode: every shot back to pending."""
    conn.execute(
        "DELETE FROM frames WHERE shot_id IN (SELECT id FROM shots WHERE episode_id = ?)",
        (episode_id,),
    )
    conn.execute(
        "UPDATE shots SET status = ? WHERE episode_id = ?", (ShotStatus.PENDING.value, episode_id)
    )


def list_frames(conn: sqlite3.Connection, shot_id: int) -> list[Frame]:
    rows = conn.execute(
        f"SELECT {_FRAME_COLUMNS} FROM frames WHERE shot_id = ? ORDER BY frame_index", (shot_id,)
    ).fetchall()
    return [_frame(row) for row in rows]


def list_episode_frames(conn: sqlite3.Connection, episode_id: int) -> list[Frame]:
    rows = conn.execute(
        f"SELECT {_F_FRAME_COLUMNS}"
        " FROM frames AS f JOIN shots AS s ON s.id = f.shot_id"
        " WHERE s.episode_id = ? ORDER BY f.frame_index",
        (episode_id,),
    ).fetchall()
    return [_frame(row) for row in rows]


def frame_tags(conn: sqlite3.Connection, frame_id: int) -> list[Tag]:
    rows: list[tuple[str, str, float]] = conn.execute(
        "SELECT tag, category, score FROM frame_tags WHERE frame_id = ? ORDER BY score DESC",
        (frame_id,),
    ).fetchall()
    return [Tag(name, TagCategory(category), score) for name, category, score in rows]


def shot_tags(conn: sqlite3.Connection, shot_id: int, limit: int = 5) -> list[tuple[str, float]]:
    """The shot's strongest WD14 general tags, scored by their mean over its sampled frames."""
    rows: list[tuple[str, float]] = conn.execute(
        "SELECT t.tag, SUM(t.score) / (SELECT COUNT(*) FROM frames WHERE shot_id = ?)"
        " FROM frame_tags AS t JOIN frames AS f ON f.id = t.frame_id"
        " WHERE f.shot_id = ? AND t.category = 'general'"
        " GROUP BY t.tag ORDER BY SUM(t.score) DESC, t.tag LIMIT ?",
        (shot_id, shot_id, limit),
    ).fetchall()
    return rows


def status_counts(conn: sqlite3.Connection, episode_id: int | None = None) -> dict[ShotStatus, int]:
    if episode_id is None:
        rows: list[tuple[str, int]] = conn.execute(
            "SELECT status, COUNT(*) FROM shots GROUP BY status"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM shots WHERE episode_id = ? GROUP BY status",
            (episode_id,),
        ).fetchall()
    counts = dict.fromkeys(ShotStatus, 0)
    for status, count in rows:
        counts[ShotStatus(status)] = count
    return counts
