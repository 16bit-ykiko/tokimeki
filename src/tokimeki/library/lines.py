import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from tokimeki.library.records import Line, LineKind, Part, PartKind

_LINE_COLUMNS = "l.id, l.episode_id, l.start, l.end, l.kind, l.style, l.text"


@dataclass(frozen=True, slots=True)
class NewLine:
    start: float
    end: float
    kind: LineKind
    style: str
    text: str


def _line(row: tuple[int, int, float, float, str, str, str]) -> Line:
    id_, episode_id, start, end, kind, style, text = row
    return Line(id_, episode_id, start, end, LineKind(kind), style, text)


def replace_lines(
    conn: sqlite3.Connection, episode_id: int, lines: Iterable[NewLine], parts: Iterable[Part]
) -> None:
    conn.execute("DELETE FROM lines WHERE episode_id = ?", (episode_id,))
    conn.execute("DELETE FROM episode_parts WHERE episode_id = ?", (episode_id,))
    conn.executemany(
        "INSERT INTO lines (episode_id, start, end, kind, style, text) VALUES (?, ?, ?, ?, ?, ?)",
        [(episode_id, x.start, x.end, x.kind.value, x.style, x.text) for x in lines],
    )
    conn.executemany(
        "INSERT INTO episode_parts (episode_id, kind, start, end) VALUES (?, ?, ?, ?)",
        [(episode_id, p.kind.value, p.start, p.end) for p in parts],
    )


def list_lines(
    conn: sqlite3.Connection, episode_id: int, kind: LineKind | None = None
) -> list[Line]:
    query = f"SELECT {_LINE_COLUMNS} FROM lines AS l WHERE l.episode_id = ?"
    args: tuple[int | str, ...] = (episode_id,)
    if kind is not None:
        query += " AND l.kind = ?"
        args += (kind.value,)
    rows = conn.execute(query + " ORDER BY l.start, l.id", args).fetchall()
    return [_line(row) for row in rows]


def lines_by_shot(
    conn: sqlite3.Connection, episode_id: int, kind: LineKind = LineKind.DIALOGUE
) -> dict[int, list[Line]]:
    """Lines overlapping each shot of an episode, keyed by shot id."""
    rows: list[tuple[int, int, int, float, float, str, str, str]] = conn.execute(
        f"SELECT sl.shot_id, {_LINE_COLUMNS} FROM shot_lines AS sl"
        " JOIN lines AS l ON l.id = sl.line_id"
        " WHERE l.episode_id = ? AND l.kind = ? ORDER BY sl.shot_id, l.start, l.id",
        (episode_id, kind.value),
    ).fetchall()
    out: dict[int, list[Line]] = {}
    for row in rows:
        out.setdefault(row[0], []).append(_line(row[1:]))
    return out


def list_parts(conn: sqlite3.Connection, episode_id: int) -> list[Part]:
    rows: list[tuple[int, str, float, float]] = conn.execute(
        "SELECT episode_id, kind, start, end FROM episode_parts"
        " WHERE episode_id = ? ORDER BY start",
        (episode_id,),
    ).fetchall()
    return [Part(e, PartKind(k), s, t) for e, k, s, t in rows]


def shots_in_parts(conn: sqlite3.Connection, episode_id: int) -> dict[int, PartKind]:
    """Shots overlapping the opening or ending, keyed by shot id."""
    rows: list[tuple[int, str]] = conn.execute(
        "SELECT s.id, p.kind FROM shots AS s"
        " JOIN episodes AS e ON e.id = s.episode_id"
        " JOIN episode_parts AS p ON p.episode_id = s.episode_id"
        " WHERE s.episode_id = ?"
        " AND p.start < CAST(s.end_frame AS REAL) * e.fps_den / e.fps_num"
        " AND p.end > CAST(s.start_frame AS REAL) * e.fps_den / e.fps_num",
        (episode_id,),
    ).fetchall()
    return {shot_id: PartKind(kind) for shot_id, kind in rows}
