"""The voice tables of the library: each line's voice, speakers, fixes, transcripts, styles.

Speakers mirror the face clusters of the cast: ids are never reused, named speakers are
kept through every re-clustering, and a fix pins a line to a named speaker (or to nobody).
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class VoiceStatus(StrEnum):
    CLEAN = "clean"
    OVERLAP = "overlap"
    """Another line is said at the same time."""
    SHORT = "short"
    """Too little voice to tell the speaker."""
    LONG = "long"
    SILENT = "silent"
    """Nothing in the voice stem: off-screen, or a dropped shot."""
    MIXED = "mixed"
    """Two voices one after the other within the line."""


@dataclass(frozen=True, slots=True)
class NewLineVoice:
    line_id: int
    start: float
    end: float
    status: VoiceStatus
    embedding: NDArray[np.float32] | None


@dataclass(frozen=True, slots=True)
class LineVoice:
    line_id: int
    episode_id: int
    start: float
    end: float
    status: VoiceStatus
    speaker_id: int | None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Speaker:
    id: int
    name: str | None
    lines: int
    seconds: float

    @property
    def label(self) -> str:
        return self.name or f"#{self.id}"


@dataclass(frozen=True, slots=True)
class SpeakerFix:
    line_id: int
    speaker_id: int | None


type _VoiceRow = tuple[int, int, float, float, str, int | None]
_VOICE = (
    "SELECT v.line_id, l.episode_id, v.start, v.end, v.status, v.speaker_id"
    " FROM line_voices AS v JOIN lines AS l ON l.id = v.line_id"
)


def _voice(row: _VoiceRow) -> LineVoice:
    line_id, episode_id, start, end, status, speaker_id = row
    return LineVoice(line_id, episode_id, start, end, VoiceStatus(status), speaker_id)


def replace_line_voices(
    conn: sqlite3.Connection, episode_id: int, voices: Iterable[NewLineVoice]
) -> None:
    """Replace the voices of an episode's lines; speakers are assigned afterwards."""
    clear_line_voices(conn, episode_id)
    conn.executemany(
        "INSERT INTO line_voices (line_id, start, end, status, embedding) VALUES (?, ?, ?, ?, ?)",
        [
            (
                v.line_id,
                v.start,
                v.end,
                v.status.value,
                None
                if v.embedding is None
                else np.ascontiguousarray(v.embedding, dtype=np.float32).tobytes(),
            )
            for v in voices
        ],
    )


def clear_line_voices(conn: sqlite3.Connection, episode_id: int) -> None:
    conn.execute(
        "DELETE FROM line_voices WHERE line_id IN (SELECT id FROM lines WHERE episode_id = ?)",
        (episode_id,),
    )
    delete_empty_speakers(conn)


def episode_voices(conn: sqlite3.Connection, episode_id: int) -> list[LineVoice]:
    rows: list[_VoiceRow] = conn.execute(
        f"{_VOICE} WHERE l.episode_id = ? ORDER BY l.start, l.id", (episode_id,)
    ).fetchall()
    return [_voice(row) for row in rows]


def get_voice(conn: sqlite3.Connection, line_id: int) -> LineVoice:
    row: _VoiceRow | None = conn.execute(f"{_VOICE} WHERE v.line_id = ?", (line_id,)).fetchone()
    if row is None:
        raise KeyError(f"line {line_id} has no voice (not a clean dialogue line, or not cut yet)")
    return _voice(row)


def clean_voices(conn: sqlite3.Connection, speaker_id: int | None = None) -> list[LineVoice]:
    """Every clean voice, or those of one speaker, in episode and time order."""
    query = f"{_VOICE} WHERE v.status = 'clean'"
    args: tuple[int, ...] = ()
    if speaker_id is not None:
        query += " AND v.speaker_id = ?"
        args = (speaker_id,)
    rows: list[_VoiceRow] = conn.execute(query + " ORDER BY l.episode_id, l.start", args).fetchall()
    return [_voice(row) for row in rows]


def voice_embeddings(conn: sqlite3.Connection, line_ids: Sequence[int]) -> NDArray[np.float32]:
    """Embeddings of the given lines' voices, one row each, in the order asked for."""
    blobs: dict[int, bytes] = {}
    for start in range(0, len(line_ids), 500):
        chunk = line_ids[start : start + 500]
        rows: list[tuple[int, bytes]] = conn.execute(
            "SELECT line_id, embedding FROM line_voices"
            f" WHERE embedding IS NOT NULL AND line_id IN ({', '.join('?' * len(chunk))})",
            chunk,
        ).fetchall()
        blobs.update(rows)
    vectors = [np.frombuffer(blobs[line_id], dtype=np.float32) for line_id in line_ids]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vectors)


def create_speaker(conn: sqlite3.Connection, name: str | None = None) -> int:
    cur = conn.execute("INSERT INTO speakers (name) VALUES (?)", (name,))
    assert cur.lastrowid is not None
    return cur.lastrowid


def assign_speaker(
    conn: sqlite3.Connection, line_ids: Iterable[int], speaker_id: int | None
) -> None:
    conn.executemany(
        "UPDATE line_voices SET speaker_id = ? WHERE line_id = ?",
        [(speaker_id, line_id) for line_id in line_ids],
    )


def list_speakers(conn: sqlite3.Connection) -> list[Speaker]:
    """Every speaker with its clean lines and their seconds, the most heard first."""
    rows: list[tuple[int, str | None, int, float | None]] = conn.execute(
        "SELECT s.id, s.name, COUNT(v.line_id), SUM(v.end - v.start) FROM speakers AS s"
        " LEFT JOIN line_voices AS v ON v.speaker_id = s.id"
        " GROUP BY s.id ORDER BY SUM(v.end - v.start) DESC, s.id"
    ).fetchall()
    return [Speaker(id_, name, n, seconds or 0.0) for id_, name, n, seconds in rows]


def get_speaker(conn: sqlite3.Connection, speaker_id: int) -> Speaker:
    for speaker in list_speakers(conn):
        if speaker.id == speaker_id:
            return speaker
    raise KeyError(f"no speaker {speaker_id}")


def find_speaker(conn: sqlite3.Connection, ref: str) -> Speaker:
    """A speaker by name, or by id (`3` or `#3`)."""
    speakers = list_speakers(conn)
    for speaker in speakers:
        if speaker.name == ref:
            return speaker
    number = ref.removeprefix("#")
    for speaker in speakers:
        if number.isdigit() and speaker.id == int(number):
            return speaker
    raise KeyError(f"no speaker {ref!r}")


def name_speaker(conn: sqlite3.Connection, speaker_id: int, name: str | None) -> None:
    get_speaker(conn, speaker_id)
    taken = conn.execute(
        "SELECT id FROM speakers WHERE name = ? AND id != ?", (name, speaker_id)
    ).fetchone()
    if name is not None and taken is not None:
        raise ValueError(f"speaker #{taken[0]} is already {name}; merge the two instead")
    conn.execute("UPDATE speakers SET name = ? WHERE id = ?", (name, speaker_id))


def merge_speakers(conn: sqlite3.Connection, source_id: int, target_id: int) -> None:
    """Move every line of `source_id` to `target_id`; the target keeps its name if it has one."""
    if source_id == target_id:
        raise ValueError("cannot merge a speaker into itself")
    source, target = get_speaker(conn, source_id), get_speaker(conn, target_id)
    conn.execute(
        "UPDATE line_voices SET speaker_id = ? WHERE speaker_id = ?", (target_id, source_id)
    )
    conn.execute(
        "UPDATE speaker_fixes SET speaker_id = ? WHERE speaker_id = ?", (target_id, source_id)
    )
    conn.execute(
        "UPDATE OR IGNORE voice_styles SET speaker_id = ? WHERE speaker_id = ?",
        (target_id, source_id),
    )
    conn.execute("DELETE FROM speakers WHERE id = ?", (source_id,))
    if target.name is None and source.name is not None:
        conn.execute("UPDATE speakers SET name = ? WHERE id = ?", (source.name, target_id))


def fix_line_speaker(conn: sqlite3.Connection, line_id: int, speaker_id: int | None) -> None:
    """Record that the line is said by `speaker_id` (None: by nobody to keep) and apply it."""
    conn.execute(
        "INSERT INTO speaker_fixes (line_id, speaker_id) VALUES (?, ?)"
        " ON CONFLICT (line_id) DO UPDATE SET speaker_id = excluded.speaker_id",
        (line_id, speaker_id),
    )
    apply_speaker_fixes(conn)


def apply_speaker_fixes(conn: sqlite3.Connection) -> None:
    """Give every fixed clean line its fixed speaker; run after anything that assigns them."""
    conn.execute(
        "UPDATE line_voices SET speaker_id ="
        " (SELECT x.speaker_id FROM speaker_fixes AS x WHERE x.line_id = line_voices.line_id)"
        " WHERE status = 'clean' AND line_id IN (SELECT line_id FROM speaker_fixes)"
    )
    delete_empty_speakers(conn)


def list_speaker_fixes(conn: sqlite3.Connection) -> list[SpeakerFix]:
    rows: list[tuple[int, int | None]] = conn.execute(
        "SELECT line_id, speaker_id FROM speaker_fixes ORDER BY line_id"
    ).fetchall()
    return [SpeakerFix(*row) for row in rows]


def fixed_lines(conn: sqlite3.Connection) -> set[int]:
    rows: list[tuple[int]] = conn.execute("SELECT line_id FROM speaker_fixes").fetchall()
    return {row[0] for row in rows}


def unassign_unnamed_speakers(conn: sqlite3.Connection) -> None:
    """Take every line from the speakers the user has not named, and drop those speakers."""
    conn.execute(
        "UPDATE line_voices SET speaker_id = NULL"
        " WHERE speaker_id IN (SELECT id FROM speakers WHERE name IS NULL)"
    )
    delete_empty_speakers(conn)


def delete_empty_speakers(conn: sqlite3.Connection) -> None:
    """Drop speakers with no lines left, unless the user named them."""
    conn.execute(
        "DELETE FROM speakers WHERE name IS NULL"
        " AND id NOT IN (SELECT speaker_id FROM line_voices WHERE speaker_id IS NOT NULL)"
    )


def set_transcripts(conn: sqlite3.Connection, model: str, texts: Iterable[tuple[int, str]]) -> None:
    conn.executemany(
        "INSERT INTO transcripts (line_id, model, text) VALUES (?, ?, ?)"
        " ON CONFLICT (line_id) DO UPDATE SET model = excluded.model, text = excluded.text",
        [(line_id, model, text) for line_id, text in texts],
    )


def transcripts(conn: sqlite3.Connection, episode_id: int | None = None) -> dict[int, str]:
    """Japanese text by line id, of one episode or all."""
    query = "SELECT t.line_id, t.text FROM transcripts AS t JOIN lines AS l ON l.id = t.line_id"
    args: tuple[int, ...] = ()
    if episode_id is not None:
        query += " WHERE l.episode_id = ?"
        args = (episode_id,)
    rows: list[tuple[int, str]] = conn.execute(query, args).fetchall()
    return dict(rows)


def clear_transcripts(conn: sqlite3.Connection, episode_id: int) -> None:
    conn.execute(
        "DELETE FROM transcripts WHERE line_id IN (SELECT id FROM lines WHERE episode_id = ?)",
        (episode_id,),
    )


def set_style(conn: sqlite3.Connection, speaker_id: int, style: str, line_id: int) -> None:
    conn.execute(
        "INSERT INTO voice_styles (speaker_id, style, line_id) VALUES (?, ?, ?)"
        " ON CONFLICT (speaker_id, style) DO UPDATE SET line_id = excluded.line_id",
        (speaker_id, style, line_id),
    )


def styles(conn: sqlite3.Connection, speaker_id: int) -> dict[str, int]:
    """Reference line id by style name."""
    rows: list[tuple[str, int]] = conn.execute(
        "SELECT style, line_id FROM voice_styles WHERE speaker_id = ? ORDER BY style",
        (speaker_id,),
    ).fetchall()
    return dict(rows)
