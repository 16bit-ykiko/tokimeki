"""Queries over the stored WD14 tags, and which of them describe a facial expression."""

import sqlite3
from collections.abc import Sequence

from tokimeki.library.records import TagStat

EXPRESSION_TAGS = (
    "smile",
    "blush",
    "light_blush",
    "full-face_blush",
    "nose_blush",
    ":d",
    "grin",
    "laughing",
    "^_^",
    "one_eye_closed",
    "pout",
    ":t",
    ">_<",
    ":o",
    "surprised",
    "embarrassed",
    "wavy_mouth",
    "flying_sweatdrops",
    "sweatdrop",
    "tears",
    "crying",
    "closed_eyes",
    "half-closed_eyes",
    "smug",
    ":3",
    ":p",
    "tongue_out",
    "heart",
    "spoken_heart",
    "frown",
    "angry",
    "sad",
    "scared",
    "nervous",
)
"""WD14 general tags that describe a face; `one_eye_closed` is Danbooru's wink."""


def _marks(count: int) -> str:
    return ", ".join("?" * count)


def tag_stats(
    conn: sqlite3.Connection, shot_ids: Sequence[int], tags: Sequence[str] = EXPRESSION_TAGS
) -> list[TagStat]:
    """`tags` over all sampled frames of the shots, strongest first; absent tags are left out."""
    if not shot_ids:
        return []
    total_row: tuple[int] = conn.execute(
        f"SELECT COUNT(*) FROM frames WHERE shot_id IN ({_marks(len(shot_ids))})", tuple(shot_ids)
    ).fetchone()
    rows: list[tuple[str, float, int]] = conn.execute(
        "SELECT t.tag, MAX(t.score), COUNT(*) FROM frame_tags AS t"
        " JOIN frames AS f ON f.id = t.frame_id"
        f" WHERE f.shot_id IN ({_marks(len(shot_ids))}) AND t.tag IN ({_marks(len(tags))})"
        " GROUP BY t.tag ORDER BY MAX(t.score) DESC, t.tag",
        (*shot_ids, *tags),
    ).fetchall()
    total = max(total_row[0], 1)
    return [TagStat(tag, peak, count / total) for tag, peak, count in rows]


def frame_tag_scores(
    conn: sqlite3.Connection, shot_id: int, tags: Sequence[str] = EXPRESSION_TAGS
) -> dict[int, dict[str, float]]:
    """For each sampled frame of a shot (by frame index), the scores of `tags` on it."""
    out: dict[int, dict[str, float]] = {
        index: {}
        for (index,) in conn.execute(
            "SELECT frame_index FROM frames WHERE shot_id = ? ORDER BY frame_index", (shot_id,)
        ).fetchall()
    }
    rows: list[tuple[int, str, float]] = conn.execute(
        "SELECT f.frame_index, t.tag, t.score FROM frame_tags AS t"
        " JOIN frames AS f ON f.id = t.frame_id"
        f" WHERE f.shot_id = ? AND t.tag IN ({_marks(len(tags))})",
        (shot_id, *tags),
    ).fetchall()
    for index, tag, score in rows:
        out[index][tag] = score
    return out
