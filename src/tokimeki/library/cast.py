import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.records import Box, Cluster, Face, ShotCast

type Embeddings = NDArray[np.float32]

_FACE_COLUMNS = "fa.id, fa.frame_id, fa.x0, fa.y0, fa.x1, fa.y1, fa.score, fa.cluster_id"


@dataclass(frozen=True, slots=True)
class NewFace:
    frame_id: int
    box: Box
    score: float
    embedding: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FaceInShot:
    face: Face
    episode_id: int
    shot_id: int
    frame_index: int


type _FaceInShotRow = tuple[int, int, float, float, float, float, float, int | None, int, int, int]
_FACE_IN_SHOT = (
    f"SELECT {_FACE_COLUMNS}, s.episode_id, f.shot_id, f.frame_index"
    " FROM faces AS fa JOIN frames AS f ON f.id = fa.frame_id JOIN shots AS s ON s.id = f.shot_id"
)


def _face_in_shot(row: _FaceInShotRow) -> FaceInShot:
    return FaceInShot(_face(row[:8]), row[8], row[9], row[10])


def _face(row: tuple[int, int, float, float, float, float, float, int | None]) -> Face:
    id_, frame_id, x0, y0, x1, y1, score, cluster_id = row
    return Face(id_, frame_id, Box(x0, y0, x1, y1), score, cluster_id)


def add_faces(conn: sqlite3.Connection, faces: Iterable[NewFace]) -> list[int]:
    ids: list[int] = []
    for face in faces:
        b = face.box
        cur = conn.execute(
            "INSERT INTO faces (frame_id, x0, y0, x1, y1, score, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                face.frame_id,
                b.x0,
                b.y0,
                b.x1,
                b.y1,
                face.score,
                np.ascontiguousarray(face.embedding, dtype=np.float32).tobytes(),
            ),
        )
        assert cur.lastrowid is not None
        ids.append(cur.lastrowid)
    return ids


def clear_episode_faces(conn: sqlite3.Connection, episode_id: int) -> None:
    conn.execute(
        "DELETE FROM faces WHERE frame_id IN"
        " (SELECT f.id FROM frames AS f JOIN shots AS s ON s.id = f.shot_id"
        "  WHERE s.episode_id = ?)",
        (episode_id,),
    )
    delete_empty_clusters(conn)


def episode_faces(conn: sqlite3.Connection, episode_id: int) -> list[FaceInShot]:
    rows: list[_FaceInShotRow] = conn.execute(
        f"{_FACE_IN_SHOT} WHERE s.episode_id = ? ORDER BY fa.id", (episode_id,)
    ).fetchall()
    return [_face_in_shot(row) for row in rows]


def cluster_faces(conn: sqlite3.Connection, cluster_id: int) -> list[FaceInShot]:
    rows: list[_FaceInShotRow] = conn.execute(
        f"{_FACE_IN_SHOT} WHERE fa.cluster_id = ? ORDER BY fa.id", (cluster_id,)
    ).fetchall()
    return [_face_in_shot(row) for row in rows]


def face_embeddings(conn: sqlite3.Connection, face_ids: Sequence[int]) -> Embeddings:
    """Embeddings of the given faces, one row each, in the order asked for."""
    blobs: dict[int, bytes] = {}
    for start in range(0, len(face_ids), 500):
        chunk = face_ids[start : start + 500]
        rows: list[tuple[int, bytes]] = conn.execute(
            f"SELECT id, embedding FROM faces WHERE id IN ({', '.join('?' * len(chunk))})",
            chunk,
        ).fetchall()
        blobs.update(rows)
    vectors = [np.frombuffer(blobs[face_id], dtype=np.float32) for face_id in face_ids]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vectors)


def clustered_face_ids(conn: sqlite3.Connection) -> dict[int, list[int]]:
    """Face ids of every cluster, keyed by cluster id."""
    rows: list[tuple[int, int]] = conn.execute(
        "SELECT cluster_id, id FROM faces WHERE cluster_id IS NOT NULL ORDER BY cluster_id, id"
    ).fetchall()
    out: dict[int, list[int]] = {}
    for cluster_id, face_id in rows:
        out.setdefault(cluster_id, []).append(face_id)
    return out


def create_cluster(conn: sqlite3.Connection, name: str | None = None) -> int:
    cur = conn.execute("INSERT INTO clusters (name) VALUES (?)", (name,))
    assert cur.lastrowid is not None
    return cur.lastrowid


def assign_faces(conn: sqlite3.Connection, face_ids: Iterable[int], cluster_id: int | None) -> None:
    conn.executemany(
        "UPDATE faces SET cluster_id = ? WHERE id = ?",
        [(cluster_id, face_id) for face_id in face_ids],
    )


def list_clusters(conn: sqlite3.Connection) -> list[Cluster]:
    rows: list[tuple[int, str | None, int]] = conn.execute(
        "SELECT c.id, c.name, COUNT(fa.id) FROM clusters AS c"
        " LEFT JOIN faces AS fa ON fa.cluster_id = c.id"
        " GROUP BY c.id ORDER BY COUNT(fa.id) DESC, c.id"
    ).fetchall()
    return [Cluster(id_, name, count) for id_, name, count in rows]


def get_cluster(conn: sqlite3.Connection, cluster_id: int) -> Cluster:
    for cluster in list_clusters(conn):
        if cluster.id == cluster_id:
            return cluster
    raise KeyError(f"no cluster {cluster_id}")


def name_cluster(conn: sqlite3.Connection, cluster_id: int, name: str | None) -> None:
    get_cluster(conn, cluster_id)
    conn.execute("UPDATE clusters SET name = ? WHERE id = ?", (name, cluster_id))


def merge_clusters(conn: sqlite3.Connection, source_id: int, target_id: int) -> None:
    """Move every face of `source_id` into `target_id`; the target keeps its name if it has one."""
    if source_id == target_id:
        raise ValueError("cannot merge a cluster into itself")
    source, target = get_cluster(conn, source_id), get_cluster(conn, target_id)
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (target_id, source_id))
    conn.execute("DELETE FROM clusters WHERE id = ?", (source_id,))
    if target.name is None and source.name is not None:
        conn.execute("UPDATE clusters SET name = ? WHERE id = ?", (source.name, target_id))


def unassign_unnamed(conn: sqlite3.Connection) -> None:
    """Take every face out of the clusters the user has not named, and drop those clusters."""
    conn.execute(
        "UPDATE faces SET cluster_id = NULL"
        " WHERE cluster_id IN (SELECT id FROM clusters WHERE name IS NULL)"
    )
    delete_empty_clusters(conn)


def delete_empty_clusters(conn: sqlite3.Connection) -> None:
    """Drop clusters with no faces left, unless the user named them."""
    conn.execute(
        "DELETE FROM clusters WHERE name IS NULL"
        " AND id NOT IN (SELECT cluster_id FROM faces WHERE cluster_id IS NOT NULL)"
    )


def episode_cast(conn: sqlite3.Connection, episode_id: int) -> list[ShotCast]:
    rows: list[tuple[int, int, float, float, float]] = conn.execute(
        "SELECT sc.shot_id, sc.cluster_id, sc.presence, sc.face_height, sc.face_area"
        " FROM shot_cast AS sc JOIN shots AS s ON s.id = sc.shot_id"
        " WHERE s.episode_id = ? ORDER BY s.idx, sc.presence DESC",
        (episode_id,),
    ).fetchall()
    return [ShotCast(*row) for row in rows]


def cluster_character_tags(
    conn: sqlite3.Connection, cluster_id: int, limit: int = 3
) -> list[tuple[str, float]]:
    """WD14 character tags on the cluster's single-face frames, with the share of them they're on.

    A frame-level tag cannot say which face it means, so frames with several faces are left out.
    """
    solo = """
        SELECT fa.frame_id FROM faces AS fa WHERE fa.cluster_id = ?
        AND (SELECT COUNT(*) FROM faces AS f2 WHERE f2.frame_id = fa.frame_id) = 1
    """
    total_row: tuple[int] = conn.execute(f"SELECT COUNT(*) FROM ({solo})", (cluster_id,)).fetchone()
    if total_row[0] == 0:
        return []
    rows: list[tuple[str, int]] = conn.execute(
        f"SELECT t.tag, COUNT(*) FROM frame_tags AS t WHERE t.category = 'character'"
        f" AND t.frame_id IN ({solo}) GROUP BY t.tag ORDER BY COUNT(*) DESC, t.tag LIMIT ?",
        (cluster_id, limit),
    ).fetchall()
    return [(tag, count / total_row[0]) for tag, count in rows]


def named_cast_by_shot(
    conn: sqlite3.Connection, episode_id: int
) -> dict[int, list[tuple[str, ShotCast]]]:
    """Per shot, the characters the user has named, most present first."""
    rows: list[tuple[str, int, int, float, float, float]] = conn.execute(
        "SELECT c.name, sc.shot_id, sc.cluster_id, sc.presence, sc.face_height, sc.face_area"
        " FROM shot_cast AS sc JOIN clusters AS c ON c.id = sc.cluster_id"
        " JOIN shots AS s ON s.id = sc.shot_id"
        " WHERE s.episode_id = ? AND c.name IS NOT NULL ORDER BY s.idx, sc.presence DESC",
        (episode_id,),
    ).fetchall()
    out: dict[int, list[tuple[str, ShotCast]]] = {}
    for name, shot_id, cluster_id, presence, height, area in rows:
        out.setdefault(shot_id, []).append(
            (name, ShotCast(shot_id, cluster_id, presence, height, area))
        )
    return out


def named_face_heights(
    conn: sqlite3.Connection, episode_id: int, name: str
) -> dict[int, dict[int, float]]:
    """Per shot, per sampled frame index: the height of the named character's largest face."""
    rows: list[tuple[int, int, float]] = conn.execute(
        "SELECT f.shot_id, f.frame_index, MAX(fa.y1 - fa.y0) FROM faces AS fa"
        " JOIN clusters AS c ON c.id = fa.cluster_id"
        " JOIN frames AS f ON f.id = fa.frame_id"
        " JOIN shots AS s ON s.id = f.shot_id"
        " WHERE s.episode_id = ? AND c.name = ? GROUP BY f.shot_id, f.frame_index",
        (episode_id, name),
    ).fetchall()
    out: dict[int, dict[int, float]] = {}
    for shot_id, frame_index, height in rows:
        out.setdefault(shot_id, {})[frame_index] = height
    return out
