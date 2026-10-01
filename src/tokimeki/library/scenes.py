import sqlite3
from collections.abc import Sequence

from tokimeki.library.records import Scene


def replace_scenes(
    conn: sqlite3.Connection, episode_id: int, groups: Sequence[Sequence[int]]
) -> list[Scene]:
    """Replace an episode's scenes; each group is the shot ids of one scene, in order."""
    conn.execute("DELETE FROM scenes WHERE episode_id = ?", (episode_id,))
    for index, shot_ids in enumerate(groups):
        cur = conn.execute(
            "INSERT INTO scenes (episode_id, idx) VALUES (?, ?)", (episode_id, index)
        )
        conn.executemany(
            "INSERT INTO scene_shots (scene_id, shot_id) VALUES (?, ?)",
            [(cur.lastrowid, shot_id) for shot_id in shot_ids],
        )
    return list_scenes(conn, episode_id)


def list_scenes(conn: sqlite3.Connection, episode_id: int) -> list[Scene]:
    rows: list[tuple[int, int, int]] = conn.execute(
        "SELECT sc.id, sc.idx, ss.shot_id FROM scenes AS sc"
        " JOIN scene_shots AS ss ON ss.scene_id = sc.id"
        " JOIN shots AS s ON s.id = ss.shot_id"
        " WHERE sc.episode_id = ? ORDER BY sc.idx, s.idx",
        (episode_id,),
    ).fetchall()
    grouped: dict[int, tuple[int, list[int]]] = {}
    for scene_id, index, shot_id in rows:
        grouped.setdefault(scene_id, (index, []))[1].append(shot_id)
    return [
        Scene(scene_id, episode_id, index, tuple(shots))
        for scene_id, (index, shots) in grouped.items()
    ]
