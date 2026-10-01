import sqlite3
from fractions import Fraction

import numpy as np
import pytest

from tokimeki.library import cast as cast_db
from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.db import migrate, open_library, schema_version, transaction
from tokimeki.library.records import Box, Episode, Rating, ShotStatus, Tag, TagCategory
from tokimeki.library.schema import MIGRATIONS

SAFE = Rating(0.9, 0.1, 0.01, 0.0)


@pytest.fixture
def conn() -> sqlite3.Connection:
    return open_library(":memory:")


@pytest.fixture
def episode(conn: sqlite3.Connection) -> Episode:
    return episode_db.register_episode(conn, "ep01.mkv", 1920, 1080, Fraction(24000, 1001), 1000)


def test_migrations_apply_once(conn: sqlite3.Connection) -> None:
    assert schema_version(conn) == len(MIGRATIONS)
    migrate(conn)
    assert schema_version(conn) == len(MIGRATIONS)


def test_register_is_idempotent(conn: sqlite3.Connection, episode: Episode) -> None:
    again = episode_db.register_episode(conn, "ep01.mkv", 1, 1, Fraction(1), 1)
    assert again == episode
    assert episode.seconds(24000) == pytest.approx(1001.0)


def test_stage_marks(conn: sqlite3.Connection, episode: Episode) -> None:
    assert not episode_db.stage_done(conn, episode.id, "shots")
    episode_db.mark_stage_done(conn, episode.id, "shots", {"threshold": 0.5})
    assert episode_db.stage_done(conn, episode.id, "shots")
    assert episode_db.stage_params(conn, episode.id, "shots") == {"threshold": 0.5}
    episode_db.clear_stage(conn, episode.id, "shots")
    assert not episode_db.stage_done(conn, episode.id, "shots")


def test_keep_and_drop(conn: sqlite3.Connection, episode: Episode) -> None:
    a, b = shot_db.replace_shots(conn, episode.id, [(0, 10), (10, 30)])
    smile = Tag("smile", TagCategory.GENERAL, 0.8)
    shot_db.keep_shot(conn, a.id, [shot_db.RatedFrame(5, SAFE, [smile])])
    shot_db.drop_shot(conn, b.id)
    frames = shot_db.list_frames(conn, a.id)
    assert [f.frame_index for f in frames] == [5]
    assert shot_db.frame_tags(conn, frames[0].id) == [smile]
    counts = shot_db.status_counts(conn, episode.id)
    assert counts == {ShotStatus.PENDING: 0, ShotStatus.KEPT: 1, ShotStatus.DROPPED: 1}


def test_only_kept_shots_own_frames(conn: sqlite3.Connection, episode: Episode) -> None:
    (shot,) = shot_db.replace_shots(conn, episode.id, [(0, 10)])
    with pytest.raises(sqlite3.IntegrityError, match="only for kept shots"):
        conn.execute(
            "INSERT INTO frames (shot_id, frame_index, general, sensitive, questionable, explicit)"
            " VALUES (?, 1, 1, 0, 0, 0)",
            (shot.id,),
        )
    shot_db.keep_shot(conn, shot.id, [shot_db.RatedFrame(5, SAFE, [])])
    with pytest.raises(sqlite3.IntegrityError, match="before it leaves the kept state"):
        shot_db.drop_shot(conn, shot.id)


def test_reset_filter_returns_shots_to_pending(conn: sqlite3.Connection, episode: Episode) -> None:
    (shot,) = shot_db.replace_shots(conn, episode.id, [(0, 10)])
    shot_db.keep_shot(conn, shot.id, [shot_db.RatedFrame(5, SAFE, [])])
    shot_db.reset_filter(conn, episode.id)
    assert shot_db.get_shot(conn, shot.id).status is ShotStatus.PENDING
    assert shot_db.list_frames(conn, shot.id) == []


def test_transaction_rolls_back(conn: sqlite3.Connection, episode: Episode) -> None:
    with pytest.raises(RuntimeError), transaction(conn):
        shot_db.replace_shots(conn, episode.id, [(0, 10)])
        raise RuntimeError
    assert shot_db.list_shots(conn, episode.id) == []


def _faces(conn: sqlite3.Connection, episode: Episode) -> tuple[list[int], list[int]]:
    a, b = shot_db.replace_shots(conn, episode.id, [(0, 10), (10, 20)])
    shot_db.keep_shot(
        conn, a.id, [shot_db.RatedFrame(2, SAFE, []), shot_db.RatedFrame(6, SAFE, [])]
    )
    shot_db.keep_shot(conn, b.id, [shot_db.RatedFrame(15, SAFE, [])])
    frame_ids = [f.id for f in shot_db.list_episode_frames(conn, episode.id)]
    vector = np.ones(4, dtype=np.float32)
    faces = [
        cast_db.NewFace(frame_ids[0], Box(0.1, 0.1, 0.3, 0.5), 0.9, vector),
        cast_db.NewFace(frame_ids[1], Box(0.1, 0.1, 0.2, 0.3), 0.9, vector * 2),
        cast_db.NewFace(frame_ids[2], Box(0.5, 0.5, 0.6, 0.6), 0.8, vector * 3),
    ]
    return frame_ids, cast_db.add_faces(conn, faces)


def test_faces_and_shot_cast(conn: sqlite3.Connection, episode: Episode) -> None:
    _, face_ids = _faces(conn, episode)
    embeddings = cast_db.face_embeddings(conn, [face_ids[2], face_ids[0]])
    assert embeddings[:, 0].tolist() == [3.0, 1.0]
    cluster = cast_db.create_cluster(conn)
    cast_db.assign_faces(conn, face_ids[:2], cluster)
    (cast,) = cast_db.episode_cast(conn, episode.id)
    assert cast.cluster_id == cluster
    assert cast.presence == pytest.approx(1.0)
    assert cast.face_height == pytest.approx(0.4)
    assert cast.face_area == pytest.approx((0.08 + 0.02) / 2)


def test_name_merge_and_cleanup(conn: sqlite3.Connection, episode: Episode) -> None:
    _, face_ids = _faces(conn, episode)
    first, second, empty = (cast_db.create_cluster(conn) for _ in range(3))
    cast_db.assign_faces(conn, face_ids[:2], first)
    cast_db.assign_faces(conn, face_ids[2:], second)
    cast_db.name_cluster(conn, second, "Yami")
    cast_db.merge_clusters(conn, second, first)
    clusters = cast_db.list_clusters(conn)
    assert [(c.id, c.name, c.face_count) for c in clusters] == [
        (first, "Yami", 3),
        (empty, None, 0),
    ]
    cast_db.delete_empty_clusters(conn)
    assert [c.id for c in cast_db.list_clusters(conn)] == [first]
    cast_db.clear_episode_faces(conn, episode.id)
    assert [c.name for c in cast_db.list_clusters(conn)] == ["Yami"]
