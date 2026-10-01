import sqlite3
from fractions import Fraction
from pathlib import Path

import fakes
import numpy as np
import pytest
from clips import make_clip
from numpy.typing import NDArray
from PIL import Image

from tokimeki.library import cast as cast_db
from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.db import open_library
from tokimeki.library.records import Box, Rating
from tokimeki.models.ccip import CcipEncoder, ccip_differences
from tokimeki.models.faces import Detection, FaceDetector
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes
from tokimeki.stages.cast import MIN_CLUSTER_FACES, assign_clusters, head_box, split_cluster
from tokimeki.stages.clustering import NOISE, dbscan, evenly_spaced, nearest_cluster


def test_ccip_difference_is_half_one_minus_cosine() -> None:
    a = np.array([[1, 0], [0, 2]], dtype=np.float32)
    b = np.array([[3, 0], [-1, 0]], dtype=np.float32)
    assert ccip_differences(a, b).tolist() == [[0.0, 1.0], [0.5, 0.5]]


def test_dbscan_separates_groups_and_noise() -> None:
    points = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 9.0], dtype=np.float32)
    distances = np.abs(points[:, None] - points[None, :])
    assert dbscan(distances, eps=0.15, min_samples=2).tolist() == [0, 0, 0, 1, 1, NOISE]


def test_nearest_cluster_uses_k_nearest_exemplars() -> None:
    distances = np.array([[0.05, 0.3, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    clusters = np.array([7, 7, 9, 9])
    assert nearest_cluster(distances, clusters, threshold=0.15, k=2).tolist() == [9, NOISE]


def test_evenly_spaced() -> None:
    assert evenly_spaced(list(range(10)), 3) == [0, 3, 6]
    assert evenly_spaced([1, 2], 5) == [1, 2]


def test_head_box_is_a_raised_square_inside_the_frame() -> None:
    x0, y0, x1, y1 = head_box(Detection(0.45, 0.4, 0.55, 0.6, 1.0), 1280, 720)
    assert x1 - x0 == y1 - y0
    assert x0 >= 0 and y0 >= 0 and x1 <= 1280 and y1 <= 720
    assert (y0 + y1) / 2 < 360
    assert head_box(Detection(0.0, 0.0, 0.9, 1.0, 1.0), 1280, 720) == (216, 0, 936, 720)


def _unit(angle: float) -> NDArray[np.float32]:
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _add_episode_faces(conn: sqlite3.Connection, path: str, angles: list[float]) -> int:
    episode = episode_db.register_episode(conn, path, 1280, 720, Fraction(24), 1000)
    (shot,) = shot_db.replace_shots(conn, episode.id, [(0, 1000)])
    safe = Rating(1, 0, 0, 0)
    shot_db.keep_shot(conn, shot.id, [shot_db.RatedFrame(i, safe, []) for i in range(len(angles))])
    frames = shot_db.list_frames(conn, shot.id)
    box = Box(0.4, 0.3, 0.6, 0.7)
    cast_db.add_faces(
        conn,
        [cast_db.NewFace(f.id, box, 0.9, _unit(a)) for f, a in zip(frames, angles, strict=True)],
    )
    return episode.id


def test_clusters_grow_across_episodes() -> None:
    conn = open_library(":memory:")
    first = _add_episode_faces(conn, "ep01.mkv", [0.0, 0.05, 0.1, 0.12, 2.0, 2.05, 2.1, 2.12, 4.0])
    assert assign_clusters(conn, first) == 2
    clusters = cast_db.list_clusters(conn)
    assert [c.face_count for c in clusters] == [4, 4]
    cast_db.name_cluster(conn, clusters[0].id, "Yami")

    second = _add_episode_faces(conn, "ep02.mkv", [0.02, 0.08, 2.02, 2.07, 2.09, 2.11])
    assert assign_clusters(conn, second) == 0
    assert [c.face_count for c in cast_db.list_clusters(conn)] == [8, 6]
    assert cast_db.get_cluster(conn, clusters[0].id).name == "Yami"


def test_split_keeps_the_name_on_the_largest_part() -> None:
    conn = open_library(":memory:")
    episode = _add_episode_faces(
        conn, "ep01.mkv", [0.0, 0.01, 0.02, 0.03, 0.04, 0.7, 0.71, 0.72, 0.73]
    )
    cluster = cast_db.create_cluster(conn, "Lala")
    cast_db.assign_faces(conn, [f.face.id for f in cast_db.episode_faces(conn, episode)], cluster)
    (created,) = split_cluster(conn, cluster)
    assert cast_db.get_cluster(conn, cluster).face_count == 5
    assert cast_db.get_cluster(conn, cluster).name == "Lala"
    assert cast_db.get_cluster(conn, created).face_count == MIN_CLUSTER_FACES


def test_stage_fills_shot_cast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    (cluster,) = cast_db.list_clusters(ctx.conn)
    assert cluster.face_count == 4
    (cast,) = cast_db.episode_cast(ctx.conn, 1)
    assert (cast.presence, cast.face_height) == pytest.approx((1.0, 0.4))
    assert cast_db.cluster_character_tags(ctx.conn, cluster.id) == [("momo_velia_deviluke", 1.0)]

    pipeline.redo(ctx, episodes, "cast")
    assert cast_db.list_clusters(ctx.conn) == []
    pipeline.run(ctx, episodes)
    assert len(cast_db.list_clusters(ctx.conn)) == 1


@pytest.mark.gpu
def test_face_models_run_on_the_gpu() -> None:
    image = Image.new("RGB", (640, 360), (200, 200, 200))
    detector = FaceDetector()
    try:
        assert detector.detect([image, image]) == [[], []]
    finally:
        detector.close()
    encoder = CcipEncoder()
    try:
        assert encoder.embed([image]).shape == (1, 768)
    finally:
        encoder.close()
