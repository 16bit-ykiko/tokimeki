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
from tokimeki.stages import cast, pipeline
from tokimeki.stages.base import open_series, register_episodes
from tokimeki.stages.cast import MIN_CLUSTER_FACES, assign_clusters, head_box, split_cluster
from tokimeki.stages.clustering import average_linkage, evenly_spaced


def test_ccip_difference_is_half_one_minus_cosine() -> None:
    a = np.array([[1, 0], [0, 2]], dtype=np.float32)
    b = np.array([[3, 0], [-1, 0]], dtype=np.float32)
    assert ccip_differences(a, b).tolist() == [[0.0, 1.0], [0.5, 0.5]]


def test_average_linkage_does_not_chain() -> None:
    points = np.arange(10, dtype=np.float32) * 0.1
    distances = np.abs(points[:, None] - points[None, :])
    labels = average_linkage(distances, threshold=0.15)
    assert len(set(labels.tolist())) > 1
    pairs = np.array([0.0, 0.05, 5.0, 5.05], dtype=np.float32)
    distances = np.abs(pairs[:, None] - pairs[None, :])
    assert average_linkage(distances, threshold=0.15).tolist() == [0, 0, 1, 1]


def test_average_linkage_keeps_infinite_pairs_apart_and_honours_weights() -> None:
    distances = np.array([[0, np.inf, 0.1], [np.inf, 0, 0.12], [0.1, 0.12, 0]], dtype=np.float32)
    labels = average_linkage(distances, threshold=0.15).tolist()
    assert labels[0] != labels[1]
    assert labels[2] == labels[0]
    distances = np.array([[0, 0.12, 0.1], [0.12, 0, 0.17], [0.1, 0.17, 0]], dtype=np.float32)
    assert average_linkage(distances, 0.15).tolist() == [0, 0, 0]
    heavy = np.array([1.0, 1.0, 50.0])
    assert average_linkage(distances, 0.15, heavy).tolist() == [0, 1, 0]


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


def test_recluster_keeps_named_clusters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    (cluster,) = cast_db.list_clusters(ctx.conn)
    assert cast.recluster(ctx, episodes) == 1
    (renewed,) = cast_db.list_clusters(ctx.conn)
    assert renewed.id != cluster.id
    cast.rename(ctx, renewed.id, "Momo")
    assert cast.recluster(ctx, episodes) == 0
    assert cast_db.list_clusters(ctx.conn)[0].name == "Momo"


def _named(conn: sqlite3.Connection, episode_id: int, name: str) -> int:
    cluster = cast_db.create_cluster(conn, name)
    cast_db.assign_faces(
        conn, [f.face.id for f in cast_db.episode_faces(conn, episode_id)], cluster
    )
    return cluster


def _shot_of(conn: sqlite3.Connection, episode_id: int) -> int:
    return cast_db.episode_faces(conn, episode_id)[0].shot_id


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {c.label: c.face_count for c in cast_db.list_clusters(conn)}


def test_fixes_chain_and_follow_merges() -> None:
    conn = open_library(":memory:")
    first = _add_episode_faces(conn, "ep01.mkv", [0.0, 0.1])
    second = _add_episode_faces(conn, "ep02.mkv", [0.05, 0.08, 0.09])
    momo = _named(conn, first, "Momo")
    cast_db.assign_faces(conn, [f.face.id for f in cast_db.episode_faces(conn, second)], momo)
    nana, mea = cast_db.create_cluster(conn, "Nana"), cast_db.create_cluster(conn, "Mea")
    shot = _shot_of(conn, second)

    cast_db.fix_shot_cast(conn, shot, momo, mea)
    cast_db.fix_shot_cast(conn, shot, mea, nana)
    fix = cast_db.CastFix
    assert cast_db.list_cast_fixes(conn) == [fix(shot, momo, nana), fix(shot, mea, nana)]
    assert _counts(conn) == {"Nana": 3, "Momo": 2, "Mea": 0}

    cast_db.merge_clusters(conn, nana, mea)
    assert cast_db.list_cast_fixes(conn) == [fix(shot, momo, mea)]
    cast_db.assign_faces(conn, [f.face.id for f in cast_db.episode_faces(conn, second)], momo)
    cast_db.apply_cast_fixes(conn)
    assert _counts(conn) == {"Mea": 3, "Momo": 2}

    cast_db.merge_clusters(conn, mea, momo)
    assert cast_db.list_cast_fixes(conn) == []


def test_moved_shots_stay_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ep01.mkv", "ep02.mkv"):
        make_clip(tmp_path / name, ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    (cluster,) = cast_db.list_clusters(ctx.conn)
    cast.rename(ctx, cluster.id, "Momo")
    nana = cast_db.create_cluster(ctx.conn, "Nana")
    (shot,) = cast_db.named_cast_by_shot(ctx.conn, episodes[1].id)

    assert cast.move(ctx, [shot], None, "Nana") == [f"shot {shot}: Momo -> Nana"]
    assert _counts(ctx.conn) == {"Momo": 4, "Nana": 4}
    cast.recluster(ctx, episodes)
    cast.split(ctx, nana)
    pipeline.redo(ctx, episodes[1:], "cast")
    pipeline.run(ctx, episodes)
    assert _counts(ctx.conn) == {"Momo": 4, "Nana": 4}
    assert [n for n, _ in cast_db.named_cast_by_shot(ctx.conn, episodes[1].id)[shot]] == ["Nana"]
    assert [(src, dst) for _, src, dst in cast.fixes(ctx)] == [("Momo", "Nana")]

    cast.move(ctx, [shot], "Nana", None)
    cast.recluster(ctx, episodes)
    assert _counts(ctx.conn) == {"Momo": 4, "Nana": 0}
    with pytest.raises(ValueError, match="nobody named"):
        cast.move(ctx, [shot], None, "Momo")


def test_doubtful_shots_look_like_someone_else(tmp_path: Path) -> None:
    ctx = open_series(tmp_path)
    momo = [_add_episode_faces(ctx.conn, f"ep0{i}.mkv", [0.0, 0.02, 0.04]) for i in (1, 2)]
    odd = _add_episode_faces(ctx.conn, "ep03.mkv", [0.5, 0.52])
    cluster = _named(ctx.conn, momo[0], "Momo")
    for episode in (momo[1], odd):
        faces = cast_db.episode_faces(ctx.conn, episode)
        cast_db.assign_faces(ctx.conn, [f.face.id for f in faces], cluster)
    _named(ctx.conn, _add_episode_faces(ctx.conn, "ep04.mkv", [0.6, 0.62]), "Nana")

    first, *_ = cast.doubtful(ctx, "Momo", 5)
    assert (first.shot_id, first.nearest) == (_shot_of(ctx.conn, odd), "Nana")
    assert first.margin < 0 and first.keyframe.endswith(".jpg")
