"""What the report shows, gathered from the library. Dropped shots contribute counts only."""

import sqlite3
from dataclasses import dataclass

from tokimeki.library.cast import (
    FaceInShot,
    cluster_character_tags,
    cluster_faces,
    episode_cast,
    list_clusters,
)
from tokimeki.library.episodes import list_episodes, stage_params
from tokimeki.library.records import Cluster, Episode, Frame, Shot, ShotCast, ShotStatus
from tokimeki.library.shots import list_frames, list_shots, shot_tags

SAMPLE_FACES = 8


@dataclass(frozen=True)
class KeptShot:
    shot: Shot
    thumbnail: Frame
    cast: list[ShotCast]
    tags: list[tuple[str, float]]
    max_risk: float


@dataclass(frozen=True)
class EpisodeReport:
    episode: Episode
    total: int
    kept: list[KeptShot]
    dropped: list[Shot]
    filter_params: dict[str, str | int | float | bool] | None


@dataclass(frozen=True)
class ClusterReport:
    cluster: Cluster
    shots: int
    hints: list[tuple[str, float]]
    samples: list[FaceInShot]


@dataclass(frozen=True)
class ReportData:
    episodes: list[EpisodeReport]
    clusters: list[ClusterReport]


def _kept_shot(conn: sqlite3.Connection, shot: Shot, cast: list[ShotCast]) -> KeptShot | None:
    frames = list_frames(conn, shot.id)
    if not frames:
        return None
    return KeptShot(
        shot,
        frames[len(frames) // 2],
        cast,
        shot_tags(conn, shot.id),
        max(f.rating.questionable + f.rating.explicit for f in frames),
    )


def _samples(faces: list[FaceInShot], limit: int = SAMPLE_FACES) -> list[FaceInShot]:
    """The largest face of each shot, from the shots with the largest faces."""
    best: dict[int, FaceInShot] = {}
    for face in faces:
        current = best.get(face.shot_id)
        if current is None or face.face.box.area > current.face.box.area:
            best[face.shot_id] = face
    return sorted(best.values(), key=lambda f: f.face.box.area, reverse=True)[:limit]


def gather(conn: sqlite3.Connection) -> ReportData:
    episodes: list[EpisodeReport] = []
    for episode in list_episodes(conn):
        shots = list_shots(conn, episode.id)
        by_shot: dict[int, list[ShotCast]] = {}
        for entry in episode_cast(conn, episode.id):
            by_shot.setdefault(entry.shot_id, []).append(entry)
        kept = [
            k
            for s in shots
            if s.status is ShotStatus.KEPT
            and (k := _kept_shot(conn, s, by_shot.get(s.id, []))) is not None
        ]
        episodes.append(
            EpisodeReport(
                episode,
                len(shots),
                kept,
                [s for s in shots if s.status is ShotStatus.DROPPED],
                stage_params(conn, episode.id, "filter"),
            )
        )
    clusters: list[ClusterReport] = []
    for cluster in list_clusters(conn):
        faces = cluster_faces(conn, cluster.id)
        clusters.append(
            ClusterReport(
                cluster,
                len({f.shot_id for f in faces}),
                cluster_character_tags(conn, cluster.id),
                _samples(faces),
            )
        )
    return ReportData(episodes, clusters)
