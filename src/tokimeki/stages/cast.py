"""Stage 3: who is on screen. Face detection and CCIP embeddings on kept frames, then clustering.

Clusters are series-wide: a new episode's faces first join the existing clusters they match,
and only the rest are clustered among themselves into new clusters. Names given to clusters
therefore carry over to every later episode.
"""

import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.library.cast import (
    NewFace,
    add_faces,
    assign_faces,
    clear_episode_faces,
    cluster_character_tags,
    cluster_faces,
    clustered_face_ids,
    create_cluster,
    episode_faces,
    face_embeddings,
    get_cluster,
    list_clusters,
    merge_clusters,
    name_cluster,
    unassign_unnamed,
)
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Box, Cluster, Episode, Frame, ShotStatus
from tokimeki.library.shots import list_episode_frames, list_shots
from tokimeki.media.images import load_image, square_around
from tokimeki.models.ccip import CcipEncoder, ccip_differences
from tokimeki.models.faces import Detection, FaceDetector
from tokimeki.models.gpu import loaded, prefetched
from tokimeki.stages.base import Context
from tokimeki.stages.clustering import average_linkage, evenly_spaced
from tokimeki.stages.frames import ensure_frames

NAME = "cast"

MIN_FACE_HEIGHT = 0.06
"""Faces shorter than this share of the frame are too small for CCIP and are ignored."""

HEAD_CROP_SCALE = 1.8
"""CCIP sees a square this many face-heights wide around the face: hair tells characters apart."""

CLUSTER_THRESHOLD = 0.15
"""Faces stay in one cluster while their mean CCIP difference is at most this.

CCIP's own same-character threshold is 0.178 (pairwise, best F1). Slightly lower keeps
look-alikes (the pink-haired Deviluke sisters) apart; a character split over two clusters
is one `tokimeki cast merge` away, two characters in one cluster are not.
"""

MIN_CLUSTER_FACES = 4
MAX_EXEMPLARS = 64
SPLIT_FACTOR = 0.7
DETECT_LONG_SIDE = 640

log = logging.getLogger("tokimeki")


@dataclass(frozen=True, slots=True)
class FoundFace:
    frame: Frame
    detection: Detection


def head_box(detection: Detection, width: int, height: int) -> tuple[int, int, int, int]:
    """The pixel box CCIP sees: a square around the face, raised a little to take in the hair."""
    d = detection
    return square_around((d.x0, d.y0, d.x1, d.y1), width, height, HEAD_CROP_SCALE, lift=0.1)


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [
        e
        for e in episodes
        if stage_done(ctx.conn, e.id, "filter") and not stage_done(ctx.conn, e.id, NAME)
    ]
    if not todo:
        return
    start = time.monotonic()
    found: dict[int, list[FoundFace]] = {}
    with loaded("face detector", FaceDetector) as detector:
        for episode in todo:
            reset(ctx, episode)
            ensure_frames(ctx, episode, list_shots(ctx.conn, episode.id, ShotStatus.KEPT))
            found[episode.id] = _detect(ctx, episode, detector)
    detected = time.monotonic()
    with loaded("CCIP", CcipEncoder) as encoder:
        for episode in todo:
            _embed(ctx, episode, found[episode.id], encoder)
    embedded = time.monotonic()
    for episode in todo:
        with transaction(ctx.conn):
            new_clusters = assign_clusters(ctx.conn, episode.id)
            mark_stage_done(
                ctx.conn,
                episode.id,
                NAME,
                {"min_face_height": MIN_FACE_HEIGHT, "head_crop_scale": HEAD_CROP_SCALE},
            )
        log.info(
            "%s: %d faces, %d new clusters", episode.path, len(found[episode.id]), new_clusters
        )
    log.info(
        "cast: detection %.0fs, CCIP %.0fs, clustering %.0fs",
        detected - start,
        embedded - detected,
        time.monotonic() - embedded,
    )


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        clear_episode_faces(ctx.conn, episode.id)
        clear_stage(ctx.conn, episode.id, NAME)


def _detect(ctx: Context, episode: Episode, detector: FaceDetector) -> list[FoundFace]:
    frames = list_episode_frames(ctx.conn, episode.id)
    size = detector.batch_size
    chunks = [frames[i : i + size] for i in range(0, len(frames), size)]

    def prepare(chunk: list[Frame]) -> NDArray[np.float32]:
        paths = [ctx.paths.frame_path(episode.id, f.frame_index) for f in chunk]
        return detector.prepare([load_image(p, DETECT_LONG_SIDE) for p in paths])

    found: list[FoundFace] = []
    for chunk, batch in prefetched(chunks, prepare):
        for frame, detections in zip(chunk, detector.infer(batch), strict=True):
            found += [FoundFace(frame, d) for d in detections if d.y1 - d.y0 >= MIN_FACE_HEIGHT]
    return found


def _embed(ctx: Context, episode: Episode, found: list[FoundFace], encoder: CcipEncoder) -> None:
    size = encoder.batch_size
    chunks = [found[i : i + size] for i in range(0, len(found), size)]

    def prepare(chunk: list[FoundFace]) -> NDArray[np.float32]:
        images: dict[int, Image.Image] = {}
        crops: list[Image.Image] = []
        for face in chunk:
            index = face.frame.frame_index
            if index not in images:
                images[index] = load_image(ctx.paths.frame_path(episode.id, index))
            image = images[index]
            crops.append(image.crop(head_box(face.detection, image.width, image.height)))
        return encoder.prepare(crops)

    for chunk, batch in prefetched(chunks, prepare):
        faces = [
            NewFace(
                f.frame.id,
                Box(f.detection.x0, f.detection.y0, f.detection.x1, f.detection.y1),
                f.detection.score,
                embedding,
            )
            for f, embedding in zip(chunk, encoder.infer(batch), strict=True)
        ]
        with transaction(ctx.conn):
            add_faces(ctx.conn, faces)


def assign_clusters(conn: sqlite3.Connection, episode_id: int) -> int:
    """Put an episode's unassigned faces into clusters; returns how many clusters were created."""
    face_ids = [f.face.id for f in episode_faces(conn, episode_id) if f.face.cluster_id is None]
    if not face_ids:
        return 0
    known = {
        cluster_id: face_embeddings(conn, evenly_spaced(members, MAX_EXEMPLARS))
        for cluster_id, members in clustered_face_ids(conn).items()
    }
    embeddings = face_embeddings(conn, face_ids)
    return len(_agglomerate(conn, face_ids, embeddings, CLUSTER_THRESHOLD, known))


def _agglomerate(
    conn: sqlite3.Connection,
    face_ids: Sequence[int],
    embeddings: NDArray[np.float32],
    threshold: float,
    known: Mapping[int, NDArray[np.float32]],
) -> list[int]:
    """Cluster faces together with the existing clusters, each a fixed group of exemplars.

    Faces that end up with an existing cluster join it; other groups of at least
    `MIN_CLUSTER_FACES` become new clusters. Returns the new cluster ids, largest first.
    """
    clusters = list(known)
    k, n = len(clusters), len(face_ids)
    distances = np.full((k + n, k + n), np.inf, dtype=np.float32)
    distances[k:, k:] = ccip_differences(embeddings, embeddings)
    for j, cluster_id in enumerate(clusters):
        column = ccip_differences(embeddings, known[cluster_id]).mean(axis=1)
        distances[k:, j] = column
        distances[j, k:] = column
    weights = np.array([len(known[c]) for c in clusters] + [1] * n, dtype=np.float64)
    labels = average_linkage(distances, threshold, weights)
    groups = sorted((np.flatnonzero(labels == label) for label in np.unique(labels)), key=len)
    created: list[int] = []
    for members in reversed(groups):
        anchors = [clusters[int(m)] for m in members if m < k]
        faces = [face_ids[int(m) - k] for m in members if m >= k]
        if anchors:
            assign_faces(conn, faces, anchors[0])
        elif len(faces) >= MIN_CLUSTER_FACES:
            cluster_id = create_cluster(conn)
            assign_faces(conn, faces, cluster_id)
            created.append(cluster_id)
    return created


def split_cluster(conn: sqlite3.Connection, cluster_id: int) -> list[int]:
    """Re-cluster one cluster's faces more tightly.

    The largest part keeps the cluster (and its name); the other parts become new clusters
    and faces that fit none are left unassigned. Returns the ids of the new clusters.
    """
    get_cluster(conn, cluster_id)
    face_ids = [f.face.id for f in cluster_faces(conn, cluster_id)]
    assign_faces(conn, face_ids, None)
    embeddings = face_embeddings(conn, face_ids)
    created = _agglomerate(conn, face_ids, embeddings, CLUSTER_THRESHOLD * SPLIT_FACTOR, {})
    if created:
        merge_clusters(conn, created[0], cluster_id)
    return created[1:]


def clusters_with_hints(ctx: Context) -> list[tuple[Cluster, list[tuple[str, float]]]]:
    """Every cluster with its WD14 character-tag name hints."""
    return [(c, cluster_character_tags(ctx.conn, c.id)) for c in list_clusters(ctx.conn)]


def rename(ctx: Context, cluster_id: int, name: str | None) -> None:
    with transaction(ctx.conn):
        name_cluster(ctx.conn, cluster_id, name)


def merge(ctx: Context, sources: Sequence[int], target: int) -> None:
    with transaction(ctx.conn):
        for source in sources:
            merge_clusters(ctx.conn, source, target)


def recluster(ctx: Context, episodes: Sequence[Episode]) -> int:
    """Cluster again every face not in a named cluster, from the stored embeddings (no GPU).

    Named clusters stay as they are and attract matching faces. Returns the new cluster count.
    """
    with transaction(ctx.conn):
        unassign_unnamed(ctx.conn)
        return sum(
            assign_clusters(ctx.conn, e.id) for e in episodes if stage_done(ctx.conn, e.id, NAME)
        )


def split(ctx: Context, cluster_id: int) -> list[int]:
    with transaction(ctx.conn):
        return split_cluster(ctx.conn, cluster_id)
