"""Stage 3: who is on screen. Face detection and CCIP embeddings on kept frames, then clustering.

Clusters are series-wide: a new episode's faces first join the existing clusters they match,
and only the rest are clustered among themselves into new clusters. Names given to clusters
therefore carry over to every later episode.
"""

import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.library.cast import (
    NewFace,
    add_faces,
    assign_faces,
    clear_episode_faces,
    cluster_faces,
    clustered_face_ids,
    create_cluster,
    episode_faces,
    face_embeddings,
    get_cluster,
    merge_clusters,
)
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Box, Episode, Frame, ShotStatus
from tokimeki.library.shots import list_episode_frames, list_shots
from tokimeki.media.images import load_image, square_around
from tokimeki.models.ccip import (
    CCIP_DBSCAN_EPS,
    CCIP_DBSCAN_MIN_SAMPLES,
    CCIP_SAME_THRESHOLD,
    CcipEncoder,
    ccip_differences,
)
from tokimeki.models.faces import Detection, FaceDetector
from tokimeki.models.gpu import loaded
from tokimeki.stages.base import Context
from tokimeki.stages.clustering import NOISE, dbscan, evenly_spaced, nearest_cluster
from tokimeki.stages.frames import ensure_frames

NAME = "cast"

MIN_FACE_HEIGHT = 0.06
"""Faces shorter than this share of the frame are too small for CCIP and are ignored."""

HEAD_CROP_SCALE = 1.8
"""CCIP sees a square this many face-heights wide around the face: hair tells characters apart."""

MIN_CLUSTER_FACES = 4
MAX_EXEMPLARS = 64
NEAREST_EXEMPLARS = 3
SPLIT_EPS_FACTOR = 0.7
DETECT_LONG_SIDE = 640
FRAMES_PER_BATCH = 64

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
    found: list[FoundFace] = []
    for start in range(0, len(frames), FRAMES_PER_BATCH):
        batch = frames[start : start + FRAMES_PER_BATCH]
        images = [
            load_image(ctx.paths.frame_path(episode.id, f.frame_index), DETECT_LONG_SIDE)
            for f in batch
        ]
        for frame, detections in zip(batch, detector.detect(images), strict=True):
            found += [FoundFace(frame, d) for d in detections if d.y1 - d.y0 >= MIN_FACE_HEIGHT]
    return found


def _embed(ctx: Context, episode: Episode, found: list[FoundFace], encoder: CcipEncoder) -> None:
    for start in range(0, len(found), FRAMES_PER_BATCH):
        batch = found[start : start + FRAMES_PER_BATCH]
        images: dict[int, Image.Image] = {}
        crops: list[Image.Image] = []
        for face in batch:
            index = face.frame.frame_index
            if index not in images:
                images[index] = load_image(ctx.paths.frame_path(episode.id, index))
            image = images[index]
            crops.append(image.crop(head_box(face.detection, image.width, image.height)))
        embeddings = encoder.embed(crops)
        with transaction(ctx.conn):
            add_faces(
                ctx.conn,
                [
                    NewFace(
                        face.frame.id,
                        Box(
                            face.detection.x0,
                            face.detection.y0,
                            face.detection.x1,
                            face.detection.y1,
                        ),
                        face.detection.score,
                        embedding,
                    )
                    for face, embedding in zip(batch, embeddings, strict=True)
                ],
            )


def assign_clusters(conn: sqlite3.Connection, episode_id: int) -> int:
    """Put an episode's unassigned faces into clusters; returns how many clusters were created."""
    face_ids = [f.face.id for f in episode_faces(conn, episode_id) if f.face.cluster_id is None]
    if not face_ids:
        return 0
    embeddings = face_embeddings(conn, face_ids)
    labels = np.full(len(face_ids), NOISE, dtype=np.int64)
    exemplar_ids: list[int] = []
    exemplar_clusters: list[int] = []
    for cluster_id, members in clustered_face_ids(conn).items():
        chosen = evenly_spaced(members, MAX_EXEMPLARS)
        exemplar_ids += chosen
        exemplar_clusters += [cluster_id] * len(chosen)
    if exemplar_ids:
        distances = ccip_differences(embeddings, face_embeddings(conn, exemplar_ids))
        labels = nearest_cluster(
            distances, np.array(exemplar_clusters), CCIP_SAME_THRESHOLD, NEAREST_EXEMPLARS
        )
        for cluster_id in np.unique(labels[labels != NOISE]):
            assign_faces(
                conn, [face_ids[i] for i in np.flatnonzero(labels == cluster_id)], int(cluster_id)
            )
    rest = np.flatnonzero(labels == NOISE)
    return len(_cluster_new(conn, [face_ids[i] for i in rest], embeddings[rest], CCIP_DBSCAN_EPS))


def _cluster_new(
    conn: sqlite3.Connection,
    face_ids: Sequence[int],
    embeddings: NDArray[np.float32],
    eps: float,
) -> list[int]:
    if not face_ids:
        return []
    labels = dbscan(ccip_differences(embeddings, embeddings), eps, CCIP_DBSCAN_MIN_SAMPLES)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels[labels != NOISE])]
    groups = sorted((g for g in groups if len(g) >= MIN_CLUSTER_FACES), key=len, reverse=True)
    created: list[int] = []
    for group in groups:
        cluster_id = create_cluster(conn)
        assign_faces(conn, [face_ids[int(i)] for i in group], cluster_id)
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
    created = _cluster_new(
        conn, face_ids, face_embeddings(conn, face_ids), CCIP_DBSCAN_EPS * SPLIT_EPS_FACTOR
    )
    if created:
        merge_clusters(conn, created[0], cluster_id)
    return created[1:]
