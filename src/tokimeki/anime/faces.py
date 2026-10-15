"""Stage 3: who is on screen. Face detection and CCIP embeddings on kept frames, then clustering.

Clusters are series-wide: a new episode's faces first join the existing clusters they match,
and only the rest are clustered among themselves into new clusters. Names given to clusters
therefore carry over to every later episode.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tokimeki.anime.frames import ensure_frames
from tokimeki.cast.clusters import assign_clusters
from tokimeki.library.cast import NewFace, add_faces, clear_episode_faces
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Box, Episode, Frame, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import list_episode_frames, list_shots
from tokimeki.media.images import load_image, square_around
from tokimeki.models.ccip import CcipEncoder
from tokimeki.models.faces import Detection, FaceDetector
from tokimeki.models.gpu import loaded, prefetched

NAME = "cast"

MIN_FACE_HEIGHT = 0.06
"""Faces shorter than this share of the frame are too small for CCIP and are ignored."""

HEAD_CROP_SCALE = 1.8
"""CCIP sees a square this many face-heights wide around the face: hair tells characters apart."""

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
