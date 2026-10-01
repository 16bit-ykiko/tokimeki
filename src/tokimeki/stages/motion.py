"""Stage 7: motion. Per frame of every kept shot, how much the picture changes, from small
luma frames NVDEC decodes and scales on the GPU; differenced there too, cached per episode.

Frames of dropped shots are discarded inside the decoder and never measured.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Episode, ShotStatus
from tokimeki.library.shots import list_shots
from tokimeki.media.decode import MOTION_SIZE, decode_luma
from tokimeki.models.gpu import loaded
from tokimeki.models.motion import CHANGED, MotionMeter
from tokimeki.paths import SeriesPaths
from tokimeki.stages.base import Context

NAME = "motion"

log = logging.getLogger("tokimeki")


def motion_path(paths: SeriesPaths, episode_id: int) -> Path:
    return paths.cache / "motion" / f"{episode_id}.npy"


def load_motion(paths: SeriesPaths, episode_id: int) -> NDArray[np.float32] | None:
    """`(frames, 2)`: per frame of the episode, its mean difference from the frame before and
    the share of its pixels that changed (0 at each shot's first frame, NaN outside kept
    shots); None before the stage has run."""
    path = motion_path(paths, episode_id)
    return np.load(path).astype(np.float32) if path.exists() else None


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [
        e
        for e in episodes
        if stage_done(ctx.conn, e.id, "filter") and not stage_done(ctx.conn, e.id, NAME)
    ]
    if not todo:
        return
    with loaded("motion", MotionMeter) as meter:
        for episode in todo:
            ranges = [
                (s.start_frame, s.end_frame)
                for s in list_shots(ctx.conn, episode.id, ShotStatus.KEPT)
            ]
            indices = np.concatenate([np.arange(a, b) for a, b in ranges] or [np.zeros(0, int)])
            size = max(episode.frame_count, int(indices.max(initial=-1)) + 1)
            diffs = np.full((size, 2), np.nan, dtype=np.float32)
            source = ctx.paths.episode_file(episode.path)
            position = 0
            previous: NDArray[np.uint8] | None = None
            for chunk in decode_luma(source, episode.width, episode.height, ranges):
                take = min(len(chunk), len(indices) - position)
                diffs[indices[position : position + take]] = meter.differences(
                    chunk[:take], previous
                )
                previous = chunk[take - 1]
                position += take
            if position < len(indices):
                log.warning("%s: %d kept frames not decoded", episode.path, len(indices) - position)
            for a, _ in ranges:
                if not np.isnan(diffs[a, 0]):
                    diffs[a] = 0.0
            path = motion_path(ctx.paths, episode.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, diffs)
            with transaction(ctx.conn):
                params = {"size": "x".join(map(str, MOTION_SIZE)), "changed": CHANGED}
                mark_stage_done(ctx.conn, episode.id, NAME, params)
            log.info("%s: motion of %d kept frames", episode.path, position)


def reset(ctx: Context, episode: Episode) -> None:
    motion_path(ctx.paths, episode.id).unlink(missing_ok=True)
    with transaction(ctx.conn):
        clear_stage(ctx.conn, episode.id, NAME)
