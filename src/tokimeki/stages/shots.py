"""Stage 1: split each episode into shots (TransNetV2 over NVDEC-decoded frames)."""

import logging
import shutil
import time
from collections.abc import Sequence

from tokimeki.library.db import transaction
from tokimeki.library.episodes import (
    clear_stage,
    get_episode,
    mark_stage_done,
    set_frame_count,
    stage_done,
)
from tokimeki.library.records import Episode
from tokimeki.library.shots import replace_shots
from tokimeki.media.decode import decode_for_transnet
from tokimeki.media.probe import probe, require_nvdec
from tokimeki.models.gpu import loaded
from tokimeki.models.transnet import THRESHOLD, TransNet, shots_from_predictions
from tokimeki.stages.base import Context
from tokimeki.stages.frames import ensure_frames

NAME = "shots"
log = logging.getLogger("tokimeki")


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [e for e in episodes if not stage_done(ctx.conn, e.id, NAME)]
    if not todo:
        return
    with loaded("TransNetV2", TransNet) as network:
        for episode in todo:
            _detect(ctx, episode, network)


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        replace_shots(ctx.conn, episode.id, [])
        clear_stage(ctx.conn, episode.id, NAME)
    shutil.rmtree(ctx.paths.frames_dir(episode.id), ignore_errors=True)


def _detect(ctx: Context, episode: Episode, network: TransNet) -> None:
    reset(ctx, episode)
    path = ctx.paths.episode_file(episode.path)
    require_nvdec(probe(path), path)
    start = time.monotonic()
    frames = decode_for_transnet(path, episode.width, episode.height)
    decoded = time.monotonic()
    spans = shots_from_predictions(network.predict(frames))
    detected = time.monotonic()
    with transaction(ctx.conn):
        set_frame_count(ctx.conn, episode.id, len(frames))
        shots = replace_shots(ctx.conn, episode.id, spans)
    episode = get_episode(ctx.conn, episode.id)
    sampled = ensure_frames(ctx, episode, shots)
    with transaction(ctx.conn):
        mark_stage_done(ctx.conn, episode.id, NAME, {"model": "transnetv2", "threshold": THRESHOLD})
    log.info(
        "%s: %d frames, %d shots, %d sampled frames"
        " (decode %.0fs, TransNetV2 %.0fs, extract %.0fs)",
        episode.path,
        len(frames),
        len(shots),
        sampled,
        decoded - start,
        detected - decoded,
        time.monotonic() - detected,
    )
