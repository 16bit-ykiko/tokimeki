"""Stage 2: the content filter. It runs before any other stage sees a frame.

Every sampled frame of a shot is rated by WD14; one unsafe frame drops the whole shot.
A dropped shot keeps only its time range and the dropped flag: its frames leave the cache
and the database, so nothing later (cast, report, cloud models) can see them.
"""

import logging
import time
from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.cast import delete_empty_clusters
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, mark_stage_done, stage_done
from tokimeki.library.records import Episode, Rating, Shot, ShotStatus, Tag, TagCategory
from tokimeki.library.shots import (
    RatedFrame,
    drop_shot,
    keep_shot,
    list_episode_frames,
    list_shots,
    reset_filter,
    status_counts,
)
from tokimeki.media.images import load_image
from tokimeki.models.gpu import loaded, prefetched
from tokimeki.models.wd14 import Prediction, Wd14Tagger
from tokimeki.stages.base import Context
from tokimeki.stages.frames import ensure_frames, planned_frames

NAME = "filter"

UNSAFE_THRESHOLD = 0.2
"""A frame is unsafe when WD14's `questionable` and `explicit` scores add up to this or more.

Conservative on purpose: WD14 calls a frame questionable when that score is the highest of
the four ratings, usually well above 0.5. Lowering the value drops more shots.
"""

SHOTS_PER_COMMIT = 32
log = logging.getLogger("tokimeki")


def is_unsafe(rating: Rating, threshold: float = UNSAFE_THRESHOLD) -> bool:
    return rating.questionable + rating.explicit >= threshold


def rated_frame(frame_index: int, prediction: Prediction) -> RatedFrame:
    r = prediction.rating
    tags = [Tag(name, TagCategory.GENERAL, score) for name, score in prediction.general.items()]
    tags += [
        Tag(name, TagCategory.CHARACTER, score) for name, score in prediction.character.items()
    ]
    return RatedFrame(
        frame_index,
        Rating(r["general"], r["sensitive"], r["questionable"], r["explicit"]),
        tags,
    )


def run(ctx: Context, episodes: Sequence[Episode]) -> None:
    todo = [
        e
        for e in episodes
        if stage_done(ctx.conn, e.id, "shots") and not stage_done(ctx.conn, e.id, NAME)
    ]
    if not todo:
        return
    with loaded("WD14", Wd14Tagger) as tagger:
        for episode in todo:
            _filter(ctx, episode, tagger)


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        reset_filter(ctx.conn, episode.id)
        delete_empty_clusters(ctx.conn)
        clear_stage(ctx.conn, episode.id, NAME)


def _predictions(
    ctx: Context, episode: Episode, tagger: Wd14Tagger, frames: Sequence[int]
) -> Iterator[Prediction]:
    """WD14 predictions for `frames` in order; the next batch is decoded while the GPU runs."""
    size = tagger.batch_size
    chunks = [frames[i : i + size] for i in range(0, len(frames), size)]

    def prepare(chunk: Sequence[int]) -> NDArray[np.float32]:
        paths = [ctx.paths.frame_path(episode.id, i) for i in chunk]
        return tagger.prepare([load_image(p, tagger.size) for p in paths])

    for _, batch in prefetched(chunks, prepare):
        yield from tagger.infer(batch)


def _commit(ctx: Context, episode: Episode, verdicts: list[tuple[Shot, list[RatedFrame]]]) -> None:
    dropped = {shot.id for shot, rated in verdicts if any(is_unsafe(f.rating) for f in rated)}
    for shot, rated in verdicts:
        if shot.id in dropped:
            for frame in rated:
                ctx.paths.frame_path(episode.id, frame.frame_index).unlink(missing_ok=True)
    with transaction(ctx.conn):
        for shot, rated in verdicts:
            if shot.id in dropped:
                drop_shot(ctx.conn, shot.id)
            else:
                keep_shot(ctx.conn, shot.id, rated)


def _filter(ctx: Context, episode: Episode, tagger: Wd14Tagger) -> None:
    start = time.monotonic()
    pending = list_shots(ctx.conn, episode.id, ShotStatus.PENDING)
    ensure_frames(ctx, episode, pending)
    plan = [(shot, planned_frames(episode, shot)) for shot in pending]
    predictions = _predictions(ctx, episode, tagger, [i for _, frames in plan for i in frames])
    for first in range(0, len(plan), SHOTS_PER_COMMIT):
        group = plan[first : first + SHOTS_PER_COMMIT]
        _commit(
            ctx,
            episode,
            [(shot, [rated_frame(i, next(predictions)) for i in frames]) for shot, frames in group],
        )
    prune_frames(ctx, episode)
    with transaction(ctx.conn):
        mark_stage_done(
            ctx.conn, episode.id, NAME, {"model": tagger.repo, "unsafe_threshold": UNSAFE_THRESHOLD}
        )
    counts = status_counts(ctx.conn, episode.id)
    total = sum(counts.values())
    log.info(
        "%s: kept %d, dropped %d of %d shots (%.0f%%) in %.0fs",
        episode.path,
        counts[ShotStatus.KEPT],
        counts[ShotStatus.DROPPED],
        total,
        100 * counts[ShotStatus.DROPPED] / max(total, 1),
        time.monotonic() - start,
    )


def prune_frames(ctx: Context, episode: Episode) -> int:
    """Delete cached frames that belong to no kept shot; returns how many."""
    keep = {f.frame_index for f in list_episode_frames(ctx.conn, episode.id)}
    removed = 0
    for path in ctx.paths.frames_dir(episode.id).glob("*.jpg"):
        if int(path.stem) not in keep:
            path.unlink()
            removed += 1
    return removed
