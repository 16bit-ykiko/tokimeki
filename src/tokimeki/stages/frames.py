from collections.abc import Sequence

from tokimeki.library.records import Episode, Shot
from tokimeki.media.decode import extract_frames
from tokimeki.media.sampling import sample_frames
from tokimeki.stages.base import Context


def planned_frames(episode: Episode, shot: Shot) -> list[int]:
    return sample_frames(shot.start_frame, shot.end_frame, episode.fps)


def ensure_frames(ctx: Context, episode: Episode, shots: Sequence[Shot]) -> int:
    """Extract (NVDEC) the sampled frames of `shots` missing from the cache; returns how many."""
    missing = [
        (index, path)
        for shot in shots
        for index in planned_frames(episode, shot)
        if not (path := ctx.paths.frame_path(episode.id, index)).exists()
    ]
    if missing:
        extract_frames(
            ctx.paths.episode_file(episode.path),
            episode.width,
            episode.height,
            [index for index, _ in missing],
            [path for _, path in missing],
        )
    return len(missing)
