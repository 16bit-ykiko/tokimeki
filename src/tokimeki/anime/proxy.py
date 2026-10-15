"""Browser-playable copies of the episodes, for scrubbing: H.264 at 540p with a keyframe every
half second, AAC stereo of the main track and the index first (`+faststart`).

Everything outside kept shots is black and silent, so the content filter holds in a browser
too. NVDEC decodes and `scale_cuda` shrinks on the GPU, the blacking out runs on the 540p
frames on the CPU (`drawbox`), NVENC encodes. Made on demand (`tokimeki proxy`), cached in
`cache/proxy/<episode id>.mp4` and made again when the kept shots change.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from tokimeki.library.episodes import stage_done
from tokimeki.library.paths import SeriesPaths
from tokimeki.library.records import Episode, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import list_shots
from tokimeki.media.audio import AudioError, main_audio
from tokimeki.media.decode import gpu_resize
from tokimeki.render.video import ffmpeg

log = logging.getLogger("tokimeki")

HEIGHT = 540
QUALITY = 28
KEYFRAME = 0.5
"""Seconds between keyframes, so a seek lands at once."""


class ProxyError(RuntimeError):
    pass


def proxy_path(paths: SeriesPaths, episode_id: int) -> Path:
    return paths.cache / "proxy" / f"{episode_id}.mp4"


def _stamp_path(paths: SeriesPaths, episode_id: int) -> Path:
    return proxy_path(paths, episode_id).with_suffix(".json")


def inside(var: str, ranges: Sequence[tuple[float, float]]) -> str:
    """An ffmpeg expression, true while `var` is in any `[start, end)` range (sorted): a
    balanced tree, so a frame costs a few comparisons."""
    if not ranges:
        return "0"
    if len(ranges) == 1:
        start, end = ranges[0]
        return f"gte({var},{start:g})*lt({var},{end:g})"
    mid = len(ranges) // 2
    left, right = inside(var, ranges[:mid]), inside(var, ranges[mid:])
    return f"if(lt({var},{ranges[mid][0]:g}),{left},{right})"


def _kept(ctx: Context, episode: Episode) -> list[tuple[int, int]]:
    joined: list[tuple[int, int]] = []
    for shot in list_shots(ctx.conn, episode.id, ShotStatus.KEPT):
        if joined and joined[-1][1] == shot.start_frame:
            joined[-1] = (joined[-1][0], shot.end_frame)
        else:
            joined.append((shot.start_frame, shot.end_frame))
    return joined


def _stamp(kept: Sequence[tuple[int, int]]) -> str:
    return hashlib.sha1(json.dumps(list(kept)).encode()).hexdigest()[:16]


def fresh_proxy(ctx: Context, episode: Episode) -> Path | None:
    """The episode's proxy if it shows exactly the shots kept now; a stale one could show a
    shot the filter has since dropped."""
    out = proxy_path(ctx.paths, episode.id)
    marker = _stamp_path(ctx.paths, episode.id)
    if not stage_done(ctx.conn, episode.id, "filter") or not out.exists() or not marker.exists():
        return None
    return out if marker.read_text() == _stamp(_kept(ctx, episode)) else None


def make_proxy(ctx: Context, episode: Episode) -> Path:
    """The episode's proxy, made unless the one in the cache shows the same kept shots."""
    if not stage_done(ctx.conn, episode.id, "filter"):
        raise ProxyError(f"{episode.path}: run `tokimeki ingest` first; the filter has not run")
    found = fresh_proxy(ctx, episode)
    if found is not None:
        return found
    kept = _kept(ctx, episode)
    stamp = _stamp(kept)
    out = proxy_path(ctx.paths, episode.id)
    marker = _stamp_path(ctx.paths, episode.id)
    marker.unlink(missing_ok=True)
    source = ctx.paths.episode_file(episode.path)
    width = round(HEIGHT * episode.width / episode.height / 2) * 2
    frames = [(float(a), float(b)) for a, b in kept]
    seconds = [(episode.seconds(a), episode.seconds(b)) for a, b in kept]
    video = ",".join(
        [
            *gpu_resize(episode.width, episode.height, width, HEIGHT),
            "hwdownload",
            "format=nv12",
            "format=yuv420p",
            f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='not({inside('n', frames)})'",
        ]
    )
    graph = f"[0:v:0]{video}[v]"
    maps = ["-map", "[v]"]
    try:
        stream = main_audio(source).index
        audio = f"aformat=channel_layouts=stereo,volume=0:enable='not({inside('t', seconds)})'"
        graph += f";[0:{stream}]{audio}[a]"
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
    except AudioError:
        log.warning("%s: no audio for the proxy", episode.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = out.with_suffix(".filter.txt")
    script.write_text(graph, encoding="utf-8")
    partial = out.with_suffix(".part.mp4")
    gop = max(1, round(KEYFRAME * episode.fps))
    try:
        ffmpeg(
            [
                "-threads", "1", "-filter_complex_threads", "1",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(source),
                "-/filter_complex", str(script), *maps,
                "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1",
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(QUALITY),
                "-b:v", "0", "-g", str(gop), "-bf", "0",
                "-movflags", "+faststart", str(partial),
            ],
            f"making the proxy of {episode.path}",
        )  # fmt: skip
    finally:
        script.unlink(missing_ok=True)
    partial.replace(out)
    marker.write_text(stamp)
    log.info("%s: proxy %.0f MB", episode.path, out.stat().st_size / 1e6)
    return out
