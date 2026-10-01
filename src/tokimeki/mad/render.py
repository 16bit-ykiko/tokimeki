"""Rendering a plan: each clip on its own with NVDEC and NVENC, cached by its parameters,
then joined and laid over the song.

Changing one clip in the plan re-renders only that clip. Previews are small and fast; the
final render is 1080p.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tokimeki.mad.plan import Plan, SlotPlan
from tokimeki.media.decode import DecodeError
from tokimeki.models.timeline import TimelineClip, write_timeline
from tokimeki.paths import SeriesPaths
from tokimeki.song.analysis import SongAnalysis

RENDER_VERSION = 1
AUDIO_FADE = 0.5


@dataclass(frozen=True, slots=True)
class RenderSettings:
    label: str
    width: int
    height: int
    quality: int
    """NVENC constant-quality level (lower is better)."""
    preset: str


PREVIEW = RenderSettings("preview", 640, 360, 28, "p4")
FINAL = RenderSettings("final", 1920, 1080, 19, "p6")


def slot_frames(plan: Plan, slot: SlotPlan) -> int:
    """Output frames for a slot, counted from the excerpt start, so each cut lands on the
    frame nearest its beat."""
    fps = plan.frame_rate
    return round((slot.end - plan.start) * fps) - round((slot.start - plan.start) * fps)


def clip_key(source: Path, clip: SlotPlan, frames: int, settings: RenderSettings, fps: str) -> str:
    stat = source.stat()
    data = {
        "version": RENDER_VERSION,
        "source": str(source),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "start": clip.source_in,
        "speed": clip.derived_speed,
        "frames": frames,
        "fps": fps,
        "settings": [settings.width, settings.height, settings.quality, settings.preset],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def _ffmpeg(args: list[str], what: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise DecodeError(f"{what} failed:\n{result.stderr.decode(errors='replace').strip()}")


def render_clip(
    source: Path, clip: SlotPlan, frames: int, settings: RenderSettings, fps: str, out: Path
) -> None:
    speed = clip.derived_speed
    if clip.source_in is None or speed is None:
        raise ValueError(f"the slot at {clip.start:.3f}s has no window; refine the plan")
    chain = ",".join(
        [
            f"scale_cuda=w={settings.width}:h={settings.height}:interp_algo=lanczos:format=nv12",
            f"setpts=(PTS-STARTPTS)/{speed:.6f}",
            f"fps={fps}",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.mp4")
    _ffmpeg(
        [
            "-threads", "1", "-filter_threads", "1",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-ss", f"{clip.source_in:.6f}", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1",
            "-vf", chain, "-frames:v", str(frames),
            "-c:v", "h264_nvenc", "-preset", settings.preset, "-rc", "vbr",
            "-cq", str(settings.quality), "-b:v", "0", "-g", "48", "-bf", "0",
            "-video_track_timescale", "24000",
            str(partial),
        ],
        f"rendering the slot at {clip.start:.3f}s",
    )  # fmt: skip
    partial.replace(out)


def _filter_path(path: Path) -> str:
    """A path quoted for an ffmpeg filter argument."""
    escaped = str(path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return f"'{escaped}'"


def burn_subtitles(
    video: Path, subtitles: Path, fonts: Path, settings: RenderSettings, out: Path
) -> None:
    """Draw an ASS file onto a video: NVDEC in, libass on the CPU, NVENC out."""
    chain = ",".join(
        [
            "hwdownload",
            "format=nv12",
            "format=yuv420p",
            f"subtitles=filename={_filter_path(subtitles)}:fontsdir={_filter_path(fonts)}",
            "format=nv12",
            "hwupload_cuda",
        ]
    )
    script = out.with_suffix(".filter.txt")
    script.write_text(chain, encoding="utf-8")
    _ffmpeg(
        [
            "-threads", "1", "-filter_threads", "1",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video),
            "-/vf", str(script), "-map_chapters", "-1",
            "-c:v", "h264_nvenc", "-preset", settings.preset, "-rc", "vbr",
            "-cq", str(settings.quality), "-b:v", "0", "-g", "48", "-bf", "0",
            str(out),
        ],
        "burning in the subtitles",
    )  # fmt: skip
    script.unlink()


def render(
    paths: SeriesPaths,
    plan: Plan,
    episodes: dict[int, str],
    song: SongAnalysis,
    out_dir: Path,
    settings: RenderSettings,
    subtitles: tuple[Path, Path] | None = None,
) -> Path:
    """Render every clip (reusing cached ones), join them, burn in `subtitles` (an ASS file
    and its fonts directory) if given, and lay the song under them.

    `episodes` maps each slot's shot id to its episode path.
    """
    fps = plan.fps
    clips_dir = paths.cache / "clips"
    rendered: list[Path] = []
    for slot in plan.slots:
        if slot.shot is None:
            raise ValueError(f"the slot at {slot.start:.3f}s has no shot")
        source = paths.episode_file(episodes[slot.shot])
        frames = slot_frames(plan, slot)
        path = clips_dir / f"{clip_key(source, slot, frames, settings, fps)}.mp4"
        if not path.exists():
            render_clip(source, slot, frames, settings, fps, path)
        rendered.append(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f"{settings.label}.concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in rendered), encoding="utf-8")
    video = out_dir / f"{settings.label}.video.mp4"
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(video)], "joining")
    if subtitles is not None:
        subbed = out_dir / f"{settings.label}.subbed.mp4"
        burn_subtitles(video, subtitles[0], subtitles[1], settings, subbed)
        subbed.replace(video)
    duration = float(sum(slot_frames(plan, s) for s in plan.slots) / plan.frame_rate)
    fade = f"afade=t=in:d=0.05,afade=t=out:st={duration - AUDIO_FADE:.3f}:d={AUDIO_FADE}"
    audio_map = f"1:{song.source.stream}" if song.source.stream is not None else "1:a:0"
    final = out_dir / f"{settings.label}.mp4"
    _ffmpeg(
        [
            "-i", str(video),
            "-ss", f"{song.source.offset + plan.start:.6f}", "-t", f"{duration:.6f}",
            "-i", str(song.source.path),
            "-map", "0:v", "-map", audio_map, "-map_chapters", "-1", "-map_metadata", "-1",
            "-c:v", "copy",
            "-af", fade, "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", str(final),
        ],
        "adding the song",
    )  # fmt: skip
    video.unlink()
    listing.unlink()
    return final


def export_timeline(
    paths: SeriesPaths, plan: Plan, episodes: dict[int, str], song: SongAnalysis, path: Path
) -> None:
    """An OpenTimelineIO timeline of the cuts, for finishing in an editor (e.g. Resolve)."""
    video = [
        TimelineClip(
            f"{i} {slot.section} - shot {slot.shot}",
            paths.episode_file(episodes[slot.shot]),
            slot.source_in or 0.0,
            slot_frames(plan, slot),
            slot.derived_speed or 1.0,
        )
        for i, slot in enumerate(plan.slots)
        if slot.shot is not None
    ]
    total = sum(slot_frames(plan, s) for s in plan.slots)
    audio = [TimelineClip(song.title, song.source.path, song.source.offset + plan.start, total)]
    write_timeline(path, plan.name, float(plan.frame_rate), video, audio)
