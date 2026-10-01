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

from tokimeki.mad.plan import Clip, Plan, PlanSlot
from tokimeki.media.decode import DecodeError
from tokimeki.paths import SeriesPaths

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


def slot_frames(plan: Plan, slot: PlanSlot) -> int:
    """Output frames for a slot, so cuts land on the frame nearest each beat."""
    fps = plan.fps
    return round(slot.end * fps) - round(slot.start * fps)


def clip_key(source: Path, clip: Clip, frames: int, settings: RenderSettings, fps: str) -> str:
    stat = source.stat()
    data = {
        "version": RENDER_VERSION,
        "source": str(source),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "start": clip.source_start,
        "speed": clip.speed,
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
    source: Path, clip: Clip, frames: int, settings: RenderSettings, fps: str, out: Path
) -> None:
    if clip.source_start is None or clip.speed is None:
        raise ValueError(f"slot {clip.slot} has not been placed")
    chain = ",".join(
        [
            f"scale_cuda=w={settings.width}:h={settings.height}:interp_algo=lanczos:format=nv12",
            f"setpts=(PTS-STARTPTS)/{clip.speed:.6f}",
            f"fps={fps}",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.mp4")
    _ffmpeg(
        [
            "-threads", "1", "-filter_threads", "1",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-ss", f"{clip.source_start:.6f}", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1",
            "-vf", chain, "-frames:v", str(frames),
            "-c:v", "h264_nvenc", "-preset", settings.preset, "-rc", "vbr",
            "-cq", str(settings.quality), "-b:v", "0", "-g", "48", "-bf", "0",
            "-video_track_timescale", "24000",
            str(partial),
        ],
        f"rendering slot {clip.slot}",
    )  # fmt: skip
    partial.replace(out)


def render(paths: SeriesPaths, plan: Plan, out_dir: Path, settings: RenderSettings) -> Path:
    """Render every clip (reusing cached ones), join them and lay the song under them."""
    fps = f"{plan.fps_num}/{plan.fps_den}"
    clips_dir = paths.cache / "clips"
    rendered: list[Path] = []
    for clip in sorted(plan.clips, key=lambda c: c.slot):
        source = paths.episode_file(clip.episode)
        frames = slot_frames(plan, plan.slots[clip.slot])
        path = clips_dir / f"{clip_key(source, clip, frames, settings, fps)}.mp4"
        if not path.exists():
            render_clip(source, clip, frames, settings, fps, path)
        rendered.append(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f"{settings.label}.concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in rendered), encoding="utf-8")
    video = out_dir / f"{settings.label}.video.mp4"
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(video)], "joining")
    song = plan.song
    duration = sum(slot_frames(plan, s) for s in plan.slots) / plan.fps
    fade = f"afade=t=in:d=0.05,afade=t=out:st={float(duration) - AUDIO_FADE:.3f}:d={AUDIO_FADE}"
    final = out_dir / f"{settings.label}.mp4"
    _ffmpeg(
        [
            "-i", str(video),
            "-ss", f"{song.offset + song.start:.6f}", "-t", f"{float(duration):.6f}",
            "-i", song.path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-af", fade, "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", str(final),
        ],
        "adding the song",
    )  # fmt: skip
    video.unlink()
    listing.unlink()
    return final
