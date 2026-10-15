"""Rendering a plan: each clip on its own with NVDEC and NVENC, cached by its parameters
(effects included), then joined and laid over the mixed sound (the song with the plan's lines
over it).

Changing one clip in the plan re-renders only that clip. Previews are small and fast; the
final render is 1080p. A clip without effects stays on the GPU from decoding to encoding; one
with effects is built as `mad/fx.py` describes.
"""

import hashlib
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

from tokimeki.library.paths import SeriesPaths
from tokimeki.mad.fx import ClipFx, clip_fx, flash_source, zoom_filter
from tokimeki.mad.plan import Plan, SlotPlan, VoicePlan
from tokimeki.mad.song.analysis import SongAnalysis
from tokimeki.media.audio import decode_audio, encode_audio
from tokimeki.models import CPU_THREADS
from tokimeki.models.timeline import TimelineClip, write_timeline
from tokimeki.render.mix import DUCK_DB, RATE, VoiceClip, mix
from tokimeki.render.video import RenderSettings, burn_subtitles, ffmpeg

RENDER_VERSION = 1


def slot_frames(plan: Plan, slot: SlotPlan) -> int:
    """Output frames for a slot, counted from the excerpt start, so each cut lands on the
    frame nearest its beat."""
    fps = plan.frame_rate
    return round((slot.end - plan.start) * fps) - round((slot.start - plan.start) * fps)


def clip_key(
    source: Path,
    clip: SlotPlan,
    frames: int,
    settings: RenderSettings,
    fps: str,
    fx: ClipFx | None = None,
) -> str:
    stat = source.stat()
    data: dict[str, object] = {
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
    if fx is not None:
        data["fx"] = fx.key()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def _encode(settings: RenderSettings) -> list[str]:
    return [
        "-c:v", "h264_nvenc", "-preset", settings.preset, "-rc", "vbr",
        "-cq", str(settings.quality), "-b:v", "0", "-g", "48", "-bf", "0",
        "-video_track_timescale", "24000",
    ]  # fmt: skip


def _fx_graph(fx: ClipFx, speed: float, frames: int, settings: RenderSettings, fps: str) -> str:
    """The clip's filter graph with effects: scaled on the GPU, zoomed on the CPU, flashed on
    the GPU again."""
    graph = (
        f"[0:v]scale_cuda=w={settings.width}:h={settings.height}:interp_algo=lanczos"
        f":format=yuv420p,setpts=(PTS-STARTPTS)/{speed:.6f},fps={fps}"
    )
    zoom = zoom_filter(fx, frames, Fraction(fps))
    flash = flash_source(fx, settings.width, settings.height, fps)
    if zoom is not None:
        graph += f",hwdownload,format=yuv420p,{zoom}"
        if flash is not None:
            graph += ",hwupload_cuda"
    if flash is None:
        return f"{graph}[out]"
    overlay = "overlay_cuda=eof_action=pass:repeatlast=0"
    return f"{graph}[main];{flash}[white];[main][white]{overlay}[out]"


def render_clip(
    source: Path,
    clip: SlotPlan,
    frames: int,
    settings: RenderSettings,
    fps: str,
    out: Path,
    fx: ClipFx | None = None,
) -> None:
    speed = clip.derived_speed
    if clip.source_in is None or speed is None:
        raise ValueError(f"the slot at {clip.start:.3f}s has no window; refine the plan")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.mp4")
    what = f"rendering the slot at {clip.start:.3f}s"
    clean = ["-an", "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1"]
    if fx is None:
        chain = ",".join(
            [
                f"scale_cuda=w={settings.width}:h={settings.height}:interp_algo=lanczos:format=nv12",
                f"setpts=(PTS-STARTPTS)/{speed:.6f}",
                f"fps={fps}",
            ]
        )
        ffmpeg(
            [
                "-threads", "1", "-filter_threads", "1",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-ss", f"{clip.source_in:.6f}", "-i", str(source),
                "-map", "0:v:0", *clean, "-vf", chain, "-frames:v", str(frames),
                *_encode(settings), str(partial),
            ],
            what,
        )  # fmt: skip
        partial.replace(out)
        return
    script = out.with_suffix(".filter.txt")
    script.write_text(_fx_graph(fx, speed, frames, settings, fps), encoding="utf-8")
    try:
        ffmpeg(
            [
                "-threads", "1", "-filter_complex_threads", str(CPU_THREADS),
                "-init_hw_device", "cuda=gpu", "-filter_hw_device", "gpu",
                "-hwaccel", "cuda", "-hwaccel_device", "gpu", "-hwaccel_output_format", "cuda",
                "-ss", f"{clip.source_in:.6f}", "-i", str(source),
                "-/filter_complex", str(script), "-map", "[out]", *clean,
                "-frames:v", str(frames), *_encode(settings), str(partial),
            ],
            what,
        )  # fmt: skip
    finally:
        script.unlink(missing_ok=True)
    partial.replace(out)


def duration(plan: Plan) -> float:
    return float(sum(slot_frames(plan, s) for s in plan.slots) / plan.frame_rate)


def render_audio(
    plan: Plan, song: SongAnalysis, voices: Sequence[tuple[VoicePlan, Path]], out: Path
) -> Path:
    """The MAD's sound: the song excerpt with each voice (taken from its episode's voice stem)
    mixed over it."""
    seconds = duration(plan)
    length = round(seconds * RATE)
    music = decode_audio(
        song.source.path,
        RATE,
        song.source.stream,
        start=song.source.offset + plan.start,
        duration=seconds,
        channels=2,
    )
    music = np.pad(music[:, :length], ((0, 0), (0, max(0, length - music.shape[1]))))
    clips: list[VoiceClip] = []
    for v, stem in voices:
        if v.source_in is None or v.length is None:
            raise ValueError(f"the voice at {v.at:.2f}s has no in/out; refine the plan")
        samples = decode_audio(stem, RATE, start=v.source_in, duration=v.length, channels=2)
        duck = v.duck if v.duck is not None else DUCK_DB
        clips.append(VoiceClip(v.at - plan.start, samples, v.gain, duck))
    encode_audio(mix(music, clips), RATE, out)
    return out


def render(
    paths: SeriesPaths,
    plan: Plan,
    episodes: dict[int, str],
    audio: Path,
    out_dir: Path,
    settings: RenderSettings,
    subtitles: tuple[Path, Path] | None = None,
) -> Path:
    """Render every clip (reusing cached ones), join them, burn in `subtitles` (an ASS file
    and its fonts directory) if given, and lay the mixed `audio` under them.

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
        fx = clip_fx(plan, slot)
        path = clips_dir / f"{clip_key(source, slot, frames, settings, fps, fx)}.mp4"
        if not path.exists():
            render_clip(source, slot, frames, settings, fps, path, fx)
        rendered.append(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f"{settings.label}.concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in rendered), encoding="utf-8")
    video = out_dir / f"{settings.label}.video.mp4"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(video)], "joining")
    if subtitles is not None:
        subbed = out_dir / f"{settings.label}.subbed.mp4"
        burn_subtitles(video, subtitles[0], subtitles[1], settings, subbed)
        subbed.replace(video)
    final = out_dir / f"{settings.label}.mp4"
    ffmpeg(
        [
            "-i", str(video), "-i", str(audio),
            "-map", "0:v", "-map", "1:a:0", "-map_chapters", "-1", "-map_metadata", "-1",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-t", f"{duration(plan):.6f}", "-movflags", "+faststart", str(final),
        ],
        "adding the sound",
    )  # fmt: skip
    video.unlink()
    listing.unlink()
    return final


def export_timeline(
    paths: SeriesPaths,
    plan: Plan,
    episodes: dict[int, str],
    song: SongAnalysis,
    path: Path,
    voices: Sequence[tuple[VoicePlan, Path]] = (),
) -> None:
    """An OpenTimelineIO timeline of the cuts, the song and the voices (from the stems; the
    ducking is left to the editor), for finishing in an editor (e.g. Resolve)."""
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
    rate = plan.frame_rate
    music = [TimelineClip(song.title, song.source.path, song.source.offset + plan.start, total)]
    spoken = [
        TimelineClip(
            f"voice {i}: {v.text}",
            stem,
            v.source_in,
            round(v.length * rate),
            at=round((v.at - plan.start) * rate),
        )
        for i, (v, stem) in enumerate(voices)
        if v.source_in is not None and v.length is not None
    ]
    write_timeline(path, plan.name, float(rate), video, [music, spoken] if spoken else [music])
