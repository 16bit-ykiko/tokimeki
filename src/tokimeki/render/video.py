"""Video through ffmpeg on the GPU: encoder settings for previews and finals, and subtitles
burnt in (NVDEC in, libass on the CPU, NVENC out)."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from tokimeki.media.decode import DecodeError


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


def ffmpeg(args: list[str], what: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise DecodeError(f"{what} failed:\n{result.stderr.decode(errors='replace').strip()}")


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
    ffmpeg(
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
