import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

NVDEC_CODECS = frozenset({"h264", "hevc", "av1", "vp9", "vp8", "mpeg2video", "mpeg4", "vc1"})
H264_NVDEC_PIX_FMTS = frozenset({"yuv420p", "yuvj420p"})


class ProbeError(RuntimeError):
    pass


class NvdecUnsupportedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    codec: str
    pix_fmt: str

    @property
    def estimated_frames(self) -> int:
        return round(self.duration * self.fps)


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = cast(dict[str, object], json.loads(result.stdout))
    streams = cast(list[dict[str, object]], data.get("streams", []))
    if not streams:
        raise ProbeError(f"{path} has no video stream")
    stream = streams[0]
    fmt = cast(dict[str, object], data.get("format", {}))
    fps = Fraction(str(stream.get("avg_frame_rate", "0/1")))
    if fps == 0:
        fps = Fraction(str(stream.get("r_frame_rate", "0/1")))
    if fps == 0:
        raise ProbeError(f"{path}: cannot tell the frame rate")
    return VideoInfo(
        width=int(str(stream["width"])),
        height=int(str(stream["height"])),
        fps=fps,
        duration=float(str(fmt.get("duration", "0"))),
        codec=str(stream.get("codec_name", "")),
        pix_fmt=str(stream.get("pix_fmt", "")),
    )


def require_nvdec(info: VideoInfo, path: Path) -> None:
    """Fail before decoding when the GPU's decoder cannot handle the stream at all."""
    if info.codec not in NVDEC_CODECS:
        raise NvdecUnsupportedError(f"{path}: NVDEC cannot decode {info.codec}")
    if info.codec == "h264" and info.pix_fmt not in H264_NVDEC_PIX_FMTS:
        raise NvdecUnsupportedError(
            f"{path}: H.264 {info.pix_fmt} (e.g. 10-bit Hi10P) has no NVDEC support;"
            " use an 8-bit H.264 or an HEVC/AV1 release instead"
        )
