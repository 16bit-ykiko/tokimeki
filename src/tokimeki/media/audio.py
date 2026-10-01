"""Audio streams: picking the programme track and decoding it to PCM (audio decoding is light)."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int
    codec: str
    language: str
    title: str
    default: bool
    commentary: bool


def audio_streams(path: Path) -> list[AudioStream]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name:stream_tags=language,title:stream_disposition=default,comment",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = cast(dict[str, list[dict[str, object]]], json.loads(result.stdout))
    streams: list[AudioStream] = []
    for s in data.get("streams", []):
        tags = cast(dict[str, str], s.get("tags", {}))
        disposition = cast(dict[str, int], s.get("disposition", {}))
        title = tags.get("title", "")
        streams.append(
            AudioStream(
                index=int(str(s["index"])),
                codec=str(s.get("codec_name", "")),
                language=tags.get("language", ""),
                title=title,
                default=bool(disposition.get("default", 0)),
                commentary=bool(disposition.get("comment", 0)) or "comment" in title.lower(),
            )
        )
    return streams


def main_audio(path: Path) -> AudioStream:
    """The programme audio: never a commentary track, the default one if it is marked."""
    candidates = [s for s in audio_streams(path) if not s.commentary]
    if not candidates:
        raise AudioError(f"{path} has no audio stream other than commentary")
    return next((s for s in candidates if s.default), candidates[0])


def decode_audio(
    path: Path,
    sample_rate: int,
    stream_index: int | None = None,
    start: float | None = None,
    duration: float | None = None,
    channels: int = 1,
) -> NDArray[np.float32]:
    """Float32 samples of one audio stream (the first by default): `(samples,)` for mono,
    `(channels, samples)` otherwise."""
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    if start is not None:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    args += ["-map", f"0:{stream_index}" if stream_index is not None else "0:a:0"]
    args += ["-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        raise AudioError(f"decoding audio of {path} failed: {result.stderr.decode().strip()}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if channels == 1:
        return samples
    return np.ascontiguousarray(samples.reshape(-1, channels).T)


def encode_audio(samples: NDArray[np.float32], sample_rate: int, path: Path) -> None:
    """Write `(channels, samples)` float audio to `path` (the format follows its suffix)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.part{path.suffix}")
    channels = samples.shape[0]
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1"]
    args += ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "pipe:0"]
    args += [str(partial)]
    data = np.ascontiguousarray(samples.T, dtype=np.float32).tobytes()
    result = subprocess.run(args, input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise AudioError(f"writing {path} failed: {result.stderr.decode().strip()}")
    partial.replace(path)
