"""Decoding on the GPU (NVDEC), resizing there too, and only small frames copied back.

ffmpeg never falls back to CPU decoding silently here: frames stay in CUDA memory until
`hwdownload`, so a software-decoded frame cannot enter the `scale_cuda` filters and the
run fails instead.
"""

import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.media.probe import probe

TRANSNET_SIZE = (48, 27)
FRAME_HEIGHT = 720
JPEG_QUALITY = 3

_CPU_THREADS = ["-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1"]


class DecodeError(RuntimeError):
    pass


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def _input(path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *_CPU_THREADS,
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
    ]


def gpu_resize(width: int, height: int, target_width: int, target_height: int) -> list[str]:
    """`scale_cuda` steps from `width`x`height` down to the target, also converting to 8-bit nv12.

    A single step from 1080p to a thumbnail aliases badly, so the frame is halved (a 2x2
    average) while it stays at least as large as the target, then resized once more.
    """
    steps: list[str] = []
    w, h = width, height
    while w // 2 >= target_width and h // 2 >= target_height:
        w, h = _even(w / 2), _even(h / 2)
        steps.append(f"scale_cuda=w={w}:h={h}:interp_algo=bilinear:format=nv12")
    steps.append(f"scale_cuda=w={target_width}:h={target_height}:interp_algo=lanczos:format=nv12")
    return steps


def _run(args: Sequence[str], what: str) -> bytes:
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise DecodeError(f"NVDEC {what} failed (no CPU fallback):\n{stderr}")
    return result.stdout


def decode_for_transnet(path: Path, width: int, height: int) -> NDArray[np.uint8]:
    """Every frame of a `width`x`height` video as `(frames, 27, 48, 3)` RGB for TransNetV2."""
    tw, th = TRANSNET_SIZE
    mid_w, mid_h = _even(min(width, tw * 5)), _even(min(height, th * 5))
    chain = [*gpu_resize(width, height, mid_w, mid_h), "hwdownload", "format=nv12"]
    chain.append(f"scale={tw}:{th}:flags=area")
    raw = _run(
        [*_input(path), "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        "decode",
    )
    frame_bytes = tw * th * 3
    if len(raw) % frame_bytes:
        raise DecodeError(f"{path}: decoder returned a partial frame")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, th, tw, 3)


MOTION_SIZE = (160, 90)


def _ranges_expr(ranges: Sequence[tuple[int, int]]) -> str:
    """True for frames in any `[first, end)` range: a balanced `if(lt(n, k), …)` tree."""
    if len(ranges) == 1:
        first, end = ranges[0]
        return f"between(n\\,{first}\\,{end - 1})"
    mid = len(ranges) // 2
    left, right = _ranges_expr(ranges[:mid]), _ranges_expr(ranges[mid:])
    return f"if(lt(n\\,{ranges[mid][0]})\\,{left}\\,{right})"


def decode_luma(
    path: Path,
    width: int,
    height: int,
    ranges: Sequence[tuple[int, int]],
    chunk: int = 512,
) -> Iterator[NDArray[np.uint8]]:
    """The luma of the frames in `[first, end)` frame ranges (sorted, apart), scaled down to
    `MOTION_SIZE` on the GPU, in chunks of `(frames, height, width)`.

    Frames outside the ranges are dropped on the GPU, before anything is copied back.
    """
    if not ranges:
        return
    w, h = MOTION_SIZE
    chain = [f"select={_ranges_expr(ranges)}", *gpu_resize(width, height, w, h), "hwdownload"]
    chain.append("format=nv12")
    with tempfile.TemporaryDirectory(prefix="tokimeki-motion-") as tmp:
        script = Path(tmp) / "filter.txt"
        script.write_text(",".join(chain))
        args = [*_input(path), "-/vf", str(script), "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None and process.stderr is not None
        frame_bytes = w * h * 3 // 2
        try:
            while True:
                data = process.stdout.read(frame_bytes * chunk)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise DecodeError(f"{path}: decoder returned a partial frame")
                frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, frame_bytes)
                yield frames[:, : w * h].reshape(-1, h, w)
        finally:
            process.stdout.close()
            stderr = process.stderr.read().decode(errors="replace").strip()
            process.stderr.close()
            if process.wait() != 0:
                raise DecodeError(f"NVDEC motion decode failed (no CPU fallback):\n{stderr}")


def frame_size(width: int, height: int, max_height: int = FRAME_HEIGHT) -> tuple[int, int]:
    out_h = _even(min(max_height, height))
    return _even(width * out_h / height), out_h


def _select_expr(indices: Sequence[int]) -> str:
    """A balanced `if(lt(n, k), …)` tree, so each frame costs O(log n) to test."""
    if len(indices) == 1:
        return f"eq(n\\,{indices[0]})"
    mid = len(indices) // 2
    left, right = _select_expr(indices[:mid]), _select_expr(indices[mid:])
    return f"if(lt(n\\,{indices[mid]})\\,{left}\\,{right})"


def extract_frames(
    path: Path,
    width: int,
    height: int,
    frame_indices: Sequence[int],
    out_paths: Sequence[Path],
) -> None:
    """Write the frames at `frame_indices` (decode order, from 0) as JPEGs to `out_paths`.

    Frames are scaled down to at most `FRAME_HEIGHT` lines on the GPU.
    """
    if len(frame_indices) != len(out_paths):
        raise ValueError("one output path per frame index")
    if not frame_indices:
        return
    order = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    indices = [frame_indices[i] for i in order]
    if len(set(indices)) != len(indices):
        raise ValueError("frame indices must be unique")
    out_w, out_h = frame_size(width, height)
    chain = [
        f"select={_select_expr(indices)}",
        *gpu_resize(width, height, out_w, out_h),
        "hwdownload",
        "format=nv12",
    ]
    with tempfile.TemporaryDirectory(prefix="tokimeki-frames-") as tmp:
        tmp_dir = Path(tmp)
        script = tmp_dir / "filter.txt"
        script.write_text(",".join(chain))
        _run(
            [
                *_input(path),
                "-/vf",
                str(script),
                "-c:v",
                "mjpeg",
                "-q:v",
                str(JPEG_QUALITY),
                "-start_number",
                "0",
                str(tmp_dir / "%07d.jpg"),
            ],
            "frame extraction",
        )
        written = sorted(tmp_dir.glob("*.jpg"))
        if len(written) != len(indices):
            raise DecodeError(
                f"{path}: asked for {len(indices)} frames, decoder wrote {len(written)}"
            )
        for src, i in zip(written, order, strict=True):
            out_paths[i].parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, out_paths[i])


def nvdec_selftest(work_dir: Path) -> int:
    """Encode a one-second synthetic clip and decode it through NVDEC; returns the frame count."""
    clip = work_dir / "selftest.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    info = probe(clip)
    return len(decode_for_transnet(clip, info.width, info.height))
