import subprocess
from collections.abc import Sequence
from pathlib import Path


def make_clip(
    path: Path,
    sources: Sequence[str],
    seconds: float = 1.0,
    size: str = "320x180",
    rate: str = "24",
) -> Path:
    """An H.264 clip of lavfi `sources` (such as `color=c=red`) with hard cuts between them."""
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for source in sources:
        joint = ":" if "=" in source else "="
        args += ["-f", "lavfi", "-t", str(seconds), "-i", f"{source}{joint}size={size}:rate={rate}"]
    inputs = "".join(f"[{i}:v]" for i in range(len(sources)))
    args += ["-filter_complex", f"{inputs}concat=n={len(sources)}:v=1[out]", "-map", "[out]"]
    args += ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path
