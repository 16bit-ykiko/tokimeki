import math
from pathlib import Path

from PIL import Image


def load_image(path: Path, long_side: int | None = None) -> Image.Image:
    """Open an image as RGB.

    With `long_side`, a JPEG is decoded at the smallest DCT scale (1/2, 1/4, 1/8) that keeps
    its long side at least that long: much cheaper than decoding in full and resizing.
    """
    with Image.open(path) as image:
        if long_side is not None:
            w, h = image.size
            scale = min(1.0, long_side / max(w, h))
            image.draft("RGB", (math.ceil(w * scale), math.ceil(h * scale)))
        return image.convert("RGB")


def square_around(
    box: tuple[float, float, float, float], width: int, height: int, scale: float, lift: float = 0.0
) -> tuple[int, int, int, int]:
    """A pixel square `scale` times the larger side of a normalised `box`, kept inside the image.

    `lift` moves the square up by that share of its side.
    """
    x0, y0, x1, y1 = box
    side = min(max((x1 - x0) * width, (y1 - y0) * height) * scale, width, height)
    cx = (x0 + x1) / 2 * width
    cy = (y0 + y1) / 2 * height - lift * side
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return round(left), round(top), round(left + side), round(top + side)
