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
