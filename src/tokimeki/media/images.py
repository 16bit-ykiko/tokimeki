from pathlib import Path

from PIL import Image


def load_image(path: Path, at_least: tuple[int, int] | None = None) -> Image.Image:
    """Open an image as RGB; a JPEG is decoded at a reduced scale no smaller than `at_least`."""
    with Image.open(path) as image:
        if at_least is not None:
            image.draft("RGB", at_least)
        return image.convert("RGB")
