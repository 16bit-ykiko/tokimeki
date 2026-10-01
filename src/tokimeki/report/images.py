from pathlib import Path

from PIL import Image

from tokimeki.library.cast import FaceInShot
from tokimeki.media.images import load_image, square_around
from tokimeki.paths import SeriesPaths

THUMBNAIL_WIDTH = 320
FACE_SIZE = 112
FACE_CROP_SCALE = 1.6
JPEG_QUALITY = 82


def write_thumbnail(paths: SeriesPaths, episode_id: int, frame_index: int, out: Path) -> bool:
    source = paths.frame_path(episode_id, frame_index)
    if not source.exists():
        return False
    image = load_image(source, THUMBNAIL_WIDTH)
    height = round(image.height * THUMBNAIL_WIDTH / image.width)
    image.resize((THUMBNAIL_WIDTH, height), Image.Resampling.LANCZOS).save(
        out, quality=JPEG_QUALITY
    )
    return True


def write_face(paths: SeriesPaths, face: FaceInShot, out: Path) -> bool:
    source = paths.frame_path(face.episode_id, face.frame_index)
    if not source.exists():
        return False
    image = load_image(source)
    b = face.face.box
    box = square_around((b.x0, b.y0, b.x1, b.y1), image.width, image.height, FACE_CROP_SCALE)
    crop = image.crop(box).resize((FACE_SIZE, FACE_SIZE), Image.Resampling.LANCZOS)
    crop.save(out, quality=JPEG_QUALITY)
    return True
