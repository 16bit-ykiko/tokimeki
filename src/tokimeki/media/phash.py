"""Perceptual hashes of pictures: the DCT hash (pHash), 64 bits that barely change when a
picture is re-encoded or rescaled and change a lot between different pictures."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

SIZE = 32
KEEP = 8


def _dct_matrix(n: int) -> NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


_DCT = _dct_matrix(SIZE)


def phash(image: Image.Image) -> tuple[int, float]:
    """The picture's 64-bit DCT hash, and the spread of its grey thumbnail (standard deviation
    as a share of white): a flat picture (black, white, a plain sky) hashes like any other."""
    grey = np.asarray(
        image.convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low = (_DCT @ grey @ _DCT.T)[:KEEP, :KEEP].flatten()
    bits = low > np.median(low[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big"), float(grey.std() / 255)


def distances(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint8]:
    """Hamming distances between every hash of `a` (rows) and of `b` (columns)."""
    return np.bitwise_count(a[:, None] ^ b[None, :])
