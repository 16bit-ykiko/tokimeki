"""How much the picture changes from frame to frame, on the GPU (PyTorch, no model).

Two measures per frame: the mean absolute luma difference from the frame before, which
sees slow pans, and the share of pixels that changed by more than `CHANGED`, which is 0 on
a held drawing and jumps when the next one comes (anime is drawn on twos and threes).
"""

import numpy as np
import torch
from numpy.typing import NDArray

from tokimeki.models.gpu import torch_cuda

CHANGED = 0.04


class MotionMeter:
    def __init__(self) -> None:
        self._device = torch_cuda()

    def differences(
        self, frames: NDArray[np.uint8], previous: NDArray[np.uint8] | None
    ) -> NDArray[np.float32]:
        """`(n, 2)`: per `(n, h, w)` frame, its mean absolute luma difference (0-1) from the
        frame before it (`previous` for the first; 0 when there is none) and the share of
        its pixels that changed by more than `CHANGED`."""
        with torch.inference_mode():
            x = torch.tensor(frames, device=self._device, dtype=torch.float32) / 255.0
            before = (
                torch.tensor(previous, device=self._device, dtype=torch.float32)[None] / 255.0
                if previous is not None
                else x[:1]
            )
            diff = (x - torch.cat([before, x[:-1]])).abs()
            mean = diff.mean(dim=(1, 2))
            changed = (diff > CHANGED).float().mean(dim=(1, 2))
            return torch.stack([mean, changed], dim=1).cpu().numpy().astype(np.float32)

    def close(self) -> None:
        torch.cuda.empty_cache()
