"""TransNetV2 shot-boundary detection (the `transnetv2-pytorch` port and its bundled weights)."""

import importlib
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from tokimeki.models import CPU_THREADS
from tokimeki.models.gpu import torch_cuda

THRESHOLD = 0.5
WINDOW = 100
CONTEXT = 25
STEP = WINDOW - 2 * CONTEXT
WINDOWS_PER_BATCH = 16


def _load_network(device: torch.device) -> torch.nn.Module:
    module = importlib.import_module("transnetv2_pytorch")
    factory = cast(Callable[..., torch.nn.Module], module.TransNetV2)
    network = factory(device=device.type)
    # Its constructor switches the whole process to deterministic kernels; nothing here needs that.
    torch.use_deterministic_algorithms(False)
    return network.eval()


class TransNet:
    def __init__(self) -> None:
        self._device = torch_cuda()
        torch.set_num_threads(CPU_THREADS)
        self._network: torch.nn.Module | None = _load_network(self._device)

    def predict(self, frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Per-frame probability that a shot transition happens at that frame.

        `frames` is `(n, 27, 48, 3)` RGB. Windows of 100 frames advance by 50 and keep their
        middle 50 predictions, with the ends padded by repeating the first and last frame.
        """
        if self._network is None:
            raise RuntimeError("TransNet is closed")
        n = len(frames)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        video = torch.tensor(frames, dtype=torch.uint8, device=self._device)
        tail = CONTEXT + STEP - (n % STEP or STEP)
        padded = torch.cat(
            [video[:1].expand(CONTEXT, -1, -1, -1), video, video[-1:].expand(tail, -1, -1, -1)]
        )
        windows = padded.unfold(0, WINDOW, STEP).permute(0, 4, 1, 2, 3)
        out: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(windows), WINDOWS_PER_BATCH):
                batch = windows[start : start + WINDOWS_PER_BATCH].contiguous()
                logits = cast(tuple[torch.Tensor, object], self._network(batch))[0]
                out.append(torch.sigmoid(logits[:, CONTEXT : CONTEXT + STEP, 0]).reshape(-1))
        return torch.cat(out)[:n].float().cpu().numpy()

    def close(self) -> None:
        self._network = None


def shots_from_predictions(
    predictions: NDArray[np.float32], threshold: float = THRESHOLD
) -> list[tuple[int, int]]:
    """Shots as `[start, end)` frame spans.

    A shot ends on the first frame of a transition (for a hard cut, its own last frame) and
    the next starts after the transition; frames inside a gradual transition belong to no shot.
    """
    is_transition = predictions > threshold
    shots: list[tuple[int, int]] = []
    start: int | None = 0 if len(predictions) and not is_transition[0] else None
    for i in range(1, len(predictions)):
        if start is None and not is_transition[i] and is_transition[i - 1]:
            start = i
        elif start is not None and is_transition[i]:
            shots.append((start, i + 1))
            start = None
    if start is not None:
        shots.append((start, len(predictions)))
    if not shots and len(predictions):
        shots.append((0, len(predictions)))
    return shots
