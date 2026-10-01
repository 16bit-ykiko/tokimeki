"""Vocal separation with MDX-Net (UVR's Kim_Vocal_2, ONNX on CUDA), for the vocal line.

Only how much of the mix is voice is needed, so the separated spectrogram is compared with
the mix's directly and never turned back into audio.
"""

import numpy as np
import torch
from numpy.typing import NDArray

from tokimeki.models.gpu import torch_cuda
from tokimeki.models.hub import fetch
from tokimeki.models.ort import OnnxModel

MODEL_REPO = "seanghay/uvr_models"
MODEL_FILE = "Kim_Vocal_2.onnx"
SAMPLE_RATE = 44100
N_FFT = 7680
HOP = 1024
BINS = 3072
FRAMES = 256
CHUNK = HOP * (FRAMES - 1)
FPS = SAMPLE_RATE / HOP
VOCAL_BAND = (200.0, 5000.0)
BATCH_SIZE = 1


def mdx_input(chunks: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """`(n, 2, CHUNK)` stereo chunks as the model's `(n, 4, BINS, FRAMES)` input:
    left real, left imaginary, right real, right imaginary."""
    n = len(chunks)
    spec = torch.stft(
        chunks.reshape(-1, CHUNK), N_FFT, HOP, window=window, center=True, return_complex=True
    )
    parts = torch.view_as_real(spec).permute(0, 3, 1, 2)
    return parts.reshape(n, 4, N_FFT // 2 + 1, FRAMES)[:, :, :BINS].contiguous()


def band_mask(device: torch.device) -> torch.Tensor:
    freqs = torch.arange(BINS, device=device) * SAMPLE_RATE / N_FFT
    low, high = VOCAL_BAND
    return (freqs >= low) & (freqs <= high)


class VocalSeparator:
    def __init__(self) -> None:
        self._device = torch_cuda()
        self._model = OnnxModel(fetch(MODEL_REPO, MODEL_FILE))
        self._window = torch.hann_window(N_FFT, device=self._device)
        self._band = band_mask(self._device)

    def vocal_share(self, stereo: NDArray[np.float32]) -> NDArray[np.float32]:
        """Per STFT frame (`FPS` a second) of 44.1 kHz stereo `(2, samples)` audio, the share
        of the vocal band's energy that is voice."""
        x = torch.tensor(stereo, device=self._device)
        length = x.shape[1]
        x = torch.nn.functional.pad(x, (0, (-length) % CHUNK))
        chunks = x.reshape(2, -1, CHUNK).permute(1, 0, 2)
        shares: list[torch.Tensor] = []
        name = self._model.input_names[0]
        with torch.inference_mode():
            for start in range(0, len(chunks), BATCH_SIZE):
                mix = mdx_input(chunks[start : start + BATCH_SIZE], self._window)
                voice = torch.tensor(
                    self._model.run({name: mix.cpu().numpy()})[0], device=self._device
                )
                mix_energy = (mix**2).sum(dim=1)[:, self._band].sum(dim=1)
                voice_energy = (voice**2).sum(dim=1)[:, self._band].sum(dim=1)
                shares.append((voice_energy / (mix_energy + 1e-9)).reshape(-1))
        frames = int(np.ceil(length / HOP))
        return torch.cat(shares)[:frames].clamp(0, 1).cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._model.close()
