"""Beat This! (CPJKU) beat and downbeat tracking on CUDA.

Only the network comes from Beat This! (vendored in `models/beat_this`). Its audio loading
and mel front end need torchaudio, which conda-forge does not build for this PyTorch, so the log-mel
spectrogram (the same as torchaudio's `MelSpectrogram` with the model's settings), the
chunked inference and the "minimal" peak picking are reimplemented here.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from tokimeki.models import CPU_THREADS
from tokimeki.models.gpu import torch_cuda

CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 441
FPS = SAMPLE_RATE / HOP
F_MIN, F_MAX = 30.0, 11000.0
N_MELS = 128
LOG_MULTIPLIER = 1000.0
CHUNK = 1500
BORDER = 6


@dataclass(frozen=True, slots=True)
class Beats:
    beats: NDArray[np.float64]
    downbeats: NDArray[np.float64]


def _hz_to_mel(hz: NDArray[np.float64]) -> NDArray[np.float64]:
    linear = hz / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(hz, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(hz >= 1000.0, log, linear)


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    linear = mel * (200.0 / 3)
    log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel - 15.0))
    return np.where(mel >= 15.0, log, linear)


def mel_filters() -> NDArray[np.float64]:
    """Triangular Slaney-scale filters, `(N_FFT // 2 + 1, N_MELS)`, unnormalised."""
    freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    edges = _mel_to_hz(
        np.linspace(_hz_to_mel(np.array(F_MIN)), _hz_to_mel(np.array(F_MAX)), N_MELS + 2)
    )
    widths = np.diff(edges)
    slopes = edges[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    return np.maximum(0.0, np.minimum(down, up))


def mel_centres() -> NDArray[np.float64]:
    """Centre frequency (Hz) of each mel bin."""
    mels = np.linspace(_hz_to_mel(np.array(F_MIN)), _hz_to_mel(np.array(F_MAX)), N_MELS + 2)
    return _mel_to_hz(mels)[1:-1]


def log_mel(audio: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """`(frames, N_MELS)` log-magnitude mel spectrogram of 22.05 kHz mono audio."""
    window = torch.hann_window(N_FFT, device=audio.device)
    spectrum = torch.stft(
        audio,
        N_FFT,
        HOP,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=True,
        return_complex=True,
    )
    return torch.log1p(LOG_MULTIPLIER * (filters.T @ spectrum.abs()).T)


def chunk_starts(length: int) -> list[int]:
    """Where each model window starts; windows overlap by `BORDER` frames on either side."""
    starts = list(range(-BORDER, length - BORDER, CHUNK - 2 * BORDER))
    if length > CHUNK - 2 * BORDER:
        starts[-1] = length - (CHUNK - BORDER)
    return starts


def pick_peaks(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    """Frames (as seconds) that are the maximum within ±3 frames and above probability 0.5;
    runs of adjacent peaks become their mean."""
    padded = np.pad(logits, 3, constant_values=-1000.0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 7)
    frames = np.flatnonzero((logits == windows.max(axis=1)) & (logits > 0))
    merged: list[float] = []
    count = 0
    for frame in frames:
        if merged and frame - merged[-1] <= 1:
            count += 1
            merged[-1] += (frame - merged[-1]) / count
        else:
            merged.append(float(frame))
            count = 1
    return np.array(merged, dtype=np.float64) / FPS


def snap_downbeats(
    beats: NDArray[np.float64], downbeats: NDArray[np.float64]
) -> NDArray[np.float64]:
    if not len(beats):
        return downbeats
    nearest = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
    return np.unique(beats[nearest])


def _network(device: torch.device) -> torch.nn.Module:
    module = importlib.import_module("tokimeki.models.beat_this.beat_tracker")
    factory = cast(Callable[..., torch.nn.Module], module.BeatThis)
    checkpoint = cast(
        dict[str, dict[str, object]],
        torch.hub.load_state_dict_from_url(
            CHECKPOINT_URL,
            file_name="beat_this-final0.ckpt",
            map_location=device,
            weights_only=True,
        ),
    )
    params = {
        k: v
        for k, v in checkpoint["hyper_parameters"].items()
        if k in {"spect_dim", "transformer_dim", "ff_mult", "n_layers", "head_dim", "stem_dim",
                 "dropout", "sum_head", "partial_transformers"}
    }  # fmt: skip
    network = factory(**params)
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    network.load_state_dict(state)
    return network.to(device).eval()


class BeatTracker:
    def __init__(self) -> None:
        self._device = torch_cuda()
        torch.set_num_threads(CPU_THREADS)
        self._network: torch.nn.Module | None = _network(self._device)
        self._filters = torch.tensor(mel_filters(), dtype=torch.float32, device=self._device)

    def spectrogram(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """The model's log-mel input, `(frames, N_MELS)` at `FPS`, for structure analysis."""
        with torch.inference_mode():
            return log_mel(torch.tensor(audio, device=self._device), self._filters).cpu().numpy()

    def logits(self, audio: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Beat and downbeat logits per frame (`FPS` a second) of 22.05 kHz mono audio."""
        if self._network is None:
            raise RuntimeError("the beat tracker is closed")
        with torch.inference_mode():
            spect = log_mel(torch.tensor(audio, device=self._device), self._filters)
            length = len(spect)
            beat = torch.full((length,), -1000.0, device=self._device)
            downbeat = torch.full((length,), -1000.0, device=self._device)
            for start in reversed(chunk_starts(length)):
                piece = spect[max(start, 0) : min(start + CHUNK, length)]
                left = max(0, -start)
                right = max(0, min(BORDER, start + CHUNK - length))
                piece = torch.nn.functional.pad(piece, (0, 0, left, right))
                out = cast(dict[str, torch.Tensor], self._network(piece.unsqueeze(0)))
                keep = slice(start + BORDER, start + CHUNK - BORDER)
                beat[keep] = out["beat"][0][BORDER:-BORDER][: len(beat[keep])]
                downbeat[keep] = out["downbeat"][0][BORDER:-BORDER][: len(downbeat[keep])]
        return beat.cpu().numpy(), downbeat.cpu().numpy()

    def track(self, audio: NDArray[np.float32]) -> Beats:
        beat, downbeat = self.logits(audio)
        beats = pick_peaks(beat)
        return Beats(beats, snap_downbeats(beats, pick_peaks(downbeat)))

    def close(self) -> None:
        self._network = None
