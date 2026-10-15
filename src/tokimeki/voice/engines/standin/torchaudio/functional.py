"""Band-limited resampling as torchaudio does it (windowed sinc, Hann window), in plain PyTorch."""

import math

import torch


def _kernel(
    orig: int, new: int, width: int, rolloff: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    base = min(orig, new) * rolloff
    span = math.ceil(width * orig / base)
    idx = torch.arange(-span, span + orig, device=device, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, device=device, dtype=torch.float64)[:, None, None] / new + idx
    t = (t * base).clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    scale = base / orig
    kernels = torch.where(
        t == 0, torch.tensor(1.0, device=device, dtype=torch.float64), t.sin() / t
    )
    return (kernels * window * scale).to(dtype), span


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    **_: object,
) -> torch.Tensor:
    if orig_freq == new_freq:
        return waveform
    kernel, span = _kernel(
        int(orig_freq),
        int(new_freq),
        lowpass_filter_width,
        rolloff,
        waveform.device,
        waveform.dtype,
    )
    return _apply(waveform, int(orig_freq), int(new_freq), kernel, span)


def _apply(
    waveform: torch.Tensor, orig: int, new: int, kernel: torch.Tensor, span: int
) -> torch.Tensor:
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    shape = waveform.shape
    flat = waveform.reshape(-1, shape[-1])
    length = flat.shape[-1]
    padded = torch.nn.functional.pad(flat, (span, span + orig))
    out = torch.nn.functional.conv1d(padded[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(flat.shape[0], -1)
    target = math.ceil(new * length / orig)
    return out[..., :target].reshape(shape[:-1] + (target,))
