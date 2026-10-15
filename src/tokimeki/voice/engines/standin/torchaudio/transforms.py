import torch

from .functional import _apply, _kernel


class Resample(torch.nn.Module):
    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        **_: object,
    ) -> None:
        super().__init__()
        self.orig_freq, self.new_freq = int(orig_freq), int(new_freq)
        kernel, self.span = _kernel(
            self.orig_freq,
            self.new_freq,
            lowpass_filter_width,
            rolloff,
            torch.device("cpu"),
            torch.float32,
        )
        self.register_buffer("kernel", kernel)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        kernel = self.kernel.to(waveform.dtype)
        return _apply(waveform, self.orig_freq, self.new_freq, kernel, self.span)
