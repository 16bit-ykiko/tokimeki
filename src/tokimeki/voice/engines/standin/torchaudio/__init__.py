"""The few torchaudio calls GPT-SoVITS and Seed-VC make (load, save, resampling, Kaldi fbank),
for an environment whose PyTorch has no torchaudio build. Written for tokimeki, not torchaudio;
compliance/kaldi.py is torchaudio's (BSD-2-Clause), as GPT-SoVITS bundles it."""

import soundfile
import torch

from . import compliance, functional, transforms

__version__ = "0.0+tokimeki"


def load(path, *_, **__):
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def save(path, src, sample_rate, *_, **__):
    soundfile.write(str(path), src.detach().cpu().float().numpy().T, int(sample_rate))
