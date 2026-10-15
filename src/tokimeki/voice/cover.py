"""A song sung in a character's voice (an "AI cover"), for listening on this machine only.

The song's vocal is separated from the mix with MDX-Net, sung again in her voice by Seed-VC
(fine-tuned on her bank, or zero-shot from it), and mixed back over the accompaniment: the
release's own instrumental when there is one (aligned to the mix), otherwise what MDX-Net left.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.series import Context
from tokimeki.media.audio import decode_audio, encode_audio
from tokimeki.models.gpu import loaded
from tokimeki.models.separation import SAMPLE_RATE, VocalSeparator
from tokimeki.voice.bank import bank_lines
from tokimeki.voice.engines import seed_vc
from tokimeki.voice.engines.seed_vc import Conversion, Engine
from tokimeki.voice.paths import bank_dir, character_dir, model_dir

REFERENCE_SECONDS = 25.0
"""Seed-VC conditions on at most about 25 s of reference."""
REFERENCE_GAP = 0.15
ALIGN_RATE = 1000
MAX_SHIFT = 3.0
"""Seconds the instrumental may be offset from the mix on the disc."""
PEAK = 0.98


@dataclass(frozen=True, slots=True)
class Source:
    path: Path
    start: float | None = None
    duration: float | None = None

    def decode(self) -> NDArray[np.float32]:
        return decode_audio(
            self.path, SAMPLE_RATE, start=self.start, duration=self.duration, channels=2
        )


@dataclass(frozen=True, slots=True)
class CoverReport:
    character: str
    output: str
    vocal: str
    seconds: float
    model: str
    reference_lines: list[int]
    instrumental_offset: float | None


def reference_clip(ctx: Context, character: str, out: Path) -> list[int]:
    """Her clearest bank lines joined into one reference of about `REFERENCE_SECONDS`."""
    lines = sorted(bank_lines(ctx.paths, character), key=lambda x: -x.seconds)
    wavs = bank_dir(ctx.paths, character) / "wavs"
    parts: list[NDArray[np.float32]] = []
    used: list[int] = []
    total = 0.0
    gap = np.zeros(round(REFERENCE_GAP * SAMPLE_RATE), dtype=np.float32)
    for x in lines:
        if x.seconds > 8.0:
            continue
        if total + x.seconds > REFERENCE_SECONDS:
            break
        parts += [decode_audio(wavs / x.wav, SAMPLE_RATE), gap]
        used.append(x.line)
        total += x.seconds + REFERENCE_GAP
    encode_audio(np.concatenate(parts)[None], SAMPLE_RATE, out)
    return used


def envelope(stereo: NDArray[np.float32]) -> NDArray[np.float64]:
    mono = np.abs(stereo.mean(axis=0)).astype(np.float64)
    hop = SAMPLE_RATE // ALIGN_RATE
    frames = len(mono) // hop
    env = mono[: frames * hop].reshape(frames, hop).mean(axis=1)
    return (env - env.mean()) / (env.std() + 1e-9)


def offset(reference: NDArray[np.float32], other: NDArray[np.float32]) -> float:
    """Seconds `other` lags `reference` (positive: it starts later), by envelope correlation."""
    a, b = envelope(reference), envelope(other)
    size = 1 << (len(a) + len(b) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    span = round(MAX_SHIFT * ALIGN_RATE)
    lags = np.arange(-span, span + 1)
    return int(lags[int(np.argmax(corr[lags % size]))]) / ALIGN_RATE


def aligned(track: NDArray[np.float32], lag: float, length: int) -> NDArray[np.float32]:
    shift = round(lag * SAMPLE_RATE)
    out = np.zeros((track.shape[0], length), dtype=np.float32)
    part = track[:, shift:] if shift >= 0 else np.pad(track, ((0, 0), (-shift, 0)))
    out[:, : min(length, part.shape[1])] = part[:, :length]
    return out


def mix(
    vocal: NDArray[np.float32], accompaniment: NDArray[np.float32], loudness: float
) -> NDArray[np.float32]:
    """`vocal` (mono) at the RMS `loudness`, over the stereo accompaniment, kept off clipping."""
    n = min(len(vocal), accompaniment.shape[1])
    rms = float(np.sqrt(np.mean(vocal[:n].astype(np.float64) ** 2))) + 1e-9
    voiced = vocal[:n] * (loudness / rms)
    out = accompaniment[:, :n] + voiced[None]
    peak = float(np.abs(out).max())
    return (out * (PEAK / peak) if peak > PEAK else out).astype(np.float32)


def cover(
    ctx: Context,
    character: str,
    song: Source,
    instrumental: Source | None,
    name: str,
    semitones: int = 0,
    fine_tuned: bool = True,
) -> CoverReport:
    """Write `<character>/covers/<name>/cover.flac` (and the converted vocal alone)."""
    engine = Engine.default()
    engine.require()
    root = character_dir(ctx.paths, character) / "covers" / name
    root.mkdir(parents=True, exist_ok=True)
    mixture = song.decode()
    with loaded("MDX-Net", VocalSeparator) as separator:
        vocal = separator.separate(mixture)
    lag: float | None = None
    if instrumental is not None:
        backing = instrumental.decode()
        lag = offset(mixture - vocal, backing)
        accompaniment = aligned(backing, lag, mixture.shape[1])
    else:
        accompaniment = mixture - vocal
    dry = vocal.mean(axis=0)
    encode_audio(dry[None], SAMPLE_RATE, root / "original-vocal.wav")
    used = reference_clip(ctx, character, root / "reference.wav")
    checkpoint = model_dir(ctx.paths, character, seed_vc.NAME) / "ft_model.pth"
    model = checkpoint if fine_tuned and checkpoint.exists() else None
    job = Conversion(root / "original-vocal.wav", root / "reference.wav", semitones)
    with loaded("Seed-VC", _Idle):
        sung = seed_vc.convert(engine, job, model, root / "vocal.wav")
    converted = decode_audio(sung, SAMPLE_RATE)
    loudness = float(np.sqrt(np.mean(dry.astype(np.float64) ** 2)))
    out = root / "cover.flac"
    encode_audio(mix(converted, accompaniment, loudness), SAMPLE_RATE, out)
    report = CoverReport(
        character,
        str(out),
        str(sung),
        round(mixture.shape[1] / SAMPLE_RATE, 1),
        "fine-tuned" if model else "zero-shot",
        used,
        lag,
    )
    (root / "cover.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=1))
    return report


class _Idle:
    """Holds the in-process GPU slot while a Seed-VC subprocess uses the card."""

    def close(self) -> None:
        pass
