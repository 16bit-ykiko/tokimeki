"""A song sung in a character's voice (an "AI cover"), for listening on this machine only.

The song's lead vocal is separated from the mix, sung again in her voice by Seed-VC (fine-tuned
on her bank, or zero-shot from it) and mixed back over the accompaniment: the release's own
instrumental when there is one (aligned to the mix), otherwise what separation left. With the
`audio-separator` engine installed, RoFormer models take the lead apart from the backing vocals
and its reverb, and only the dry lead is converted (the backing vocals stay, quieter); without
it, MDX-Net's whole vocal is.
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
from tokimeki.voice.engines import seed_vc, separator
from tokimeki.voice.engines.seed_vc import Conversion, Engine
from tokimeki.voice.paths import bank_dir, character_dir, model_dir

REFERENCE_SECONDS = 25.0
"""Seed-VC conditions on at most about 25 s of reference."""
REFERENCE_GAP = 0.15
REFERENCE_MAX = 8.0
CLARITY_RATE = 16000
PITCH_RANGE = (80.0, 650.0)
VOICED = 0.45
"""Normalised autocorrelation at the pitch lag above which a frame counts as voiced."""
ALIGN_RATE = 1000
MAX_SHIFT = 3.0
"""Seconds the instrumental may be offset from the mix on the disc."""
PEAK = 0.98
BACKING = 0.6
"""Gain of the original backing vocals under her lead."""


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
    separation: str
    reference_lines: list[int]
    instrumental_offset: float | None


def clarity(samples: NDArray[np.float32], rate: int) -> float:
    """Mean harmonics-to-noise ratio (dB) of the voiced 40 ms frames, from each frame's peak
    normalised autocorrelation over the lags of a voice's pitch; -inf when nothing is voiced.

    Seed-VC copies the reference's texture: a reference of shouted or breathy lines made the
    converted singing noisier and less like her, one of her clearest calm lines made it as
    clean as the source at no cost in likeness (docs/voices.md)."""
    size, hop = round(0.04 * rate), round(0.01 * rate)
    if len(samples) < size:
        return float("-inf")
    starts = np.arange(0, len(samples) - size + 1, hop)
    frames = samples[starts[:, None] + np.arange(size)].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    energy = np.sum(frames**2, axis=1)
    frames = frames[energy > 0.01 * energy.max()]
    spectrum = np.fft.rfft(frames, 2 * size)
    corr = np.fft.irfft(np.abs(spectrum) ** 2)[:, :size]
    lo, hi = round(rate / PITCH_RANGE[1]), round(rate / PITCH_RANGE[0])
    unbiased = corr[:, lo:hi] * (size / (size - np.arange(lo, hi)))
    peak = unbiased.max(axis=1) / np.maximum(corr[:, 0], 1e-12)
    voiced = np.clip(peak[peak > VOICED], VOICED, 0.999)
    if len(voiced) == 0:
        return float("-inf")
    return float(np.mean(10 * np.log10(voiced / (1 - voiced))))


def reference_clip(ctx: Context, character: str, out: Path) -> list[int]:
    """Her clearest bank lines (see `clarity`) joined into one reference of about
    `REFERENCE_SECONDS`, clearest first."""
    wavs = bank_dir(ctx.paths, character) / "wavs"
    fitting = [x for x in bank_lines(ctx.paths, character) if x.seconds <= REFERENCE_MAX]
    scored = [(clarity(decode_audio(wavs / x.wav, CLARITY_RATE), CLARITY_RATE), x) for x in fitting]
    lines = [x for _, x in sorted(scored, key=lambda pair: -pair[0])]
    parts: list[NDArray[np.float32]] = []
    used: list[int] = []
    total = 0.0
    gap = np.zeros(round(REFERENCE_GAP * SAMPLE_RATE), dtype=np.float32)
    for x in lines:
        if total + x.seconds > REFERENCE_SECONDS:
            continue
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


def fitted(track: NDArray[np.float32], length: int) -> NDArray[np.float32]:
    """`track` (channels first) cut or padded with silence to `length` samples."""
    if track.shape[-1] >= length:
        return track[..., :length]
    return np.pad(track, [(0, 0)] * (track.ndim - 1) + [(0, length - track.shape[-1])])


def rms(samples: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


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
    length = mixture.shape[1]
    roformer = separator.Engine.default()
    if roformer.installed():
        encode_audio(mixture, SAMPLE_RATE, root / "song.wav")
        with loaded("RoFormer", _Idle):
            stems = separator.separate(roformer, root / "song.wav", root / "stems")
        (root / "song.wav").unlink()
        dry = fitted(decode_audio(stems.lead_dry, SAMPLE_RATE), length)
        loudness = rms(decode_audio(stems.lead, SAMPLE_RATE))
        rest = fitted(decode_audio(stems.instrumental, SAMPLE_RATE, channels=2), length)
        backing = fitted(decode_audio(stems.backing, SAMPLE_RATE, channels=2), length)
    else:
        with loaded("MDX-Net", VocalSeparator) as mdx:
            vocal = mdx.separate(mixture)
        dry = vocal.mean(axis=0)
        loudness = rms(dry)
        rest = mixture - vocal
        backing = np.zeros_like(mixture)
    lag: float | None = None
    if instrumental is not None:
        official = instrumental.decode()
        lag = offset(rest, official)
        rest = aligned(official, lag, length)
    accompaniment = (rest + BACKING * backing).astype(np.float32)
    encode_audio(dry[None], SAMPLE_RATE, root / "original-vocal.wav")
    used = reference_clip(ctx, character, root / "reference.wav")
    checkpoint = model_dir(ctx.paths, character, seed_vc.NAME) / "ft_model.pth"
    model = checkpoint if fine_tuned and checkpoint.exists() else None
    job = Conversion(root / "original-vocal.wav", root / "reference.wav", semitones)
    with loaded("Seed-VC", _Idle):
        sung = seed_vc.convert(engine, job, model, root / "vocal.wav")
    converted = decode_audio(sung, SAMPLE_RATE)
    out = root / "cover.flac"
    encode_audio(mix(converted, accompaniment, loudness), SAMPLE_RATE, out)
    report = CoverReport(
        character,
        str(out),
        str(sung),
        round(mixture.shape[1] / SAMPLE_RATE, 1),
        "fine-tuned" if model else "zero-shot",
        "roformer" if roformer.installed() else "mdx-net",
        used,
        lag,
    )
    (root / "cover.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=1))
    return report


class _Idle:
    """Holds the in-process GPU slot while an engine's subprocess uses the card."""

    def close(self) -> None:
        pass
