"""Her voice model: fine-tuning GPT-SoVITS on her bank, and speaking new lines with it.

A line is spoken in the manner of a reference clip from her bank: a style names one (set
with `voice style`), and without one the most typical clip of three to ten seconds is used.
The zero-shot baseline runs the same engine on its pretrained weights with the same
reference, so the two can be compared line for line.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tokimeki.library.db import transaction
from tokimeki.library.series import Context
from tokimeki.models.gpu import loaded
from tokimeki.voice.bank import BankLine, bank_lines
from tokimeki.voice.embedding import cosine_distances
from tokimeki.voice.engines import gpt_sovits
from tokimeki.voice.engines.gpt_sovits import Engine, Request, Server, TrainSettings, Weights
from tokimeki.voice.engines.monitor import MemoryPeak
from tokimeki.voice.paths import bank_dir, model_dir
from tokimeki.voice.store import find_speaker, set_style, styles, voice_embeddings

REF_MIN, REF_MAX = 3.0, 10.0
"""GPT-SoVITS takes references of three to ten seconds."""


class VoiceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainReport:
    character: str
    lines: int
    minutes: float
    gpt: str
    sovits: str
    prepare_seconds: float
    train_seconds: float
    peak_vram_mib: int
    """GPU memory the training added at its peak, over what was in use before."""
    settings: dict[str, int]
    trained_at: str


def _model_file(ctx: Context, character: str) -> Path:
    return model_dir(ctx.paths, character, gpt_sovits.NAME) / "model.json"


def train(ctx: Context, character: str, settings: TrainSettings) -> TrainReport:
    """Fine-tune GPT-SoVITS on the character's exported bank (it takes the whole GPU)."""
    engine = Engine.default()
    engine.require()
    lines = bank_lines(ctx.paths, character)
    settings = settings.fitted(len(lines))
    bank = bank_dir(ctx.paths, character)
    root = model_dir(ctx.paths, character, gpt_sovits.NAME)
    work = root / "work"
    listing = bank / "gpt-sovits.list"
    prepared_text = work / "2-name2text.txt"
    if prepared_text.exists() and prepared_text.stat().st_mtime < listing.stat().st_mtime:
        for stale in work.glob("*"):
            if stale.is_file():
                stale.unlink()
    work.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with MemoryPeak() as peak:
        gpt_sovits.prepare(engine, listing, bank / "wavs", work, character)
        prepared = time.monotonic()
        weights = gpt_sovits.train(engine, work, root, character, settings, listing)
    report = TrainReport(
        character,
        len(lines),
        round(sum(x.seconds for x in lines) / 60, 2),
        str(weights.gpt),
        str(weights.sovits),
        round(prepared - start, 1),
        round(time.monotonic() - prepared, 1),
        peak.added,
        {k: int(v) for k, v in asdict(settings).items()},
        datetime.now(UTC).isoformat(),
    )
    _model_file(ctx, character).write_text(json.dumps(asdict(report), ensure_ascii=False, indent=1))
    return report


def trained(ctx: Context, character: str) -> Weights:
    path = _model_file(ctx, character)
    if not path.exists():
        raise VoiceError(f"no trained model of {character}; run `tokimeki voice train` first")
    data: dict[str, object] = json.loads(path.read_text())
    return Weights(Path(str(data["gpt"])), Path(str(data["sovits"])))


@dataclass(frozen=True, slots=True)
class Reference:
    line: int
    wav: Path
    text: str
    style: str


def typical_reference(ctx: Context, lines: list[BankLine]) -> BankLine:
    """The bank line of three to ten seconds closest to her mean voice."""
    fitting = [x for x in lines if REF_MIN <= x.seconds <= REF_MAX] or lines
    embeddings = voice_embeddings(ctx.conn, [x.line for x in fitting])
    centre = embeddings.mean(axis=0, keepdims=True)
    return fitting[int(np.argmin(cosine_distances(embeddings, centre)[:, 0]))]


def reference(ctx: Context, character: str, style: str | None, line: int | None) -> Reference:
    """The bank clip a line is spoken like: a given line, a named style, or the typical one."""
    lines = bank_lines(ctx.paths, character)
    by_line = {x.line: x for x in lines}
    wavs = bank_dir(ctx.paths, character) / "wavs"
    if line is None and style is not None:
        known = styles(ctx.conn, find_speaker(ctx.conn, character).id)
        if style not in known:
            names = ", ".join(sorted(known)) or "none yet"
            raise VoiceError(f"{character} has no style {style!r} (styles: {names})")
        line = known[style]
    if line is not None:
        if line not in by_line:
            raise VoiceError(f"line {line} is not in {character}'s bank")
        x = by_line[line]
        return Reference(x.line, wavs / x.wav, x.text, style or f"line {line}")
    x = typical_reference(ctx, lines)
    return Reference(x.line, wavs / x.wav, x.text, "typical")


def define_style(ctx: Context, character: str, style: str, line: int) -> None:
    """Name a bank line as the reference for a style of hers."""
    if line not in {x.line for x in bank_lines(ctx.paths, character)}:
        raise VoiceError(f"line {line} is not in {character}'s bank; export the bank first")
    with transaction(ctx.conn):
        set_style(ctx.conn, find_speaker(ctx.conn, character).id, style, line)


@dataclass(frozen=True, slots=True)
class Utterance:
    text: str
    out: Path
    ref: Reference
    seed: int = 1234


def speak(ctx: Context, character: str, utterances: list[Utterance], zero_shot: bool) -> float:
    """Synthesise every utterance with one server start; returns the seconds it took."""
    engine = Engine.default()
    weights = gpt_sovits.pretrained_weights(engine) if zero_shot else trained(ctx, character)
    work = model_dir(ctx.paths, character, gpt_sovits.NAME) / (
        "zero-shot" if zero_shot else "serve"
    )
    start = time.monotonic()
    with loaded("GPT-SoVITS", lambda: Server(engine, weights, work)) as server:
        for u in utterances:
            request = Request(u.text, u.ref.wav, u.ref.text, seed=u.seed)
            u.out.parent.mkdir(parents=True, exist_ok=True)
            u.out.write_bytes(server.synthesize(request))
    return time.monotonic() - start
