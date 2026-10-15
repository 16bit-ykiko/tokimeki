"""A character's voice bank: her clean lines as WAV files with their Japanese text, listed in
the formats TTS trainers read, and how much there is.

`gpt-sovits.list` is `absolute wav path|speaker|ja|text` (GPT-SoVITS); `esd.list` is
`wav name|speaker|JP|text` (Style-Bert-VITS2, which takes the wavs from its own folder).
The bank stays in the series' data directory and is never shared.
"""

import json
import re
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from tokimeki.anime.stems import stem_path
from tokimeki.library.episodes import get_episode
from tokimeki.library.paths import SeriesPaths
from tokimeki.library.series import Context
from tokimeki.media.audio import decode_audio, encode_audio
from tokimeki.voice.paths import bank_dir
from tokimeki.voice.store import LineVoice, clean_voices, find_speaker, transcripts

BANK_RATE = 44100
MIN_SECONDS = 1.0
MAX_SECONDS = 12.0
"""GPT-SoVITS trains on lines of a few seconds; much longer ones are better split first."""
MIN_LETTERS = 2
FADE = 0.01
_LETTERS = re.compile(r"[぀-ヿ㐀-鿿A-Za-z0-9]")


class BankError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BankLine:
    line: int
    episode: int
    wav: str
    seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class BankStats:
    character: str
    lines: int
    minutes: float
    episodes: dict[int, float]
    """Minutes per episode id."""
    skipped: dict[str, int]
    directory: str


def letters(text: str) -> int:
    return len(_LETTERS.findall(text))


def faded(samples: NDArray[np.float32], rate: int) -> NDArray[np.float32]:
    n = min(round(FADE * rate), len(samples) // 2)
    out = samples.astype(np.float32, copy=True)
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def pick(voices: list[LineVoice], texts: dict[int, str]) -> tuple[list[LineVoice], dict[str, int]]:
    """The voices fit for training, and how many were left out for what."""
    skipped = {"too short": 0, "too long": 0, "no text": 0}
    kept: list[LineVoice] = []
    for v in voices:
        text = texts.get(v.line_id, "")
        if v.duration < MIN_SECONDS:
            skipped["too short"] += 1
        elif v.duration > MAX_SECONDS:
            skipped["too long"] += 1
        elif letters(text) < MIN_LETTERS:
            skipped["no text"] += 1
        else:
            kept.append(v)
    return kept, skipped


def export_bank(ctx: Context, character: str) -> BankStats:
    """Write the bank of a named speaker; wavs of lines no longer in it are removed."""
    speaker = find_speaker(ctx.conn, character)
    if speaker.name is None:
        raise BankError(f"speaker #{speaker.id} has no name; name it before exporting a bank")
    texts = transcripts(ctx.conn)
    voices, skipped = pick(clean_voices(ctx.conn, speaker.id), texts)
    if not voices:
        raise BankError(f"{speaker.name} has no clean, transcribed lines to export")
    root = bank_dir(ctx.paths, speaker.name)
    wavs = root / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    lines: list[BankLine] = []
    for episode_id in sorted({v.episode_id for v in voices}):
        stem = decode_audio(stem_path(ctx.paths, get_episode(ctx.conn, episode_id).id), BANK_RATE)
        for v in (v for v in voices if v.episode_id == episode_id):
            name = f"{episode_id:03d}-{v.line_id}.wav"
            part = stem[round(v.start * BANK_RATE) : round(v.end * BANK_RATE)]
            encode_audio(faded(part, BANK_RATE)[None], BANK_RATE, wavs / name)
            text = " ".join(texts[v.line_id].split())
            lines.append(BankLine(v.line_id, episode_id, name, round(v.duration, 3), text))
    wanted = {line.wav for line in lines}
    for stale in wavs.glob("*.wav"):
        if stale.name not in wanted:
            stale.unlink()
    who = speaker.name
    (root / "gpt-sovits.list").write_text(
        "".join(f"{(wavs / x.wav).resolve()}|{who}|ja|{x.text}\n" for x in lines)
    )
    (root / "esd.list").write_text("".join(f"{x.wav}|{who}|JP|{x.text}\n" for x in lines))
    episodes: dict[int, float] = {}
    for x in lines:
        episodes[x.episode] = episodes.get(x.episode, 0.0) + x.seconds / 60
    stats = BankStats(
        who,
        len(lines),
        round(sum(x.seconds for x in lines) / 60, 2),
        {e: round(m, 2) for e, m in episodes.items()},
        skipped,
        str(root),
    )
    (root / "bank.json").write_text(
        json.dumps(
            {**asdict(stats), "items": [asdict(x) for x in lines]}, ensure_ascii=False, indent=1
        )
    )
    return stats


def bank_lines(paths: SeriesPaths, character: str) -> list[BankLine]:
    """The lines of an exported bank."""
    path = bank_dir(paths, character) / "bank.json"
    if not path.exists():
        raise BankError(f"no bank for {character}; export it first (`tokimeki voice bank`)")
    data: dict[str, list[dict[str, str | int | float]]] = json.loads(path.read_text())
    return [
        BankLine(
            int(x["line"]), int(x["episode"]), str(x["wav"]), float(x["seconds"]), str(x["text"])
        )
        for x in data["items"]
    ]
