"""Analysing any song once: beats and bars (Beat This! on the GPU), vocal line, sections,
an excerpt and the slots it suggests.

A song is an audio file (a track of a `.cue` image is a convenience). The vocal line, which
places the intro, the sung sections and the outro, comes from the mix itself by vocal
separation (MDX-Net on the GPU); an LRC lyrics file or an instrumental version, when given,
are used instead. Analyses live in a song store shared by every series
(`$TOKIMEKI_HOME/songs`, by default `~/.local/share/tokimeki/songs`), one per song id.
"""

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tokimeki.media.audio import audio_streams, decode_audio, main_audio
from tokimeki.media.cue import read_cue
from tokimeki.models.beats import FPS, SAMPLE_RATE, Beats, BeatTracker, mel_centres
from tokimeki.models.gpu import loaded
from tokimeki.models.separation import FPS as SEPARATION_FPS
from tokimeki.models.separation import MODEL_FILE as SEPARATION_MODEL
from tokimeki.models.separation import SAMPLE_RATE as SEPARATION_RATE
from tokimeki.models.separation import VocalSeparator
from tokimeki.song import bounds, structure
from tokimeki.song.lyrics import LyricLine, LyricSource, read_lyrics
from tokimeki.song.slots import Excerpt, Slot, excerpt_between, first_chorus, make_slots
from tokimeki.song.structure import Section

ANALYSIS_VERSION = 2
UNLIMITED = 1_000_000
BOUNDS_RATE = 16000

type Json = dict[str, object]


class SongError(RuntimeError):
    pass


@dataclass(frozen=True)
class SongSource:
    """Where the song's audio is: a whole file, or a stretch of one (a CD image track)."""

    path: Path
    offset: float
    duration: float
    title: str
    track: int | None = None
    stream: int | None = None
    """The audio stream of a file with several (an episode's main track, not commentary)."""


@dataclass
class SongAnalysis:
    id: str
    title: str
    source: SongSource
    bpm: float
    beats: list[float]
    bars: list[float]
    energy: list[float]
    """Loudness per bar."""
    vocal: list[float] | None
    """Share of each bar that is sung; None when no vocal line was available."""
    vocal_source: str
    """"lyrics", "instrumental", "separation" or "none"."""
    sections: list[Section]
    lyrics: list[LyricLine]
    excerpt: tuple[float, float] = (0.0, 0.0)
    slots: list[Slot] = field(default_factory=list[Slot])
    inputs: dict[str, str] = field(default_factory=dict[str, str])
    """What the beats and sections were computed from, to know when to redo them."""

    def to_json(self) -> str:
        data = asdict(self)
        data["source"]["path"] = str(self.source.path)
        data["version"] = ANALYSIS_VERSION
        return json.dumps(data, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> "SongAnalysis":
        d = cast(Json, json.loads(text))
        if d.get("version") != ANALYSIS_VERSION:
            raise SongError("the stored analysis is from another version")
        src = cast(Json, d["source"])
        excerpt = cast(list[float], d["excerpt"])
        return SongAnalysis(
            id=str(d["id"]),
            title=str(d["title"]),
            source=SongSource(
                Path(str(src["path"])),
                _float(src["offset"]),
                _float(src["duration"]),
                str(src["title"]),
                cast(int | None, src.get("track")),
                cast(int | None, src.get("stream")),
            ),
            bpm=_float(d["bpm"]),
            beats=cast(list[float], d["beats"]),
            bars=cast(list[float], d["bars"]),
            energy=cast(list[float], d["energy"]),
            vocal=cast(list[float] | None, d["vocal"]),
            vocal_source=str(d["vocal_source"]),
            sections=[_section(cast(Json, s)) for s in cast(list[object], d["sections"])],
            lyrics=[_lyric(cast(Json, x)) for x in cast(list[object], d["lyrics"])],
            excerpt=(float(excerpt[0]), float(excerpt[1])),
            slots=[_slot(cast(Json, x)) for x in cast(list[object], d["slots"])],
            inputs=cast(dict[str, str], d.get("inputs", {})),
        )


def _float(value: object) -> float:
    return float(cast(float, value))


def _int(value: object) -> int:
    return int(cast(int, value))


def _section(d: Json) -> Section:
    return Section(
        str(d["label"]),
        _int(d["group"]),
        _int(d["first_bar"]),
        _int(d["end_bar"]),
        _float(d["start"]),
        _float(d["end"]),
        _float(d["vocal"]),
        _float(d["loudness"]),
    )


def _lyric(d: Json) -> LyricLine:
    return LyricLine(_float(d["start"]), _float(d["end"]), str(d["text"]), str(d.get("lang", "")))


def _slot(d: Json) -> Slot:
    return Slot(
        _int(d["index"]), _float(d["start"]), _float(d["end"]), str(d["section"]), _int(d["beats"])
    )


def store() -> Path:
    home = os.environ.get("TOKIMEKI_HOME")
    base = Path(home) if home else Path.home() / ".local" / "share" / "tokimeki"
    return base / "songs"


def analysis_path(song_id: str) -> Path:
    return store() / song_id / "analysis.json"


def load_analysis(song_id: str) -> SongAnalysis:
    path = analysis_path(song_id)
    if not path.exists():
        known = sorted(p.name for p in store().glob("*") if (p / "analysis.json").exists())
        raise SongError(f"no song {song_id!r}; analysed songs: {', '.join(known) or 'none'}")
    return SongAnalysis.from_json(path.read_text(encoding="utf-8"))


def _probe(path: Path) -> tuple[float, str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=title",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if result.returncode != 0:
        raise SongError(f"cannot read {path}: {result.stderr.strip()}")
    fmt = cast(Json, cast(Json, json.loads(result.stdout)).get("format", {}))
    tags = cast(dict[str, str], fmt.get("tags", {}))
    title = next((v for k, v in tags.items() if k.lower() == "title"), "")
    return float(str(fmt.get("duration", "0"))), title


def resolve(
    path: Path,
    track: int | None = None,
    stream: int | None = None,
    within: tuple[float, float] | None = None,
    title: str | None = None,
) -> SongSource:
    """The song at `path`: an audio file, a track of a `.cue` CD image, or the stretch of any
    media file roughly `within` a window (its exact start and end found from the audio), on
    `stream` (default: the programme audio, never commentary)."""
    if not path.exists():
        raise SongError(f"{path} does not exist")
    if path.suffix.lower() == ".cue":
        sheet = read_cue(path)
        total, _ = _probe(sheet.file)
        t = sheet.track(track or 1)
        end = t.end if t.end is not None else total
        return SongSource(sheet.file, t.start, end - t.start, title or t.title, t.number)
    duration, tag_title = _probe(path)
    if stream is None:
        streams = audio_streams(path)
        stream = main_audio(path).index if len(streams) > 1 else None
    name = title or (path.stem if within else tag_title or path.stem)
    if within is None:
        return SongSource(path, 0.0, duration, name, stream=stream)
    start, end = within
    if not 0 <= start < end <= duration:
        raise SongError(f"window {start:.1f}-{end:.1f}s is outside {path.name} (0-{duration:.1f}s)")
    lead = max(0.0, start - bounds.SEARCH_BEFORE - 2)
    tail = min(duration, end + bounds.SEARCH_BEFORE + 2)
    samples = decode_audio(path, BOUNDS_RATE, stream, start=lead, duration=tail - lead)
    first, last = bounds.song_bounds(samples, BOUNDS_RATE, lead, start, end)
    return SongSource(path, round(first, 3), round(last - first, 3), name, stream=stream)


def song_id(source: SongSource) -> str:
    """A readable, stable id: the title's ASCII words and a hash of where the audio is."""
    words = re.findall(r"[a-z0-9]+", source.title.lower())
    slug = "-".join(words)[:40] or "song"
    key = f"{source.path.resolve()}|{source.track}|{source.stream}|{source.offset:.3f}"
    return f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def _audio(source: SongSource) -> NDArray[np.float32]:
    return decode_audio(
        source.path, SAMPLE_RATE, source.stream, start=source.offset, duration=source.duration
    )


def lyrics_share(
    lyrics: list[LyricLine], bars: NDArray[np.float64], end: float
) -> NDArray[np.float64]:
    edges = np.append(bars, end)
    out: list[float] = []
    for a, b in pairwise(edges):
        sung = sum(max(0.0, min(b, x.end) - max(a, x.start)) for x in lyrics)
        out.append(sung / max(b - a, 1e-9))
    return np.array(out)


SUNG = {"lyrics": 0.2, "instrumental": 0.2, "separation": 0.3}
"""Per source of the vocal line, the share of a bar that makes it sung. Separation's scale
differs: 0.3 matched the instrumental-derived line on 95.5% of MORE&MORE's bars."""


def refrain_bars(
    lyrics: list[LyricLine], bars: NDArray[np.float64], end: float
) -> NDArray[np.bool_] | None:
    """Bars at least a third covered by lyric lines that are sung more than once."""
    if not lyrics:
        return None

    def key(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", text.lower())

    counts: dict[str, int] = {}
    for line in lyrics:
        counts[key(line.text)] = counts.get(key(line.text), 0) + 1
    repeated = [x for x in lyrics if counts[key(x.text)] > 1]
    if not repeated:
        return None
    return lyrics_share(repeated, bars, end) >= 1 / 3


def vocal_line(
    bars: NDArray[np.float64],
    end: float,
    mix: NDArray[np.float64],
    lyrics: list[LyricLine],
    backing: NDArray[np.float64] | None,
    separated: NDArray[np.float32] | None,
) -> tuple[NDArray[np.float64] | None, str]:
    """Share of each bar that is sung, from the best source given, and which one it was."""
    if lyrics:
        return lyrics_share(lyrics, bars, end), "lyrics"
    if backing is not None:
        frames = min(len(mix), len(backing))
        lag = structure.best_lag(mix[:frames].sum(axis=1), backing[:frames].sum(axis=1), 50)
        aligned = np.roll(backing[:frames], lag, axis=0)
        low, high = structure.VOCAL_BAND
        centres = mel_centres()
        band = (centres >= low) & (centres <= high)
        share = structure.vocal_share(np.expm1(mix[:frames]) / 1000, np.expm1(aligned) / 1000, band)
        return structure.per_bar(share[:, None], FPS, bars, end)[:, 0], "instrumental"
    if separated is not None:
        per = separated.astype(np.float64)[:, None]
        return structure.per_bar(per, SEPARATION_FPS, bars, end)[:, 0], "separation"
    return None, "none"


def assemble(
    source: SongSource,
    beats: Beats,
    mix: NDArray[np.float64],
    end: float,
    lyrics: list[LyricLine],
    backing: NDArray[np.float64] | None = None,
    separated: NDArray[np.float32] | None = None,
) -> SongAnalysis:
    """The analysis from what the GPU models measured; pure, so it is tested without them."""
    bars = structure.bar_grid(beats.beats, beats.downbeats)
    vocal, vocal_source = vocal_line(bars, end, mix, lyrics, backing, separated)
    shares = vocal if vocal is not None else np.ones(len(bars))
    sung_at = SUNG.get(vocal_source, structure.VOCAL_ON)
    loudness = structure.per_bar(mix.mean(axis=1, keepdims=True), FPS, bars, end)[:, 0]
    features = structure.per_bar(mix, FPS, bars, end)
    refrain = refrain_bars(lyrics, bars, end)
    starts = structure.boundaries(structure.novelty(features), shares >= sung_at, refrain=refrain)
    sections = structure.merge_runs(
        structure.label_sections(
            starts,
            features,
            shares,
            loudness,
            bars,
            end,
            vocal_known=vocal is not None,
            vocal_on=sung_at,
            refrain=refrain,
        )
    )
    if len(beats.beats) and sections and beats.beats[0] < sections[0].start - 1e-6:
        sections[0] = replace(sections[0], start=float(beats.beats[0]))
    intervals = np.diff(beats.beats)
    return SongAnalysis(
        id=song_id(source),
        title=source.title,
        source=source,
        bpm=float(60.0 / np.median(intervals)) if len(intervals) else 0.0,
        beats=[float(b) for b in beats.beats],
        bars=[float(b) for b in bars],
        energy=[float(v) for v in loudness],
        vocal=None if vocal is None else [float(v) for v in vocal],
        vocal_source=vocal_source,
        sections=sections,
        lyrics=lyrics,
    )


def measure(
    source: SongSource, instrumental: SongSource | None, lyrics: list[LyricLine]
) -> SongAnalysis:
    """Run the GPU models one at a time: Beat This!, then (when neither lyrics nor an
    instrumental give the vocal line) MDX-Net on the mix."""
    audio = _audio(source)
    backing: NDArray[np.float64] | None = None
    with loaded("Beat This!", BeatTracker) as tracker:
        beats = tracker.track(audio)
        mix = tracker.spectrogram(audio).astype(np.float64)
        if instrumental is not None and not lyrics:
            backing = tracker.spectrogram(_audio(instrumental)).astype(np.float64)
    separated: NDArray[np.float32] | None = None
    if not lyrics and backing is None:
        stereo = decode_audio(
            source.path,
            SEPARATION_RATE,
            source.stream,
            start=source.offset,
            duration=source.duration,
            channels=2,
        )
        with loaded("MDX-Net", VocalSeparator) as separator:
            separated = separator.vocal_share(stereo)
    return assemble(source, beats, mix, len(audio) / SAMPLE_RATE, lyrics, backing, separated)


def excerpt_of(analysis: SongAnalysis, span: tuple[float, float] | None) -> Excerpt:
    if span is None:
        return first_chorus(analysis.sections)
    start, end = span
    if not 0 <= start < end <= analysis.source.duration + 1e-6:
        raise SongError(
            f"range {start:.1f}-{end:.1f}s is outside the song (0-{analysis.source.duration:.1f}s)"
        )
    return excerpt_between(analysis.bars, analysis.source.duration, analysis.sections, start, end)


def slots_over(
    analysis: SongAnalysis,
    span: tuple[float, float] | None = None,
    budget: int = UNLIMITED,
    fixed: Mapping[str, int] | None = None,
) -> tuple[Excerpt, list[Slot]]:
    """The excerpt (`span` snapped to bar lines, or the stored one) and slots over it: a bar
    a cut in verses, two beats in the chorus (or `fixed` beats per section kind), slowed down
    section by section until no more than `budget` shots are needed."""
    excerpt = excerpt_of(analysis, span if span is not None else analysis.excerpt)
    return excerpt, make_slots(analysis.beats, analysis.bars, analysis.bpm, excerpt, budget, fixed)


def suggested_slots(analysis: SongAnalysis, budget: int = UNLIMITED) -> list[Slot]:
    return slots_over(analysis, None, budget)[1]


def analyse_song(
    path: Path,
    track: int | None = None,
    lyrics: Sequence[LyricSource] = (),
    instrumental: Path | None = None,
    span: tuple[float, float] | None = None,
    instrumental_track: int | None = None,
    stream: int | None = None,
    within: tuple[float, float] | None = None,
    title: str | None = None,
) -> SongAnalysis:
    """Analyse a song (reusing the stored beats and sections when their inputs match), pick
    its excerpt (`span`; by default the whole song when it is short, else the top through
    the first chorus) and store the result.

    The vocal line comes from the first lyrics source if any, else an instrumental (a file,
    or `instrumental_track` of the same `.cue`), else from separating the mix.
    """
    source = resolve(path, track, stream, within, title)
    backing: SongSource | None = None
    if instrumental is not None:
        backing = resolve(instrumental)
    elif instrumental_track is not None:
        if path.suffix.lower() != ".cue":
            raise SongError("--instrumental-track needs a .cue sheet")
        backing = resolve(path, instrumental_track)
    inputs = {
        "lyrics": " ".join(x.key() for x in lyrics),
        "instrumental": f"{backing.path.resolve()}|{backing.offset:.3f}" if backing else "",
        "separation": SEPARATION_MODEL,
    }
    lines = [
        line
        for src in lyrics
        for line in read_lyrics(src, source.offset)
        if line.end > 0 and line.start < source.duration
    ]
    sid = song_id(source)
    analysis: SongAnalysis | None = None
    if analysis_path(sid).exists():
        try:
            stored = load_analysis(sid)
            analysis = stored if stored.inputs == inputs and stored.source == source else None
        except SongError:
            analysis = None
    if analysis is None:
        first_language = [x for x in lines if not lyrics or x.lang == lyrics[0].lang]
        analysis = replace(measure(source, backing, first_language), lyrics=lines, inputs=inputs)
    excerpt = excerpt_of(analysis, span)
    analysis = replace(analysis, excerpt=(excerpt.start, excerpt.end))
    analysis = replace(analysis, slots=suggested_slots(analysis))
    save_analysis(analysis)
    return analysis


def save_analysis(analysis: SongAnalysis) -> None:
    out = analysis_path(analysis.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(analysis.to_json(), encoding="utf-8")
