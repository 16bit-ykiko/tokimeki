"""Analysing a song once: beats and bars (Beat This! on the GPU), vocal line, sections.

Results are cached as JSON per song under `.tokimeki/songs/`, so a MAD can be re-planned
without touching the GPU again.
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tokimeki.media.audio import decode_audio
from tokimeki.media.cue import read_cue
from tokimeki.models.beats import FPS, SAMPLE_RATE, BeatTracker, mel_centres
from tokimeki.models.gpu import loaded
from tokimeki.paths import SeriesPaths
from tokimeki.song import structure
from tokimeki.song.structure import Section

ANALYSIS_VERSION = 1
_INSTRUMENTAL = re.compile(r"\((instrumental|off vocal|karaoke)\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Song:
    """One song: a stretch of an audio file (a whole file, or a track of a CD image)."""

    id: str
    title: str
    path: Path
    offset: float
    duration: float


@dataclass(frozen=True)
class SongAnalysis:
    song: Song
    bpm: float
    beats: list[float]
    bars: list[float]
    vocal: list[float]
    loudness: list[float]
    sections: list[Section]
    has_vocal_line: bool

    def to_json(self) -> str:
        data = asdict(self)
        data["song"]["path"] = str(self.song.path)
        data["version"] = ANALYSIS_VERSION
        return json.dumps(data, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> "SongAnalysis":
        data = cast(dict[str, object], json.loads(text))
        song = cast(dict[str, object], data["song"])
        sections = cast(list[dict[str, object]], data["sections"])
        return SongAnalysis(
            Song(
                str(song["id"]),
                str(song["title"]),
                Path(str(song["path"])),
                float(cast(float, song["offset"])),
                float(cast(float, song["duration"])),
            ),
            float(cast(float, data["bpm"])),
            cast(list[float], data["beats"]),
            cast(list[float], data["bars"]),
            cast(list[float], data["vocal"]),
            cast(list[float], data["loudness"]),
            [
                Section(
                    str(s["label"]),
                    int(cast(int, s["group"])),
                    int(cast(int, s["first_bar"])),
                    int(cast(int, s["end_bar"])),
                    float(cast(float, s["start"])),
                    float(cast(float, s["end"])),
                    float(cast(float, s["vocal"])),
                    float(cast(float, s["loudness"])),
                )
                for s in sections
            ],
            bool(data["has_vocal_line"]),
        )


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def find_song(path: Path, track: int | None = None) -> tuple[Song, Song | None]:
    """The song at `path` (a CUE track or an audio file) and its instrumental version, if the
    CD has one (a track titled "<title> (Instrumental)")."""
    if path.suffix.lower() != ".cue":
        return Song(path.stem, path.stem, path, 0.0, _duration(path)), None
    sheet = read_cue(path)
    chosen = sheet.track(track or 1)
    total = _duration(sheet.file)

    def song(number: int) -> Song:
        t = sheet.track(number)
        end = t.end if t.end is not None else total
        return Song(f"{path.stem}-t{number:02d}", t.title, sheet.file, t.start, end - t.start)

    instrumental = next(
        (
            song(t.number)
            for t in sheet.tracks
            if t.number != chosen.number
            and _INSTRUMENTAL.search(t.title)
            and _INSTRUMENTAL.sub("", t.title).strip() == chosen.title.strip()
        ),
        None,
    )
    return song(chosen.number), instrumental


def _audio(song: Song) -> NDArray[np.float32]:
    return decode_audio(song.path, SAMPLE_RATE, start=song.offset, duration=song.duration)


def analyse(song: Song, instrumental: Song | None, tracker: BeatTracker) -> SongAnalysis:
    audio = _audio(song)
    beats = tracker.track(audio)
    mix = tracker.spectrogram(audio).astype(np.float64)
    bars = structure.bar_grid(beats.beats, beats.downbeats)
    end = len(audio) / SAMPLE_RATE
    if instrumental is not None:
        backing = tracker.spectrogram(_audio(instrumental)).astype(np.float64)
        frames = min(len(mix), len(backing))
        lag = structure.best_lag(mix[:frames].sum(axis=1), backing[:frames].sum(axis=1), 50)
        backing = np.roll(backing[:frames], lag, axis=0)
        low, high = structure.VOCAL_BAND
        centres = mel_centres()
        band = (centres >= low) & (centres <= high)
        share = structure.vocal_share(np.expm1(mix[:frames]) / 1000, np.expm1(backing) / 1000, band)
        vocal = structure.per_bar(share[:, None], FPS, bars, end)[:, 0]
    else:
        vocal = np.ones(len(bars))
    loudness = structure.per_bar(mix.mean(axis=1, keepdims=True), FPS, bars, end)[:, 0]
    features = structure.per_bar(mix, FPS, bars, end)
    starts = structure.boundaries(structure.novelty(features), vocal >= structure.VOCAL_ON)
    sections = structure.merge_runs(
        structure.label_sections(starts, features, vocal, loudness, bars, end)
    )
    intervals = np.diff(beats.beats)
    bpm = float(60.0 / np.median(intervals)) if len(intervals) else 0.0
    return SongAnalysis(
        song,
        bpm,
        [float(b) for b in beats.beats],
        [float(b) for b in bars],
        [float(v) for v in vocal],
        [float(v) for v in loudness],
        sections,
        instrumental is not None,
    )


def analysis_file(paths: SeriesPaths, song: Song) -> Path:
    return paths.data_dir / "songs" / song.id / "analysis.json"


def analyse_song(paths: SeriesPaths, path: Path, track: int | None = None) -> SongAnalysis:
    """The cached analysis of a song, computed on first use."""
    song, instrumental = find_song(path, track)
    cached = analysis_file(paths, song)
    if cached.exists():
        analysis = SongAnalysis.from_json(cached.read_text(encoding="utf-8"))
        if analysis.song == song:
            return analysis
    with loaded("Beat This!", BeatTracker) as tracker:
        analysis = analyse(song, instrumental, tracker)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(analysis.to_json(), encoding="utf-8")
    return analysis
