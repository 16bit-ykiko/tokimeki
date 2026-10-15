"""Where voice data lives.

Per series, under its data directory: the listening page, and per character the bank, the
trained models and the rendered samples. Shared by every series, under `$TOKIMEKI_HOME`
(default `~/.local/share/tokimeki`): the external TTS engines with their own environments.
"""

import os
from pathlib import Path

from tokimeki.library.paths import SeriesPaths


def home() -> Path:
    value = os.environ.get("TOKIMEKI_HOME")
    return Path(value) if value else Path.home() / ".local" / "share" / "tokimeki"


def engines_dir() -> Path:
    return home() / "engines"


def voice_dir(paths: SeriesPaths) -> Path:
    return paths.data_dir / "voice"


def page_dir(paths: SeriesPaths) -> Path:
    return voice_dir(paths) / "page"


def character_dir(paths: SeriesPaths, character: str) -> Path:
    return voice_dir(paths) / character


def bank_dir(paths: SeriesPaths, character: str) -> Path:
    return character_dir(paths, character) / "bank"


def model_dir(paths: SeriesPaths, character: str, engine: str) -> Path:
    return character_dir(paths, character) / "models" / engine


def samples_dir(paths: SeriesPaths, character: str) -> Path:
    return character_dir(paths, character) / "samples"
