"""Fonts for burnt-in subtitles: found on the system (or Windows, under WSL) and copied into
the series' data dir, never into the repository."""

import shutil
from dataclasses import dataclass
from pathlib import Path

SEARCH_DIRS = (
    Path.home() / ".local/share/fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/mnt/c/Windows/Fonts"),
)

JAPANESE = (
    ("Yu Gothic", "YuGothB.ttc"),
    ("Noto Sans CJK JP", "NotoSansCJKjp-Bold.otf"),
    ("Noto Sans CJK JP", "NotoSansCJK-Bold.ttc"),
    ("Noto Sans CJK SC", "NotoSansCJKsc-Bold.otf"),
    ("Noto Sans CJK SC", "NotoSansCJKsc-Regular.otf"),
)
"""Family names and files, in order of preference: a Japanese face first, so kanji take
their Japanese shapes."""

CHINESE = (
    ("Microsoft YaHei", "msyhbd.ttc"),
    ("Noto Sans CJK SC", "NotoSansCJKsc-Bold.otf"),
    ("Noto Sans CJK SC", "NotoSansCJKsc-Regular.otf"),
    ("Noto Sans CJK SC", "NotoSansCJK-Bold.ttc"),
)


class FontError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Fonts:
    japanese: str
    chinese: str
    directory: Path


def _locate(name: str) -> Path | None:
    for directory in SEARCH_DIRS:
        if not directory.is_dir():
            continue
        direct = directory / name
        if direct.is_file():
            return direct
        found = next(iter(directory.rglob(name)), None) if directory.parts[1] != "mnt" else None
        if found is not None:
            return found
    return None


def _choose(candidates: tuple[tuple[str, str], ...], target: Path) -> str:
    for family, name in candidates:
        if (target / name).is_file():
            return family
        source = _locate(name)
        if source is not None:
            shutil.copy2(source, target / name)
            return family
    raise FontError(f"no font for subtitles found; tried {[n for _, n in candidates]}")


def prepare_fonts(target: Path) -> Fonts:
    """Japanese and Chinese families for the subtitles, their files copied into `target`."""
    target.mkdir(parents=True, exist_ok=True)
    return Fonts(_choose(JAPANESE, target), _choose(CHINESE, target), target)
