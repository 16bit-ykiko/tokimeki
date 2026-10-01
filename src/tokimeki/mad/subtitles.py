"""Lyric subtitles for a MAD: Japanese above, Chinese below, clean MAD style.

Written as an ASS file next to the video (and burnt in through libass when asked) and as an
SRT for editors that take nothing else.
"""

from collections.abc import Sequence
from pathlib import Path

from tokimeki.mad.fonts import Fonts
from tokimeki.song.lyrics import LyricPair

WIDTH, HEIGHT = 1920, 1080
FILL = "&H00FFFFFF"
OUTLINE = "&H008F48D9"
"""A soft magenta (#D9488F, in ASS's BGR order)."""
SHADOW = "&H78000000"
FADE_IN, FADE_OUT = 150, 220
JAPANESE_SIZE, CHINESE_SIZE = 60, 42
BOTTOM = 56
LANGUAGES = ("ja", "zh")


def _clock(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _srt_clock(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rest = divmod(millis, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def timed(
    pairs: Sequence[LyricPair], start: float, end: float
) -> list[tuple[float, float, dict[str, str]]]:
    """The lines sung within `start`-`end` (song seconds), on the MAD's clock."""
    out: list[tuple[float, float, dict[str, str]]] = []
    for p in pairs:
        a, b = max(p.start, start) - start, min(p.end, end) - start
        if b - a > 0.2:
            out.append((a, b, p.texts))
    return out


STYLE_FIELDS = (
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour",
    "BackColour", "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing",
    "Angle", "BorderStyle", "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
    "Encoding",
)  # fmt: skip
EVENT_FIELDS = (
    "Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"
)  # fmt: skip


def _style(
    name: str, font: str, size: int, spacing: float, outline: float, shadow: float, margin: int
) -> str:
    values = (
        name, font, size, FILL, FILL, OUTLINE, SHADOW, -1, 0, 0, 0, 100, 100, spacing,
        0, 1, outline, shadow, 2, 140, 140, margin, 1,
    )  # fmt: skip
    return "Style: " + ",".join(str(v) for v in values)


def lyrics_ass(
    lines: Sequence[tuple[float, float, dict[str, str]]], fonts: Fonts, title: str
) -> str:
    chinese_margin = BOTTOM
    japanese_margin = BOTTOM + round(CHINESE_SIZE * 1.35)
    head = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: " + ", ".join(STYLE_FIELDS),
        _style("ja", fonts.japanese, JAPANESE_SIZE, 1, 3.4, 1.6, japanese_margin),
        _style("zh", fonts.chinese, CHINESE_SIZE, 0.5, 2.8, 1.4, chinese_margin),
        "",
        "[Events]",
        "Format: " + ", ".join(EVENT_FIELDS),
    ]
    fade = f"{{\\fad({FADE_IN},{FADE_OUT})}}"
    events = [
        f"Dialogue: 0,{_clock(a)},{_clock(b)},{lang},,0,0,0,,{fade}{_escape(texts[lang])}"
        for a, b, texts in lines
        for lang in LANGUAGES
        if texts.get(lang)
    ]
    return "\n".join(head + events) + "\n"


def lyrics_srt(lines: Sequence[tuple[float, float, dict[str, str]]]) -> str:
    blocks = [
        f"{i}\n{_srt_clock(a)} --> {_srt_clock(b)}\n"
        + "\n".join(texts[lang] for lang in LANGUAGES if texts.get(lang))
        for i, (a, b, texts) in enumerate(lines, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def write_lyrics(
    out_dir: Path, pairs: Sequence[LyricPair], start: float, end: float, fonts: Fonts, title: str
) -> Path | None:
    """`lyrics.ass` and `lyrics.srt` for the stretch `start`-`end`; None when nothing is sung."""
    lines = timed(pairs, start, end)
    if not lines:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ass = out_dir / "lyrics.ass"
    ass.write_text(lyrics_ass(lines, fonts, title), encoding="utf-8-sig")
    (out_dir / "lyrics.srt").write_text(lyrics_srt(lines), encoding="utf-8")
    return ass
