"""A MAD's report: which shot went where and why, next to the preview."""

import shutil
from collections.abc import Mapping
from html import escape
from pathlib import Path

from PIL import Image

from tokimeki.mad.candidates import Candidate
from tokimeki.mad.plan import Plan
from tokimeki.media.images import load_image
from tokimeki.paths import SeriesPaths

THUMB_WIDTH = 240

_CSS = """
:root { color-scheme: light dark; --muted: #888; --line: #8884; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1400px;
  padding: 1rem 2rem; }
.muted { color: var(--muted); } video { max-width: 100%; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; } td, th { padding: .3rem .5rem; text-align: left;
  vertical-align: top; border-top: 1px solid var(--line); }
td img { width: 160px; border-radius: 4px; display: block; }
"""


def _clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def write_report(
    paths: SeriesPaths, plan: Plan, out_dir: Path, candidates: Mapping[int, Candidate]
) -> Path:
    img = out_dir / "img"
    shutil.rmtree(img, ignore_errors=True)
    img.mkdir(parents=True)
    rows: list[str] = []
    for clip in sorted(plan.clips, key=lambda c: c.slot):
        slot = plan.slots[clip.slot]
        c = candidates[clip.shot]
        peak = clip.peak if clip.peak is not None else c.peak.time
        frame = round(peak * c.episode.fps)
        source = paths.frame_path(c.episode.id, frame)
        thumb = ""
        if source.exists():
            image = load_image(source, THUMB_WIDTH)
            height = round(image.height * THUMB_WIDTH / image.width)
            name = f"slot-{slot.index}.jpg"
            image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS).save(img / name)
            thumb = f'<img src="img/{name}" alt="shot {clip.shot}">'
        window = "not placed"
        if clip.source_start is not None and clip.speed is not None:
            length = slot.duration * clip.speed
            window = f"{_clock(clip.source_start)} for {length:.2f}s at {clip.speed:.2f}x"
        rows.append(
            f"<tr><td>{slot.index}</td><td>{slot.section}<br>"
            f'<span class="muted">{_clock(slot.start)}, {slot.duration:.2f}s,'
            f" {slot.beats} beats</span></td>"
            f"<td>{thumb}</td><td>shot {clip.shot}<br>{escape(c.episode.path)}<br>"
            f'<span class="muted">{window}</span></td>'
            f"<td>{escape(c.describe())}</td><td>{escape(clip.reason)}</td></tr>"
        )
    videos = "".join(
        f'<p><a href="{n}.mp4">{n}.mp4</a></p>'
        for n in ("preview", "final")
        if (out_dir / f"{n}.mp4").exists()
    )
    preview = (
        '<video controls src="preview.mp4"></video>' if (out_dir / "preview.mp4").exists() else ""
    )
    song = plan.song
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(plan.name)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(plan.name)}</h1>"
        f'<p class="muted">{escape(plan.preferences.character)} to {escape(song.title)},'
        f" {_clock(song.start)}-{_clock(song.end)} of the song"
        f" ({song.duration:.1f}s, {song.bpm:.0f} bpm);"
        f" {len(plan.slots)} cuts; arranged by {escape(plan.arranger)}</p>"
        f"{preview}{videos}"
        '<p><a href="plan.json">plan.json</a> · <a href="timeline.otio">timeline.otio</a></p>'
        "<table><thead><tr><th>#</th><th>slot</th><th>peak frame</th><th>shot</th>"
        "<th>what it shows</th><th>why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
    index = out_dir / "report.html"
    index.write_text(html, encoding="utf-8")
    return index
