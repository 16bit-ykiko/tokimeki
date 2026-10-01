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
table { border-collapse: collapse; width: 100%; } td, th { padding: .3rem .5rem;
  text-align: left; vertical-align: top; border-top: 1px solid var(--line); }
td img { width: 160px; border-radius: 4px; display: block; }
"""


def _clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def write_report(
    paths: SeriesPaths,
    plan: Plan,
    song_title: str,
    out_dir: Path,
    candidates: Mapping[int, Candidate],
) -> Path:
    img = out_dir / "img"
    shutil.rmtree(img, ignore_errors=True)
    img.mkdir(parents=True)
    rows: list[str] = []
    for i, slot in enumerate(plan.slots):
        c = candidates.get(slot.shot) if slot.shot is not None else None
        if c is None:
            continue
        thumb = ""
        middle = ((slot.source_in or c.peak.time) + (slot.source_out or c.peak.time)) / 2
        nearest = min(c.frames, key=lambda f: abs(f.time - middle), default=None)
        if nearest is not None:
            source = paths.frame_path(c.episode.id, round(nearest.time * c.episode.fps))
            if source.exists():
                image = load_image(source, THUMB_WIDTH)
                height = round(image.height * THUMB_WIDTH / image.width)
                name = f"slot-{i}.jpg"
                image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS).save(img / name)
                thumb = f'<img src="img/{name}" alt="shot {slot.shot}">'
        speed = slot.derived_speed
        window = "not placed"
        if slot.source_in is not None and speed is not None:
            window = f"{_clock(slot.source_in)} for {slot.duration * speed:.2f}s at {speed:.2f}x"
        rows.append(
            f"<tr><td>{i}</td><td>{escape(slot.section)}<br>"
            f'<span class="muted">{_clock(slot.start)}, {slot.duration:.2f}s</span></td>'
            f"<td>{thumb}</td><td>shot {slot.shot}<br>{escape(c.episode.path)}<br>"
            f'<span class="muted">{window}</span></td>'
            f"<td>{escape(c.describe())}</td><td>{escape(slot.why)}</td></tr>"
        )
    links = " · ".join(
        f'<a href="{n}">{n}</a>'
        for n in ("preview.mp4", "final.mp4", "timeline.otio", "plan.json")
        if (out_dir / n).exists()
    )
    preview = (
        '<video controls src="preview.mp4"></video>' if (out_dir / "preview.mp4").exists() else ""
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(plan.name)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(plan.name)}</h1>"
        f'<p class="muted">{escape(plan.character)} to {escape(song_title)},'
        f" {_clock(plan.start)}-{_clock(plan.end)} of the song"
        f" ({plan.end - plan.start:.1f}s); {len(plan.slots)} cuts</p>"
        f"{preview}<p>{links}</p>"
        "<table><thead><tr><th>#</th><th>slot</th><th>frame</th><th>shot</th>"
        "<th>what it shows</th><th>why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
    index = out_dir / "report.html"
    index.write_text(html, encoding="utf-8")
    return index
