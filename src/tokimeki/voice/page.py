"""A static listening page for naming speakers: audio players per speaker, nothing more."""

from html import escape
from pathlib import Path

from tokimeki.voice.review import LineClip, SpeakerReview, VoiceReview

_CSS = """
:root { color-scheme: light dark; --muted: #888; --line: #8884; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 1rem; }
section { border-top: 1px solid var(--line); margin-top: 1.5rem; }
.clip { display: grid; grid-template-columns: 300px 1fr; gap: .2rem 1rem; margin: .5rem 0; }
audio { width: 300px; height: 32px; } .muted { color: var(--muted); }
code { background: #8882; padding: 0 .3rem; border-radius: 4px; }
"""


def _clip(c: LineClip) -> str:
    ja = escape(c.transcript) if c.transcript else '<span class="muted">-</span>'
    return (
        f'<div class="clip"><audio controls preload="none" src="{escape(c.audio)}"></audio>'
        f"<div>{ja}<br><span class=muted>{escape(c.subtitle)} · line {c.line}"
        f" · ep {c.episode} {c.start:.1f}s · {c.end - c.start:.1f}s · d {c.distance:.2f}"
        "</span></div></div>"
    )


def _speaker(series: str, s: SpeakerReview) -> str:
    title = escape(s.name) if s.name else f"#{s.id} (unnamed)"
    hint = f" · looks like <b>{escape(s.suggestion)}</b>" if s.suggestion and not s.name else ""
    seen = ", ".join(f"{escape(m.name)} {m.share:.0%} lift {m.lift:.1f}" for m in s.matches) or "-"
    votes = ", ".join(f"{escape(n)} {k}" for n, k in s.seeds.items()) or "-"
    rows = "".join(_clip(c) for c in s.typical)
    if s.doubtful:
        rows += "<p class=muted>least typical:</p>" + "".join(_clip(c) for c in s.doubtful)
    command = f"tokimeki voice name {escape(series)} {s.id} NAME"
    return (
        f'<section id="s{s.id}"><h2>#{s.id} {title}</h2>'
        f"<p>{s.lines} lines, {s.seconds / 60:.1f} min{hint}<br>"
        f"<span class=muted>close-up seeds: {votes}; on screen while speaking: {seen}</span>"
        f"<br><code>{command}</code></p>"
        f"{rows}</section>"
    )


def render_page(series: str, data: VoiceReview) -> str:
    counts = ", ".join(f"{n} {status}" for status, n in data.statuses.items() if n)
    body = "".join(_speaker(series, s) for s in data.speakers)
    if data.unassigned:
        body += "<section><h2>No speaker yet</h2>"
        body += "".join(_clip(c) for c in data.unassigned) + "</section>"
    return (
        '<!doctype html><html lang="ja"><meta charset="utf-8">'
        f"<title>voices</title><style>{_CSS}</style>"
        f"<h1>Speakers</h1><p>Lines: {counts}; {data.clean_seconds / 60:.1f} min of clean voice."
        f" Move a line with <code>tokimeki voice move {escape(series)} --line ID --to NAME</code>,"
        f" merge with <code>tokimeki voice merge {escape(series)} ID… TARGET</code>.</p>"
        f"{body}</html>"
    )


def write_page(root: Path, series: str, data: VoiceReview) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.html"
    path.write_text(render_page(series, data))
    return path
