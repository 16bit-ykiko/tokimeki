from collections.abc import Iterable
from datetime import datetime
from html import escape
from pathlib import Path

from tokimeki.library.records import Episode, Shot
from tokimeki.report.data import ClusterReport, EpisodeReport, KeptShot, ReportData

CLOSE_UP_HEIGHT = 0.35
WIDE_HEIGHT = 0.15

_CSS = """
:root { color-scheme: light dark; --muted: #888; --line: #8884; --chip: #8882; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1600px;
  padding: 1rem 2rem; }
h1 { margin-bottom: 0; } h2 { margin-top: 2.5rem; border-bottom: 1px solid var(--line); }
nav a { margin-right: .8rem; } .muted { color: var(--muted); }
table { border-collapse: collapse; } td, th { padding: .25rem .8rem; text-align: right; }
td:first-child, th:first-child { text-align: left; } tr.total { font-weight: 600; }
tbody tr { border-top: 1px solid var(--line); }
code { background: var(--chip); padding: .1rem .3rem; border-radius: 4px; }
.clusters { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
  gap: 1rem; }
.cluster { border: 1px solid var(--line); border-radius: 8px; padding: .6rem .8rem; }
.cluster h3 { margin: 0; } .faces { display: flex; flex-wrap: wrap; gap: 4px; margin-top: .4rem; }
.faces img, .faces .missing { width: 56px; height: 56px; border-radius: 4px; }
.shots { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: .8rem; }
figure { margin: 0; }
figure img, figure .missing { width: 100%; aspect-ratio: 16/9; border-radius: 4px;
  object-fit: cover; display: block; }
.missing { background: var(--chip); }
figcaption { font-size: 12px; margin-top: .2rem; }
.chip { display: inline-block; border-radius: 10px; padding: 0 .45rem; margin: .1rem .15rem 0 0;
  background: hsl(var(--h) 70% 50% / .22); }
.tags { color: var(--muted); }
"""


def clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def span(episode: Episode, shot: Shot) -> str:
    start, end = episode.seconds(shot.start_frame), episode.seconds(shot.end_frame)
    return f"{clock(start)}&ndash;{clock(end)}"


def framing(face_height: float) -> str:
    if face_height >= CLOSE_UP_HEIGHT:
        return "close-up"
    return "medium" if face_height >= WIDE_HEIGHT else "wide"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "&ndash;"


def _image(name: str, images: set[str], alt: str) -> str:
    if name in images:
        return f'<img loading="lazy" src="img/{name}" alt="{escape(alt)}">'
    return '<div class="missing"></div>'


def _labels(clusters: Iterable[ClusterReport]) -> dict[int, str]:
    return {c.cluster.id: c.cluster.label for c in clusters}


def _chip(cluster_id: int, text: str) -> str:
    return f'<span class="chip" style="--h:{cluster_id * 47 % 360}">{escape(text)}</span>'


def _totals(episodes: list[EpisodeReport]) -> str:
    rows: list[str] = []
    total = kept = dropped = 0
    for ep in episodes:
        total, kept, dropped = total + ep.total, kept + len(ep.kept), dropped + len(ep.dropped)
        rows.append(
            f'<tr><td><a href="#ep-{ep.episode.id}">{escape(ep.episode.path)}</a></td>'
            f"<td>{ep.total}</td><td>{len(ep.kept)}</td><td>{len(ep.dropped)}</td>"
            f"<td>{_pct(len(ep.dropped), ep.total)}</td></tr>"
        )
    rows.append(
        f'<tr class="total"><td>all episodes</td><td>{total}</td><td>{kept}</td>'
        f"<td>{dropped}</td><td>{_pct(dropped, total)}</td></tr>"
    )
    head = "<tr><th>episode</th><th>shots</th><th>kept</th><th>dropped</th><th>dropped %</th></tr>"
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _cluster(c: ClusterReport, series_dir: str, images: set[str]) -> str:
    name = escape(c.cluster.name) if c.cluster.name else '<span class="muted">unnamed</span>'
    hints = ", ".join(f"{escape(tag)} {share:.0%}" for tag, share in c.hints)
    faces = "".join(_image(f"face-{f.face.id}.jpg", images, f"face {f.face.id}") for f in c.samples)
    command = f'pixi run tokimeki cast name {series_dir} {c.cluster.id} "NAME"'
    return (
        f'<article class="cluster" id="cluster-{c.cluster.id}">'
        f"<h3>{_chip(c.cluster.id, f'#{c.cluster.id}')} {name}</h3>"
        f'<div class="muted">{c.cluster.face_count} faces in {c.shots} shots'
        f"{' · WD14: ' + hints if hints else ''}</div>"
        f'<div class="faces">{faces}</div>'
        f'<div class="muted"><code>{escape(command)}</code></div></article>'
    )


def _shot(ep: Episode, k: KeptShot, labels: dict[int, str], images: set[str]) -> str:
    cast = "".join(
        _chip(
            c.cluster_id,
            f"{labels.get(c.cluster_id, '?')} · {framing(c.face_height)} · {c.presence:.0%}",
        )
        for c in k.cast
    )
    tags = ", ".join(escape(tag) for tag, _ in k.tags)
    seconds = k.shot.frame_count / ep.fps
    return (
        f'<figure id="shot-{k.shot.id}">'
        f"{_image(f'shot-{k.shot.id}.jpg', images, f'shot {k.shot.index}')}"
        f"<figcaption><div>#{k.shot.index} · {span(ep, k.shot)} ({float(seconds):.1f}s)"
        f' <span class="muted">q+e {k.max_risk:.2f}</span></div>'
        f'<div>{cast}</div><div class="tags">{tags}</div></figcaption></figure>'
    )


def _episode(ep: EpisodeReport, labels: dict[int, str], images: set[str]) -> str:
    params = ep.filter_params or {}
    settings = (
        f" · filter {escape(str(params.get('model')))}, q+e &ge; {params.get('unsafe_threshold')}"
        if params
        else ""
    )
    dropped = ", ".join(span(ep.episode, s) for s in ep.dropped)
    shots = "".join(_shot(ep.episode, k, labels, images) for k in ep.kept)
    return (
        f'<section id="ep-{ep.episode.id}"><h2>{escape(ep.episode.path)}</h2>'
        f'<p class="muted">{len(ep.kept)} kept of {ep.total} shots · {len(ep.dropped)} dropped'
        f" ({_pct(len(ep.dropped), ep.total)}){settings}</p>"
        f'<div class="shots">{shots}</div>'
        f"<details><summary>Dropped shots: time ranges only, no images</summary>"
        f'<p class="muted">{dropped or "none"}</p></details></section>'
    )


def render(data: ReportData, series_dir: Path, images: set[str], now: datetime) -> str:
    series = series_dir.name
    labels = _labels(data.clusters)
    nav = " ".join(
        f'<a href="#ep-{ep.episode.id}">{escape(ep.episode.path)}</a>' for ep in data.episodes
    )
    clusters = "".join(_cluster(c, str(series_dir), images) for c in data.clusters)
    episodes = "".join(_episode(ep, labels, images) for ep in data.episodes)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>tokimeki · {escape(series)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(series)}</h1>"
        f'<p class="muted">tokimeki report, {now:%Y-%m-%d %H:%M}</p>'
        f'<nav><a href="#totals">totals</a><a href="#characters">characters</a>{nav}</nav>'
        f'<section id="totals"><h2>Totals</h2>{_totals(data.episodes)}</section>'
        f'<section id="characters"><h2>Characters</h2>'
        f'<p class="muted">{len(data.clusters)} clusters. Name, merge or split them with'
        f" <code>tokimeki cast</code>; names carry over to later episodes.</p>"
        f'<div class="clusters">{clusters}</div></section>'
        f"{episodes}</body></html>"
    )
