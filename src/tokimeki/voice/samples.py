"""A small sample set to judge her voice model by ear: some of her real lines said again by
the fine-tuned model and by the zero-shot baseline next to the original, and new lines."""

import json
import shutil
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

from tokimeki.library.series import Context
from tokimeki.voice.bank import BankLine, bank_lines
from tokimeki.voice.paths import bank_dir, samples_dir, voice_dir
from tokimeki.voice.tts import REF_MAX, Reference, Utterance, reference, speak

MODELS = ("fine-tuned", "zero-shot")


@dataclass(frozen=True, slots=True)
class NewLine:
    text: str
    style: str | None


def parse_new(spec: str) -> NewLine:
    """`STYLE:text`, or just `text` for her typical voice."""
    style, sep, text = spec.partition(":")
    if sep and style and all(c.isascii() for c in style) and " " not in style:
        return NewLine(text.strip(), style)
    return NewLine(spec.strip(), None)


def real_picks(lines: list[BankLine], ref: Reference, count: int) -> list[BankLine]:
    """Lines spread over her bank, a few seconds long, other than the reference."""
    fitting = [x for x in lines if 1.5 <= x.seconds <= REF_MAX and x.line != ref.line]
    if len(fitting) <= count:
        return fitting
    step = len(fitting) / count
    return [fitting[int(i * step)] for i in range(count)]


@dataclass(frozen=True, slots=True)
class Row:
    text: str
    kind: str
    style: str
    original: str | None
    files: dict[str, str]


def render_samples(
    ctx: Context, character: str, new: list[NewLine], real: int, zero_shot: bool = True
) -> Path:
    """Write the sample WAVs and their page into the character's samples directory."""
    root = samples_dir(ctx.paths, character)
    root.mkdir(parents=True, exist_ok=True)
    lines = bank_lines(ctx.paths, character)
    typical = reference(ctx, character, None, None)
    rows: list[Row] = []
    jobs: list[tuple[str, Utterance]] = []
    for x in real_picks(lines, typical, real):
        original = f"real/{x.line}.wav"
        (root / "real").mkdir(exist_ok=True)
        shutil.copyfile(bank_dir(ctx.paths, character) / "wavs" / x.wav, root / original)
        files = {m: f"{m}/real-{x.line}.wav" for m in MODELS}
        rows.append(Row(x.text, "real (heard in training)", "typical", original, files))
        jobs += [(m, Utterance(x.text, root / files[m], typical)) for m in MODELS]
    for i, n in enumerate(new):
        ref = reference(ctx, character, n.style, None) if n.style else typical
        files = {m: f"{m}/new-{i:02d}.wav" for m in MODELS}
        rows.append(Row(n.text, "new", ref.style, None, files))
        jobs += [(m, Utterance(n.text, root / files[m], ref)) for m in MODELS]
    for model in MODELS:
        if model == "zero-shot" and not zero_shot:
            continue
        todo = [u for m, u in jobs if m == model]
        speak(ctx, character, todo, zero_shot=model == "zero-shot")
    page = write_page(character, typical, rows, root)
    write_index(voice_dir(ctx.paths))
    return page


def _audio(path: str | None, base: Path) -> str:
    if path is None or not (base / path).exists():
        return "<td class=muted>-</td>"
    return f'<td><audio controls preload="none" src="{escape(path)}"></audio></td>'


def _covers(root: Path) -> str:
    """Rows for the covers rendered for this character (next to `root`, see `cover`)."""
    rows = ""
    for report in sorted(root.parent.glob("covers/*/cover.json")):
        data: dict[str, object] = json.loads(report.read_text())
        name = report.parent.name
        files = [f"../covers/{name}/{f}" for f in ("original-vocal.wav", "vocal.wav", "cover.flac")]
        rows += (
            f"<tr><td>{escape(name)}<br><span class=muted>{escape(str(data['model']))}</span></td>"
            + "".join(_audio(f, root) for f in files)
            + "</tr>"
        )
    if not rows:
        return ""
    head = "".join(f"<th>{h}</th>" for h in ("song", "original vocal", "her vocal", "her cover"))
    return f"<h2>Singing (Seed-VC)</h2><table><tr>{head}</tr>{rows}</table>"


def write_index(voice: Path) -> Path:
    """The front page of everything voice-related that is served: speakers, then samples."""
    links = (
        ['<li><a href="page/">Speakers: who says each line</a></li>']
        if (voice / "page" / "index.html").exists()
        else []
    )
    for samples in sorted(voice.glob("*/samples/index.html")):
        name = samples.parent.parent.name
        links.append(f'<li><a href="{escape(name)}/samples/">{escape(name)}: her voice</a></li>')
    path = voice / "index.html"
    path.write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>voices</title>'
        "<style>:root{color-scheme:light dark} body{font:16px/1.6 system-ui,sans-serif;"
        f"margin:2rem}}</style><h1>Voices</h1><ul>{''.join(links)}</ul>"
        "<p>Personal use only: nothing here is to be shared.</p></html>"
    )
    return path


def write_page(character: str, ref: Reference, rows: list[Row], root: Path) -> Path:
    page = root / "index.html"
    page.write_text(_page(character, ref, rows, root))
    (root / "rows.json").write_text(
        json.dumps(
            {"reference": ref.line, "rows": [asdict(r) for r in rows]}, ensure_ascii=False, indent=1
        )
    )
    return page


def _page(character: str, ref: Reference, rows: list[Row], base: Path) -> str:
    body = "".join(
        f"<tr><td>{escape(r.text)}<br><span class=muted>{escape(r.kind)}, {escape(r.style)}"
        f"</span></td>{_audio(r.original, base)}"
        + "".join(_audio(r.files.get(m), base) for m in MODELS)
        + "</tr>"
        for r in rows
    )
    head = "".join(f"<th>{m}</th>" for m in ("line", "original", *MODELS))
    return (
        '<!doctype html><html lang="ja"><meta charset="utf-8">'
        f"<title>{escape(character)}</title><style>"
        ":root{color-scheme:light dark} body{font:14px/1.45 system-ui,sans-serif;margin:1rem}"
        "td{padding:.3rem .6rem;vertical-align:top;border-top:1px solid #8884}"
        "audio{width:240px;height:32px} .muted{color:#888}</style>"
        f"<h1>{escape(character)}</h1><p>GPT-SoVITS v2ProPlus. Typical reference:"
        f" line {ref.line}, 「{escape(ref.text)}」. Personal use only: do not share.</p>"
        f"<h2>Speaking (GPT-SoVITS)</h2><table><tr>{head}</tr>{body}</table>"
        f"{_covers(base)}</html>"
    )
