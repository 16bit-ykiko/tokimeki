"""The agent-facing API: plain JSON in and out.

Each function takes paths, ids and simple values and returns a JSON-ready dict. The CLI
prints these results; an MCP server can expose the same functions as tools unchanged.
"""

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from tokimeki.library.episodes import get_episode
from tokimeki.library.records import Episode
from tokimeki.library.shots import get_shot
from tokimeki.mad.arrange import draft
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.context import build_context
from tokimeki.mad.plan import PLAN_JSON_SCHEMA, Plan, PlanFormatError, mad_dir, write_plan
from tokimeki.mad.refine import refine
from tokimeki.mad.render import FINAL, PREVIEW, export_timeline, render
from tokimeki.mad.report import write_report
from tokimeki.mad.validate import Issue, validate
from tokimeki.song.analysis import (
    SongAnalysis,
    SongError,
    analyse_song,
    load_analysis,
    suggested_slots,
)
from tokimeki.stages.base import Context, open_series, register_episodes

type Json = dict[str, object]


class ApiError(RuntimeError):
    def __init__(self, message: str, details: object = None) -> None:
        super().__init__(message)
        self.details = details


def parse_range(text: str) -> tuple[float, float]:
    """`"a-b"` in seconds or `m:ss(.x)` into song seconds."""

    def seconds(part: str) -> float:
        pieces = part.strip().split(":")
        if not 1 <= len(pieces) <= 3:
            raise ApiError(f"cannot read {part!r} as a time")
        total = 0.0
        for piece in pieces:
            total = total * 60 + float(piece)
        return total

    match = re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", text)
    if match is None:
        raise ApiError(f"range {text!r} should look like 0:58-1:21 or 58.2-81.1")
    return seconds(match.group(1)), seconds(match.group(2))


def _song_json(analysis: SongAnalysis) -> Json:
    data = cast(Json, asdict(analysis))
    data["source"] = {**cast(Json, data["source"]), "path": str(analysis.source.path)}
    data["excerpt"] = list(analysis.excerpt)
    return data


def song_analyze(
    audio: str,
    track: int | None = None,
    lyrics: str | None = None,
    instrumental: str | None = None,
    span: str | None = None,
) -> Json:
    """Beats, bars, sections, energy, the excerpt and suggested slots of a song; stored."""
    try:
        analysis = analyse_song(
            Path(audio).expanduser(),
            track,
            Path(lyrics).expanduser() if lyrics else None,
            Path(instrumental).expanduser() if instrumental else None,
            parse_range(span) if span else None,
        )
    except SongError as error:
        raise ApiError(str(error)) from error
    return _song_json(analysis)


def song_show(song: str) -> Json:
    try:
        return _song_json(load_analysis(song))
    except SongError as error:
        raise ApiError(str(error)) from error


def _series(series: str) -> Context:
    try:
        return open_series(Path(series).expanduser())
    except FileNotFoundError as error:
        raise ApiError(str(error)) from error


def _episodes(ctx: Context, only: Sequence[str] | None) -> list[Episode]:
    episodes = register_episodes(ctx)
    if only:
        episodes = [e for e in episodes if any(part in e.path for part in only)]
    if not episodes:
        raise ApiError(f"no episodes match {list(only or [])} in {ctx.paths.root}")
    return episodes


def _analysis(song: str | None) -> SongAnalysis | None:
    if not song:
        return None
    try:
        return load_analysis(song)
    except SongError as error:
        raise ApiError(str(error)) from error


def plan_context(
    series: str,
    character: str,
    episodes: Sequence[str] | None = None,
    song: str | None = None,
    min_presence: float = 0.34,
    boost: Mapping[str, float] | None = None,
) -> Json:
    """The song (if given) and every candidate scene and shot of the character."""
    ctx = _series(series)
    return build_context(
        ctx, character, _episodes(ctx, episodes), _analysis(song), min_presence, boost or {}
    )


def plan_schema() -> Json:
    return PLAN_JSON_SCHEMA


def _load_plan(plan: str | Json) -> Plan:
    try:
        if isinstance(plan, str):
            return Plan.from_json(Path(plan).expanduser().read_text(encoding="utf-8"))
        return Plan.from_dict(plan)
    except PlanFormatError as error:
        raise ApiError("the plan does not parse", error.problems) from error
    except OSError as error:
        raise ApiError(str(error)) from error


def _plan_candidates(ctx: Context, plan: Plan) -> dict[int, Candidate]:
    """The plan's kept shots as candidates (whoever is in them)."""
    by_episode: dict[int, list[int]] = {}
    for slot in plan.slots:
        if slot.shot is None:
            continue
        try:
            shot = get_shot(ctx.conn, slot.shot)
        except KeyError:
            continue
        by_episode.setdefault(shot.episode_id, []).append(slot.shot)
    found: dict[int, Candidate] = {}
    for episode_id, shots in by_episode.items():
        episode = get_episode(ctx.conn, episode_id)
        for c in find_candidates(ctx, plan.character, [episode], {}, 0.0, only=shots):
            found[c.shot.id] = c
    return found


def _report(plan: Plan, issues: list[Issue]) -> Json:
    errors = [i for i in issues if i.level == "error"]
    return {
        "ok": not errors,
        "errors": len(errors),
        "warnings": len(issues) - len(errors),
        "issues": [asdict(i) for i in issues],
        "summary": {
            "slots": len(plan.slots),
            "seconds": round(plan.end - plan.start, 3),
            "shots": len({s.shot for s in plan.slots if s.shot is not None}),
            "without_window": sum(
                1 for s in plan.slots if s.source_in is None or s.source_out is None
            ),
        },
    }


def plan_validate(plan: str | Json) -> Json:
    """Every problem with a plan, located by slot and code; `ok` when it can be rendered."""
    parsed = _load_plan(plan)
    ctx = _series(parsed.series)
    try:
        analysis = load_analysis(parsed.song) if parsed.song else None
    except SongError:
        analysis = None
    return _report(parsed, validate(ctx.conn, parsed, analysis))


def _refined(plan: Plan) -> tuple[Context, Plan, SongAnalysis, dict[int, Candidate], list[str]]:
    ctx = _series(plan.series)
    analysis = _analysis(plan.song)
    if analysis is None:
        raise ApiError("the plan names no song")
    candidates = _plan_candidates(ctx, plan)
    refined, notes = refine(plan, analysis, candidates)
    return ctx, refined, analysis, candidates, notes


def plan_refine(plan: str, output: str | None = None) -> Json:
    """Snap slot boundaries to beats and fill missing windows around expression peaks."""
    _, refined, _, _, notes = _refined(_load_plan(plan))
    target = Path(output or plan).expanduser()
    write_plan(refined, target)
    return {"path": str(target), "notes": notes, "validation": plan_validate(str(target))}


def plan_auto(
    series: str,
    character: str,
    song: str,
    name: str,
    episodes: Sequence[str] | None = None,
    min_presence: float = 0.34,
    boost: Mapping[str, float] | None = None,
    output: str | None = None,
) -> Json:
    """A complete draft plan from the heuristic arranger, refined and written to disk."""
    ctx = _series(series)
    analysis = _analysis(song)
    if analysis is None:
        raise ApiError("a song id is needed")
    candidates = find_candidates(
        ctx, character, _episodes(ctx, episodes), boost or {}, min_presence
    )
    if not candidates:
        raise ApiError(f"no kept shots of {character} outside the OP/ED")
    slots = suggested_slots(analysis, len(candidates))
    if len(slots) > len(candidates):
        raise ApiError(f"{len(slots)} slots but only {len(candidates)} shots; pick a shorter range")
    plan = draft(name, str(ctx.paths.root), character, analysis.id, slots, candidates)
    refined, _ = refine(plan, analysis, {c.shot.id: c for c in candidates})
    target = Path(output).expanduser() if output else mad_dir(ctx.paths.root, name) / "plan.json"
    write_plan(refined, target)
    return {
        "path": str(target),
        "plan": refined.to_dict(),
        "validation": plan_validate(str(target)),
    }


def render_plan(plan: str, quality: str = "preview", otio: bool = False) -> Json:
    """Render a valid plan (refining missing windows first) into the series' data dir."""
    parsed = _load_plan(plan)
    report = plan_validate(parsed.to_dict())
    if not report["ok"]:
        raise ApiError("the plan has errors; fix them first", report)
    ctx, refined, analysis, candidates, notes = _refined(parsed)
    out_dir = mad_dir(ctx.paths.root, refined.name)
    write_plan(refined, out_dir / "plan.json")
    episodes = {shot: c.episode.path for shot, c in candidates.items()}
    settings = {"preview": [PREVIEW], "final": [FINAL], "both": [PREVIEW, FINAL]}.get(quality)
    if settings is None:
        raise ApiError(f"quality must be preview, final or both, not {quality!r}")
    outputs: list[str] = []
    timings: dict[str, float] = {}
    for s in settings:
        start = time.monotonic()
        outputs.append(str(render(ctx.paths, refined, episodes, analysis, out_dir, s)))
        timings[s.label] = round(time.monotonic() - start, 1)
    if otio:
        export_timeline(ctx.paths, refined, episodes, analysis, out_dir / "timeline.otio")
        outputs.append(str(out_dir / "timeline.otio"))
    outputs.append(str(write_report(ctx.paths, refined, analysis.title, out_dir, candidates)))
    return {
        "outputs": outputs,
        "seconds": timings,
        "refined": notes,
        "plan": str(out_dir / "plan.json"),
    }
