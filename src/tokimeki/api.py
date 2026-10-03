"""The agent-facing API: plain JSON in and out.

Each function takes paths, ids and simple values and returns a JSON-ready dict. The CLI
prints these results; an MCP server can expose the same functions as tools unchanged.
"""

import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from tokimeki.library.episodes import find_episode, get_episode, stage_done
from tokimeki.library.lines import list_lines
from tokimeki.library.records import Episode, LineKind
from tokimeki.library.shots import get_shot
from tokimeki.mad.arrange import draft
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.context import build_context
from tokimeki.mad.fonts import FontError, prepare_fonts
from tokimeki.mad.motion import ShotMotion, motions
from tokimeki.mad.plan import (
    PLAN_JSON_SCHEMA,
    Plan,
    PlanFormatError,
    VoicePlan,
    mad_dir,
    write_plan,
)
from tokimeki.mad.refine import refine
from tokimeki.mad.render import FINAL, PREVIEW, export_timeline, render, render_audio
from tokimeki.mad.report import write_report
from tokimeki.mad.subtitles import write_subtitles
from tokimeki.mad.validate import Issue, validate
from tokimeki.mad.voices import pick_voice, refine_voices, usable_lines, voice_bounds
from tokimeki.media.audio import AudioError
from tokimeki.song.analysis import (
    SongAnalysis,
    SongError,
    analyse_song,
    load_analysis,
    slots_over,
    with_accents,
)
from tokimeki.song.lyrics import LyricSource, pair
from tokimeki.stages import motion, voice
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
    lyrics: Sequence[str] = (),
    instrumental: str | None = None,
    span: str | None = None,
    instrumental_track: int | None = None,
    stream: int | None = None,
    within: str | None = None,
    title: str | None = None,
) -> Json:
    """Beats, bars, sections, energy, lyrics, the excerpt and suggested slots of a song; stored.

    `audio` is any media file: an audio file, a `.cue` image (`track`), or e.g. an episode
    with `within` a rough window around the song (its exact edges are found from the audio)
    on `stream`. `lyrics` are `PATH[#STYLE,STYLE][@LANG]` specs (LRC, or ASS lines of those
    styles on the media's clock). The vocal line comes from the first lyrics, else an
    instrumental hint, else from separating the mix. `span` picks the excerpt.
    """
    try:
        analysis = analyse_song(
            Path(audio).expanduser(),
            track,
            [LyricSource.parse(x) for x in lyrics],
            Path(instrumental).expanduser() if instrumental else None,
            parse_range(span) if span else None,
            instrumental_track,
            stream,
            parse_range(within) if within else None,
            title,
        )
    except (SongError, OSError, ValueError) as error:
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
        return with_accents(load_analysis(song))
    except (SongError, AudioError) as error:
        raise ApiError(str(error)) from error


def plan_context(
    series: str,
    character: str,
    episodes: Sequence[str] | None = None,
    song: str | None = None,
    min_presence: float = 0.34,
    boost: Mapping[str, float] | None = None,
    span: str | None = None,
    max_slots: int | None = None,
    beats: Mapping[str, int] | None = None,
) -> Json:
    """The song (if given; its slots over `span`, at most `max_slots`, `beats` per section
    kind when set) and every candidate scene and shot of the character."""
    ctx = _series(series)
    return build_context(
        ctx,
        character,
        _episodes(ctx, episodes),
        _analysis(song),
        min_presence,
        boost or {},
        parse_range(span) if span else None,
        max_slots,
        beats,
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


def _motion(ctx: Context, candidates: Iterable[Candidate]) -> Mapping[int, ShotMotion]:
    _ensure_motion(ctx, list({c.episode.id: c.episode for c in candidates}.values()))
    return motions(ctx.paths, [(c.episode, c.shot) for c in candidates])


def _ensure_motion(ctx: Context, episodes: Sequence[Episode]) -> None:
    """Measure the motion of episodes that have not been yet (the `motion` stage, GPU)."""
    todo = [e for e in episodes if not stage_done(ctx.conn, e.id, motion.NAME)]
    if todo:
        motion.run(ctx, todo)


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
        analysis = with_accents(load_analysis(parsed.song)) if parsed.song else None
    except (SongError, AudioError):
        analysis = None
    return _report(parsed, validate(ctx.conn, parsed, analysis))


def _refined(plan: Plan) -> tuple[Context, Plan, SongAnalysis, dict[int, Candidate], list[str]]:
    ctx = _series(plan.series)
    analysis = _analysis(plan.song)
    if analysis is None:
        raise ApiError("the plan names no song")
    candidates = _plan_candidates(ctx, plan)
    refined, notes = refine(plan, analysis, candidates, _motion(ctx, candidates.values()))
    voices, heard = refine_voices(ctx, refined)
    return ctx, replace(refined, voices=voices), analysis, candidates, notes + heard


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
    span: str | None = None,
    max_slots: int | None = None,
    beats: Mapping[str, int] | None = None,
    voices: bool = True,
) -> Json:
    """A complete draft plan from the heuristic arranger, refined and written to disk.

    `span` picks the excerpt (song seconds, snapped to bar lines); `max_slots` caps the cuts,
    so that only the best shots are needed; `beats` fixes beats per slot for section kinds.
    With `voices`, one of her lines goes into the intro gap if one fits (else the longest).
    """
    ctx = _series(series)
    analysis = _analysis(song)
    if analysis is None:
        raise ApiError("a song id is needed")
    chosen = _episodes(ctx, episodes)
    candidates = find_candidates(ctx, character, chosen, boost or {}, min_presence)
    if not candidates:
        raise ApiError(f"no kept shots of {character} outside the OP/ED")
    budget = min(len(candidates), max_slots) if max_slots else len(candidates)
    try:
        _, slots = slots_over(analysis, parse_range(span) if span else None, budget, beats)
    except SongError as error:
        raise ApiError(str(error)) from error
    if len(slots) > len(candidates):
        raise ApiError(f"{len(slots)} slots but only {len(candidates)} shots; pick a shorter range")
    plan = draft(name, str(ctx.paths.root), character, analysis.id, slots, candidates)
    by_id = {c.shot.id: c for c in candidates}
    refined, notes = refine(plan, analysis, by_id, _motion(ctx, candidates))
    if voices:
        picked = _draft_voice(ctx, refined, analysis, chosen, character)
        if picked is not None:
            spoken, heard = refine_voices(ctx, replace(refined, voices=[picked]))
            refined = replace(refined, voices=spoken)
            notes += heard
    target = Path(output).expanduser() if output else mad_dir(ctx.paths.root, name) / "plan.json"
    write_plan(refined, target)
    return {
        "path": str(target),
        "plan": refined.to_dict(),
        "refined": notes,
        "validation": plan_validate(str(target)),
    }


def _ensure_stems(ctx: Context, episodes: Sequence[Episode]) -> None:
    """Separate the voice of episodes that have not been yet (the `voice` stage, GPU)."""
    todo = [e for e in episodes if not stage_done(ctx.conn, e.id, voice.NAME)]
    if todo:
        voice.run(ctx, todo)


def _draft_voice(
    ctx: Context, plan: Plan, analysis: SongAnalysis, episodes: Sequence[Episode], character: str
) -> VoicePlan | None:
    _ensure_stems(ctx, episodes)
    lines = [x for e in episodes for x in usable_lines(ctx, e)]
    dialogue = {e.id: list_lines(ctx.conn, e.id, LineKind.DIALOGUE) for e in episodes}
    return pick_voice(
        plan,
        analysis,
        lines,
        character,
        lambda x: voice_bounds(ctx, x.episode, x.line, dialogue[x.episode.id]),
    )


def _voice_stems(ctx: Context, plan: Plan) -> list[tuple[VoicePlan, Path]]:
    found: dict[str, Episode] = {}
    for v in plan.voices:
        episode = find_episode(ctx.conn, v.episode)
        if episode is None:
            raise ApiError(f"voice at {v.at:.2f}s: no episode {v.episode!r}")
        found[v.episode] = episode
    _ensure_stems(ctx, list(found.values()))
    return [(v, voice.stem_path(ctx.paths, found[v.episode].id)) for v in plan.voices]


def render_plan(plan: str, quality: str = "preview", otio: bool = False, subs: str = "all") -> Json:
    """Render a valid plan (refining missing windows first) into the series' data dir.

    The song is mixed with the plan's voices (`mix.flac`). `subs` is "all" (lyrics and the
    voices' lines burnt in), "lyrics" (lyrics only) or "none"; an editable `subtitles.ass`
    and `subtitles.srt` with everything are written next to the video either way.
    """
    if subs not in ("all", "lyrics", "none"):
        raise ApiError(f"subs must be all, lyrics or none, not {subs!r}")
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
    stems = _voice_stems(ctx, refined)
    audio = render_audio(refined, analysis, stems, out_dir / "mix.flac")
    outputs: list[str] = [str(audio)]
    timings: dict[str, float] = {}
    subtitles: tuple[Path, Path] | None = None
    spoken = [(v.at, v.end, v.text) for v in refined.voices if v.sub and v.end is not None]
    if analysis.lyrics or spoken:
        try:
            fonts = prepare_fonts(ctx.paths.data_dir / "fonts")
        except FontError as error:
            raise ApiError(str(error)) from error
        pairs = pair(analysis.lyrics)
        title = refined.name
        ass = write_subtitles(out_dir, pairs, spoken, refined.start, refined.end, fonts, title)
        if ass is not None:
            outputs += [str(ass), str(ass.with_suffix(".srt"))]
            if subs != "lyrics":
                for stale in ("lyrics.ass", "lyrics.srt"):
                    (out_dir / stale).unlink(missing_ok=True)
            if subs == "lyrics":
                ass = write_subtitles(
                    out_dir, pairs, [], refined.start, refined.end, fonts, title, "lyrics"
                )
            if subs != "none" and ass is not None:
                subtitles = (ass, fonts.directory)
    for s in settings:
        start = time.monotonic()
        outputs.append(str(render(ctx.paths, refined, episodes, audio, out_dir, s, subtitles)))
        timings[s.label] = round(time.monotonic() - start, 1)
    if otio:
        export_timeline(ctx.paths, refined, episodes, analysis, out_dir / "timeline.otio", stems)
        outputs.append(str(out_dir / "timeline.otio"))
    outputs.append(str(write_report(ctx.paths, refined, analysis.title, out_dir, candidates)))
    return {
        "outputs": outputs,
        "seconds": timings,
        "refined": notes,
        "plan": str(out_dir / "plan.json"),
    }
