"""Making a MAD: song, excerpt and slots, candidate shots, arrangement, placement, renders."""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tokimeki.library.records import Episode
from tokimeki.mad import model_arranger
from tokimeki.mad.arrange import arrange, to_clips
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.place import place
from tokimeki.mad.plan import Plan, PlanSlot, Preferences, SongRef, plan_dir, problems
from tokimeki.mad.render import RenderSettings, render, slot_frames
from tokimeki.mad.report import write_report
from tokimeki.models.timeline import TimelineClip, write_timeline
from tokimeki.song.analysis import SongAnalysis, analyse_song
from tokimeki.song.slots import excerpt_between, first_chorus, make_slots
from tokimeki.stages.base import Context, register_episodes

HEURISTIC = "heuristic"
log = logging.getLogger("tokimeki")


class PlanError(RuntimeError):
    pass


def plan_path(ctx: Context, name: str) -> Path:
    return plan_dir(ctx.paths.data_dir, name) / "plan.json"


def load_plan(ctx: Context, name: str) -> Plan:
    return Plan.from_json(plan_path(ctx, name).read_text(encoding="utf-8"))


def save_plan(ctx: Context, plan: Plan) -> Path:
    path = plan_path(ctx, plan.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.to_json(), encoding="utf-8")
    return path


def slots_for(
    analysis: SongAnalysis, prefs: Preferences, budget: int
) -> tuple[SongRef, list[PlanSlot]]:
    excerpt = excerpt_between(analysis, *prefs.excerpt) if prefs.excerpt else first_chorus(analysis)
    slots = make_slots(analysis, excerpt, budget, prefs.beats_per_slot)
    song = analysis.song
    ref = SongRef(
        song.id, song.title, str(song.path), song.offset, excerpt.start, excerpt.end, analysis.bpm
    )
    return ref, [PlanSlot(s.index, s.start, s.end, s.section, s.beats) for s in slots]


def gather(
    ctx: Context, prefs: Preferences, episodes: Sequence[Episode]
) -> tuple[SongRef, list[PlanSlot], list[Candidate]]:
    analysis = analyse_song(ctx.paths, Path(prefs.song).expanduser(), prefs.track)
    candidates = find_candidates(ctx, prefs.character, episodes, prefs.boost, prefs.min_presence)
    if not candidates:
        raise PlanError(f"no kept shots of {prefs.character} outside the OP/ED")
    ref, slots = slots_for(analysis, prefs, len(candidates))
    if len(slots) > len(candidates):
        raise PlanError(f"{len(slots)} slots but only {len(candidates)} shots of {prefs.character}")
    return ref, slots, candidates


def make_plan(
    ctx: Context,
    name: str,
    prefs: Preferences,
    episodes: Sequence[Episode],
    arranger: str = HEURISTIC,
) -> Plan:
    ref, slots, candidates = gather(ctx, prefs, episodes)
    if arranger == HEURISTIC:
        assignments = arrange(slots, candidates)
    else:
        provider = arranger.removeprefix("model:")
        factory = model_arranger.PROVIDERS.get(provider)
        if factory is None:
            known = ", ".join(model_arranger.PROVIDERS) or "none configured yet"
            raise PlanError(f"no model provider {provider!r} (available: {known})")
        assignments = model_arranger.arrange_with_model(
            factory(), slots, candidates, prefs.character, ref.title, prefs.guidance
        )
    plan = Plan(name, prefs, ref, slots, to_clips(assignments), arranger)
    found = problems(plan)
    if found:
        raise PlanError("; ".join(found))
    log.info(
        "%s: %d slots over %.1fs of %s, filled from %d candidate shots",
        name,
        len(slots),
        ref.duration,
        ref.title,
        len(candidates),
    )
    return plan


def write_prompt(ctx: Context, name: str, prefs: Preferences, episodes: Sequence[Episode]) -> Path:
    """Write what a model arranger would be sent, for inspection."""
    ref, slots, candidates = gather(ctx, prefs, episodes)
    system, user = model_arranger.build_prompt(
        slots, candidates, prefs.character, ref.title, prefs.guidance
    )
    path = plan_dir(ctx.paths.data_dir, name) / "prompt.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# System\n\n{system}\n\n# User\n\n{user}\n", encoding="utf-8")
    return path


def plan_candidates(ctx: Context, plan: Plan) -> dict[int, Candidate]:
    """The plan's shots as candidates; every one must be a kept shot of the library."""
    episodes = {e.path: e for e in register_episodes(ctx)}
    wanted: dict[str, list[int]] = {}
    for clip in plan.clips:
        wanted.setdefault(clip.episode, []).append(clip.shot)
    found: dict[int, Candidate] = {}
    for path, shots in wanted.items():
        episode = episodes.get(path)
        if episode is None:
            raise PlanError(f"no episode {path!r} in the series")
        prefs = plan.preferences
        for c in find_candidates(ctx, prefs.character, [episode], prefs.boost, 0.0, only=shots):
            found[c.shot.id] = c
    missing = sorted({c.shot for c in plan.clips} - set(found))
    if missing:
        raise PlanError(f"shots {missing} are not kept shots of the library; they cannot be used")
    return found


def finish(ctx: Context, plan: Plan, settings: Sequence[RenderSettings]) -> list[Path]:
    """Place any unplaced clips, render, export the timeline and write the report."""
    candidates = plan_candidates(ctx, plan)
    plan = place(plan, candidates)
    found = problems(plan)
    if found:
        raise PlanError("; ".join(found))
    save_plan(ctx, plan)
    out_dir = plan_dir(ctx.paths.data_dir, plan.name)
    outputs: list[Path] = []
    for s in settings:
        start = time.monotonic()
        outputs.append(render(ctx.paths, plan, out_dir, s))
        log.info("%s: %s rendered in %.0fs", plan.name, s.label, time.monotonic() - start)
    export_timeline(ctx, plan, out_dir / "timeline.otio")
    outputs.append(out_dir / "timeline.otio")
    outputs.append(write_report(ctx.paths, plan, out_dir, candidates))
    return outputs


def export_timeline(ctx: Context, plan: Plan, path: Path) -> None:
    rate = float(plan.fps)
    video = [
        TimelineClip(
            f"{clip.slot} {plan.slots[clip.slot].section} - shot {clip.shot}",
            ctx.paths.episode_file(clip.episode),
            clip.source_start or 0.0,
            slot_frames(plan, plan.slots[clip.slot]),
            clip.speed or 1.0,
        )
        for clip in sorted(plan.clips, key=lambda c: c.slot)
    ]
    total = sum(slot_frames(plan, s) for s in plan.slots)
    song = plan.song
    audio = [TimelineClip(song.title, Path(song.path), song.offset + song.start, total)]
    write_timeline(path, plan.name, rate, video, audio)
