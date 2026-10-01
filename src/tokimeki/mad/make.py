"""Making a MAD's plan: song, excerpt and slots, candidate shots, arrangement."""

import logging
from collections.abc import Sequence
from pathlib import Path

from tokimeki.library.records import Episode
from tokimeki.mad import model_arranger
from tokimeki.mad.arrange import arrange, to_clips
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.plan import Plan, PlanSlot, Preferences, SongRef, plan_dir, problems
from tokimeki.song.analysis import SongAnalysis, analyse_song
from tokimeki.song.slots import excerpt_between, first_chorus, make_slots
from tokimeki.stages.base import Context

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
