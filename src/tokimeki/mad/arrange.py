"""The draft arranger: a first plan an agent (or person) starts from and edits.

It follows the editing rules in docs/mad.md where tags can tell: everyday moments in the verse,
the cutest close-ups in the chorus, an establishing look in the intro, her best smile to
close, no shot twice, and close-ups and wider shots alternating.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from tokimeki.mad.candidates import Candidate
from tokimeki.mad.plan import MIN_SPEED, Plan, SlotPlan
from tokimeki.mad.refine import usable
from tokimeki.song.slots import Slot

SMILES = {"smile": 1.0, ":d": 0.9, "^_^": 0.9, "laughing": 0.8, "grin": 0.5}


class ArrangementError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True, slots=True)
class Assignment:
    slot: int
    candidate: Candidate
    reason: str


def fits(candidate: Candidate, slot: Slot) -> bool:
    first, last = usable(candidate)
    return last - first >= slot.duration * MIN_SPEED


def _top_tags(candidate: Candidate, count: int = 3) -> str:
    tags = sorted(candidate.peak.tags.items(), key=lambda item: -item[1])[:count]
    return ", ".join(f"{t} {s:.2f}" for t, s in tags) or "neutral"


def _smile(candidate: Candidate) -> float:
    return sum(w * candidate.peak.tags.get(t, 0.0) for t, w in SMILES.items())


def score(candidate: Candidate, section: str, closing: bool) -> tuple[float, str]:
    """How well a shot suits a slot of a section, with the reason in words."""
    c = candidate
    cute = c.peak.cuteness
    size = max(c.face_height, c.peak.face)
    look = f"{c.framing}, {_top_tags(c)}"
    if closing:
        return 1.5 * _smile(c) + size + 0.3 * cute, f"closing on her best smile: {look}"
    if section == "chorus":
        return cute + 0.8 * size + 0.3 * c.presence, f"chorus wants the cutest: {look}"
    if section in ("pre-chorus", "bridge"):
        tears = c.peak.tags.get("tears", 0.0)
        return 0.7 * cute + 0.5 * c.presence + 0.3 * tears, f"{section} builds up: {look}"
    if section == "intro":
        wide = 0.3 if c.framing != "close-up" else 0.0
        calm = 0.3 * (1.0 - min(cute, 1.0))
        return 0.6 * c.presence + wide + calm, f"intro sets the scene: {look}"
    talking = 0.2 if c.lines else 0.0
    return 0.5 * cute + 0.5 * c.presence + talking, f"{section}: an everyday moment, {look}"


def _alternate(chosen: list[Assignment], slots: dict[int, Slot]) -> list[Assignment]:
    """Story order within a section (each slot taking the earliest shot that still fits it),
    then break up runs of three shots of the same framing."""
    places = sorted(a.slot for a in chosen)
    remaining = [
        a.candidate
        for a in sorted(chosen, key=lambda a: (a.candidate.episode.path, a.candidate.start))
    ]
    picks: list[Candidate] = []
    for place in places:
        pick = next((c for c in remaining if fits(c, slots[place])), None)
        if pick is None:
            return chosen
        remaining.remove(pick)
        picks.append(pick)
    for i in range(2, len(picks)):
        if picks[i].framing == picks[i - 1].framing == picks[i - 2].framing:
            for j in range(i + 1, len(picks)):
                if (
                    picks[j].framing != picks[i].framing
                    and fits(picks[j], slots[places[i]])
                    and fits(picks[i], slots[places[j]])
                ):
                    picks[i], picks[j] = picks[j], picks[i]
                    break
    reasons = {id(a.candidate): a.reason for a in chosen}
    return [Assignment(p, c, reasons[id(c)]) for p, c in zip(places, picks, strict=True)]


def _fill(
    order: Sequence[Slot], candidates: Sequence[Candidate], closing: int
) -> list[Assignment] | None:
    used: set[int] = set()
    chosen: list[Assignment] = []
    for slot in order:
        options = [c for c in candidates if c.shot.id not in used and fits(c, slot)]
        if not options:
            return None
        ranked = [(score(c, slot.section, slot.index == closing), c) for c in options]
        (value, reason), best = max(ranked, key=lambda r: (r[0][0], -r[1].shot.id))
        used.add(best.shot.id)
        chosen.append(Assignment(slot.index, best, f"{reason} (score {value:.2f})"))
    return chosen


def arrange(slots: Sequence[Slot], candidates: Sequence[Candidate]) -> list[Assignment]:
    """Fill every slot with a different candidate: the most demanding slots first, or, when
    that leaves a slot no shot can fill, the slots with the fewest shots that fit first."""
    if not slots:
        return []
    closing = slots[-1].index
    priority = {"chorus": 0, "pre-chorus": 1, "bridge": 1, "verse": 2, "intro": 3}
    by_priority = sorted(
        slots, key=lambda s: (s.index != closing, priority.get(s.section, 2), s.index)
    )
    by_scarcity = sorted(by_priority, key=lambda s: sum(1 for c in candidates if fits(c, s)))
    chosen = _fill(by_priority, candidates, closing) or _fill(by_scarcity, candidates, closing)
    if chosen is None:
        raise ArrangementError(["not enough shots long enough to fill every slot"])
    by_slot = {s.index: s for s in slots}
    result: list[Assignment] = []
    in_order = sorted(chosen, key=lambda a: a.slot)
    for _, group in groupby(in_order, key=lambda a: (by_slot[a.slot].section, a.slot == closing)):
        members = list(group)
        result += members if members[0].slot == closing else _alternate(members, by_slot)
    return sorted(result, key=lambda a: a.slot)


def draft(
    name: str,
    series: str,
    character: str,
    song: str,
    slots: Sequence[Slot],
    candidates: Sequence[Candidate],
) -> Plan:
    """A plan with every slot filled; windows are left to `refine`."""
    by_slot = {a.slot: a for a in arrange(slots, candidates)}
    return Plan(
        name,
        series,
        character,
        song,
        [
            SlotPlan(
                s.start,
                s.end,
                by_slot[s.index].candidate.shot.id,
                by_slot[s.index].reason,
                section=s.section,
                episode=by_slot[s.index].candidate.episode.path,
            )
            for s in slots
        ],
    )
