"""Arranging with a language model: the prompt, the answer's schema, and checking the answer.

No provider is wired in yet (the user has not chosen an API). Any object with `complete`
can be plugged into `arrange_with_model`; `build_prompt` can be written out to see exactly
what would be sent. Only text about kept shots goes into the prompt: candidates are built
from kept shots alone.
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Protocol, cast

from tokimeki.mad.arrange import ArrangementError, Assignment, fits
from tokimeki.mad.candidates import Candidate
from tokimeki.mad.plan import PlanSlot


class TextModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


PROVIDERS: dict[str, Callable[[], TextModel]] = {}
"""Model providers by name (`--arranger model:<name>`); none configured yet."""

ARRANGEMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "integer"},
                    "shot": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["slot", "shot", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

SYSTEM = """You edit anime MADs: short videos cut to a song that show one character at her cutest.
You get the song's slots (one cut each, in order) and the candidate shots of the character, with
what the picture shows (framing, expression tags with scores, who else is there) and the lines
spoken over it. Choose a shot for every slot.

Rules:
- every slot gets exactly one shot, and no shot is used twice;
- a shot must last at least 0.9 times its slot (it may be slowed down that far, no further);
- everyday moments early, the cutest (blush, smiles, pouts, winks) in the chorus, the emotional
  ones (tears, confessions) in a bridge, and her best smile on the last slot;
- alternate close-ups with wider shots, and do not put two shots of the same scene side by side
  unless they answer each other;
- follow the editor's request below when there is one.

Answer with JSON only, matching this schema:
{schema}"""


def build_prompt(
    slots: Sequence[PlanSlot],
    candidates: Sequence[Candidate],
    character: str,
    song_title: str,
    guidance: str = "",
) -> tuple[str, str]:
    system = SYSTEM.format(schema=json.dumps(ARRANGEMENT_SCHEMA, ensure_ascii=False))
    lines = [f"Character: {character}", f"Song: {song_title}", ""]
    if guidance:
        lines += [f"Editor's request: {guidance}", ""]
    lines.append("Slots (seconds into the excerpt):")
    lines += [
        f"- slot {s.index}: {s.section}, {s.start:.2f}-{s.end:.2f} ({s.duration:.2f}s)"
        for s in slots
    ]
    lines += ["", "Candidate shots:"]
    for c in candidates:
        said = " / ".join(c.lines)[:120]
        lines.append(
            f"- shot {c.shot.id}: {c.episode.path} at {c.start:.1f}s, {c.duration:.1f}s long,"
            f" scene {c.scene}; {c.describe()}" + (f'; lines: "{said}"' if said else "")
        )
    return system, "\n".join(lines)


def _json_block(text: str) -> object:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ArrangementError(["the answer has no JSON object"])
    try:
        return cast(object, json.loads(match.group(0)))
    except json.JSONDecodeError as error:
        raise ArrangementError([f"the answer's JSON does not parse: {error}"]) from error


def parse_answer(
    text: str, slots: Sequence[PlanSlot], candidates: Sequence[Candidate]
) -> list[Assignment]:
    """The model's assignments, or `ArrangementError` listing everything wrong with them."""
    data = _json_block(text)
    entries = cast(dict[str, object], data).get("assignments") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ArrangementError(['the answer needs an "assignments" list'])
    by_id = {c.shot.id: c for c in candidates}
    by_slot = {s.index: s for s in slots}
    problems: list[str] = []
    out: list[Assignment] = []
    seen_slots: set[int] = set()
    seen_shots: set[int] = set()
    for raw in cast(list[object], entries):
        entry = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
        slot_value, shot_value = entry.get("slot"), entry.get("shot")
        if not isinstance(slot_value, int) or not isinstance(shot_value, int):
            problems.append(f"bad entry {raw!r}")
            continue
        slot, candidate = by_slot.get(slot_value), by_id.get(shot_value)
        if slot is None:
            problems.append(f"slot {slot_value} does not exist")
        elif candidate is None:
            problems.append(f"shot {shot_value} is not a candidate")
        elif slot_value in seen_slots:
            problems.append(f"slot {slot_value} is filled twice")
        elif shot_value in seen_shots:
            problems.append(f"shot {shot_value} is used twice")
        elif not fits(candidate, slot):
            problems.append(f"shot {shot_value} is too short for slot {slot_value}")
        else:
            seen_slots.add(slot_value)
            seen_shots.add(shot_value)
            out.append(Assignment(slot_value, candidate, str(entry.get("reason", ""))))
    missing = sorted(set(by_slot) - seen_slots)
    if missing:
        problems.append(f"slots left empty: {missing}")
    if problems:
        raise ArrangementError(problems)
    return sorted(out, key=lambda a: a.slot)


def arrange_with_model(
    model: TextModel,
    slots: Sequence[PlanSlot],
    candidates: Sequence[Candidate],
    character: str,
    song_title: str,
    guidance: str = "",
    attempts: int = 2,
) -> list[Assignment]:
    """Ask the model; on a bad answer, ask again once with the problems listed."""
    system, user = build_prompt(slots, candidates, character, song_title, guidance)
    problems: list[str] = []
    for _ in range(attempts):
        prompt = user
        if problems:
            prompt += "\n\nYour previous answer had these problems; fix them:\n- "
            prompt += "\n- ".join(problems)
        try:
            return parse_answer(model.complete(system, prompt), slots, candidates)
        except ArrangementError as error:
            problems = error.problems
    raise ArrangementError(problems)
