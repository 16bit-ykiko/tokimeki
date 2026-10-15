"""Original lines laid over the song: which may be used, and where their voice really is.

A line may be used only when kept shots cover all of it and it is outside the OP/ED; the
voice stem is silent elsewhere anyway. Speakers are unknown (the subtitles carry none), so
who is on screen while a line is said is listed for the arranger to judge.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from tokimeki.library.cast import named_cast_by_shot
from tokimeki.library.episodes import get_episode
from tokimeki.library.lines import get_line, list_lines, list_parts, shots_in_parts
from tokimeki.library.records import Episode, Line, LineKind, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import kept_ranges, list_shots
from tokimeki.mad.plan import Plan, VoicePlan
from tokimeki.song.analysis import SongAnalysis
from tokimeki.song.gaps import Gap, gaps
from tokimeki.stages.voice import voice_bounds

MAX_VOICE = 8.0
GAP_MARGIN = 0.3
"""Seconds kept free at each end of a gap the draft puts a line in."""


@dataclass(frozen=True, slots=True)
class VoiceLine:
    line: Line
    episode: Episode
    shots: list[int]
    """Kept shots the line overlaps."""
    on_screen: list[str]
    """Named characters seen in those shots."""


def covered(ranges: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(a <= start + 1e-6 and end <= b + 1e-6 for a, b in ranges)


def usable_lines(ctx: Context, episode: Episode) -> list[VoiceLine]:
    """Dialogue lines wholly inside kept shots and outside the OP/ED, at most `MAX_VOICE` long."""
    kept = kept_ranges(ctx.conn, episode)
    parts = list_parts(ctx.conn, episode.id)
    in_parts = shots_in_parts(ctx.conn, episode.id)
    shots = [s for s in list_shots(ctx.conn, episode.id, ShotStatus.KEPT) if s.id not in in_parts]
    cast = named_cast_by_shot(ctx.conn, episode.id)
    out: list[VoiceLine] = []
    for line in list_lines(ctx.conn, episode.id, LineKind.DIALOGUE):
        if line.end - line.start > MAX_VOICE or not covered(kept, line.start, line.end):
            continue
        if any(p.start < line.end and line.start < p.end for p in parts):
            continue
        over = [
            s.id
            for s in shots
            if episode.seconds(s.start_frame) < line.end
            and line.start < episode.seconds(s.end_frame)
        ]
        names = sorted({name for shot in over for name, _ in cast.get(shot, [])})
        out.append(VoiceLine(line, episode, over, names))
    return out


def refine_voices(ctx: Context, plan: Plan) -> tuple[list[VoicePlan], list[str]]:
    """Each voice with its episode, text and (when missing) in/out taken from its line,
    the in/out trimmed to where the voice is in the stem."""
    out: list[VoicePlan] = []
    notes: list[str] = []
    for i, v in enumerate(plan.voices):
        if v.line is None:
            out.append(v)
            continue
        try:
            line = get_line(ctx.conn, v.line)
        except KeyError:
            out.append(v)
            continue
        episode = get_episode(ctx.conn, line.episode_id)
        if v.source_in is None or v.source_out is None:
            neighbours = list_lines(ctx.conn, episode.id, LineKind.DIALOGUE)
            a, b = voice_bounds(ctx, episode, line, neighbours)
            v = replace(v, source_in=round(a, 3), source_out=round(b, 3))
            notes.append(
                f"voice {i}: line {line.id} heard {a:.2f}-{b:.2f}s"
                f" (subtitle {line.start:.2f}-{line.end:.2f}s)"
            )
        out.append(
            replace(
                v,
                episode=v.episode or episode.path,
                text=v.text or " ".join(line.text.split()),
            )
        )
    return out, notes


def _gap_order(found: Sequence[Gap]) -> list[Gap]:
    """The intro first, then the longest."""
    return sorted(found, key=lambda g: (g.where != "intro", -g.length))


def pick_voice(
    plan: Plan,
    analysis: SongAnalysis,
    lines: Sequence[VoiceLine],
    character: str,
    heard: Callable[[VoiceLine], tuple[float, float]],
    tries: int = 8,
) -> VoicePlan | None:
    """One line for the draft: in the intro gap if one fits, else the longest gap; said
    while she is on screen, best in a shot the plan shows during the gap, best alone.
    `heard` gives where a line's voice really is; the best few are measured with it."""
    for gap in _gap_order(gaps(analysis, plan.start, plan.end)):
        room = gap.length - 2 * GAP_MARGIN
        shown = {s.shot for s in plan.slots if s.start < gap.end and gap.start < s.end}

        def score(x: VoiceLine, shown: set[int | None] = shown, room: float = room) -> float:
            length = x.line.end - x.line.start
            here = 2.0 if shown & set(x.shots) else 0.0
            alone = 1.0 if x.on_screen == [character] else 0.0
            return here + alone - abs(length - min(2.5, room)) / 2

        fitting = [
            x
            for x in lines
            if character in x.on_screen and 0.6 <= x.line.end - x.line.start <= room
        ]
        for best in sorted(fitting, key=score, reverse=True)[:tries]:
            a, b = heard(best)
            if b - a > room:
                continue
            why = f"a line of hers in the {gap.where} gap ({gap.start:.2f}-{gap.end:.2f}s)"
            return VoicePlan(
                at=round(gap.start + GAP_MARGIN, 3),
                line=best.line.id,
                source_in=round(a, 3),
                source_out=round(b, 3),
                why=why,
            )
    return None
