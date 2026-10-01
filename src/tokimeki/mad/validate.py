"""Checking a plan against the song and the library, with precise, located issues.

Errors stop a render; warnings are worth a look. Every issue names its slot and a stable
code, so an agent can fix plans mechanically.
"""

import sqlite3
from dataclasses import dataclass
from fractions import Fraction

from tokimeki.library.cast import named_cast_by_shot
from tokimeki.library.episodes import get_episode
from tokimeki.library.lines import shots_in_parts
from tokimeki.library.records import Episode, Shot, ShotStatus
from tokimeki.library.shots import get_shot
from tokimeki.mad.plan import MAX_SPEED, MIN_SPEED, Plan
from tokimeki.mad.refine import beat_grid, edge, nearest
from tokimeki.song.analysis import SongAnalysis

BEAT_TOLERANCE = 1 / 48
"""Half a frame at 24 fps: how far a slot boundary may sit from a beat."""
MIN_SLOT = 0.25
SHORT_SLOT = 0.5
LONG_SLOT = 6.0
SPEED_AGREEMENT = 0.01


@dataclass(frozen=True, slots=True)
class Issue:
    level: str
    """"error" or "warning"."""
    code: str
    slot: int | None
    message: str


def _error(code: str, slot: int | None, message: str) -> Issue:
    return Issue("error", code, slot, message)


def _warning(code: str, slot: int | None, message: str) -> Issue:
    return Issue("warning", code, slot, message)


def _timing(plan: Plan, analysis: SongAnalysis) -> list[Issue]:
    issues: list[Issue] = []
    grid = beat_grid(analysis)
    for i, slot in enumerate(plan.slots):
        if slot.end <= slot.start:
            issues.append(
                _error(
                    "slot.order", i, f"ends ({slot.end:.3f}) before it starts ({slot.start:.3f})"
                )
            )
            continue
        if i and abs(slot.start - plan.slots[i - 1].end) > 1e-3:
            gap = slot.start - plan.slots[i - 1].end
            kind = "slot.gap" if gap > 0 else "slot.overlap"
            issues.append(_error(kind, i, f"starts {gap:+.3f}s from where slot {i - 1} ends"))
        if slot.start < -1e-6 or slot.end > analysis.source.duration + 1e-3:
            issues.append(
                _error("song.range", i, f"outside the song (0-{analysis.source.duration:.2f}s)")
            )
        for name, t in (("start", slot.start), ("end", slot.end)):
            beat = nearest(grid, t)
            if abs(beat - t) > BEAT_TOLERANCE:
                issues.append(
                    _error(
                        "slot.off_beat",
                        i,
                        f"{name} {t:.3f}s is {t - beat:+.3f}s from the nearest beat ({beat:.3f}s)",
                    )
                )
        if slot.duration < MIN_SLOT:
            issues.append(
                _error("slot.too_short", i, f"{slot.duration:.2f}s is shorter than {MIN_SLOT}s")
            )
        elif slot.duration < SHORT_SLOT:
            issues.append(_warning("slot.short", i, f"{slot.duration:.2f}s flashes by"))
        elif slot.duration > LONG_SLOT:
            issues.append(_warning("slot.long", i, f"{slot.duration:.2f}s is long for one shot"))
    return issues


def _shot_issues(conn: sqlite3.Connection, plan: Plan) -> list[Issue]:
    issues: list[Issue] = []
    episodes: dict[int, Episode] = {}
    parts: dict[int, dict[int, str]] = {}
    cast: dict[int, set[str]] = {}
    seen: dict[int, int] = {}
    for i, slot in enumerate(plan.slots):
        if slot.shot is None:
            issues.append(_error("shot.missing", i, "no shot chosen"))
            continue
        if slot.shot in seen:
            issues.append(
                _error(
                    "shot.repeat", i, f"shot {slot.shot} is already used in slot {seen[slot.shot]}"
                )
            )
        seen.setdefault(slot.shot, i)
        try:
            shot: Shot = get_shot(conn, slot.shot)
        except KeyError:
            issues.append(_error("shot.unknown", i, f"shot {slot.shot} is not in the library"))
            continue
        if shot.status is not ShotStatus.KEPT:
            issues.append(
                _error(
                    "shot.not_kept",
                    i,
                    f"shot {slot.shot} is {shot.status.value}; only kept shots may be used",
                )
            )
            continue
        if shot.episode_id not in episodes:
            episodes[shot.episode_id] = get_episode(conn, shot.episode_id)
            parts[shot.episode_id] = {
                k: v.value for k, v in shots_in_parts(conn, shot.episode_id).items()
            }
            for shot_id, named in named_cast_by_shot(conn, shot.episode_id).items():
                cast[shot_id] = {name for name, _ in named}
        episode = episodes[shot.episode_id]
        if slot.shot in parts[shot.episode_id]:
            kind = "opening" if parts[shot.episode_id][slot.shot] == "op" else "ending"
            issues.append(_warning("shot.in_op_ed", i, f"shot {slot.shot} is part of the {kind}"))
        if plan.character and plan.character not in cast.get(slot.shot, set()):
            issues.append(
                _warning(
                    "shot.without_character", i, f"{plan.character} is not seen in shot {slot.shot}"
                )
            )
        if slot.episode and slot.episode != episode.path:
            issues.append(
                _error(
                    "shot.episode", i, f"shot {slot.shot} is in {episode.path}, not {slot.episode}"
                )
            )
        start, end = episode.seconds(shot.start_frame), episode.seconds(shot.end_frame)
        issues += _window(
            i,
            slot.source_in,
            slot.source_out,
            slot.speed,
            slot.duration,
            start,
            end,
            episode.fps,
        )
    return issues


def _window(
    i: int,
    source_in: float | None,
    source_out: float | None,
    speed: float | None,
    duration: float,
    start: float,
    end: float,
    fps: Fraction,
) -> list[Issue]:
    if source_in is None or source_out is None:
        return [_warning("window.missing", i, "no in/out yet; `plan refine` or render fills it")]
    issues: list[Issue] = []
    frame, margin = float(1 / fps), edge(fps)
    shot = f"shot spans {start:.3f}-{end:.3f}s"
    if source_out <= source_in:
        return [
            _error("window.order", i, f"out ({source_out:.3f}) is not after in ({source_in:.3f})")
        ]
    if source_in < start - frame / 2 or source_out > end + frame / 2:
        issues.append(
            _error(
                "window.outside",
                i,
                f"in/out {source_in:.3f}-{source_out:.3f}s leave the shot ({shot})",
            )
        )
    elif source_in < start + margin - 1e-3 or source_out > end - margin + 1e-3:
        issues.append(
            _warning(
                "window.edge",
                i,
                f"in/out touch the shot's first or last frames ({shot}), where transitions sit",
            )
        )
    if duration > 0:
        derived = (source_out - source_in) / duration
        if not MIN_SPEED - 1e-6 <= derived <= MAX_SPEED + 1e-6:
            issues.append(
                _error(
                    "speed.bounds",
                    i,
                    f"plays at {derived:.3f}x; allowed {MIN_SPEED}-{MAX_SPEED}x"
                    f" (out - in should be {duration * MIN_SPEED:.3f}"
                    f"-{duration * MAX_SPEED:.3f}s)",
                )
            )
        if speed is not None and abs(speed - derived) > SPEED_AGREEMENT:
            issues.append(
                _error(
                    "speed.mismatch", i, f"speed {speed:.3f} does not match in/out ({derived:.3f})"
                )
            )
    return issues


def validate(conn: sqlite3.Connection, plan: Plan, analysis: SongAnalysis | None) -> list[Issue]:
    issues: list[Issue] = []
    if not plan.slots:
        return [_error("slots.empty", None, "the plan has no slots")]
    if analysis is None:
        issues.append(_error("song.unknown", None, f"song {plan.song!r} has not been analysed"))
    else:
        issues += _timing(plan, analysis)
    return issues + _shot_issues(conn, plan)
