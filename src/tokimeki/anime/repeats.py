"""Shots that recur across episodes: the opening and ending, title cards and eyecatches,
next-episode previews and recaps.

Every cached frame of a kept shot gets a perceptual hash (cached per episode). A shot recurs
when at least half of its frames have a near-identical frame in another episode. Where it sits
in its own episode, and where the shots it matches sit in theirs, say what it is:

- inside an OP or ED (the ranges its lyric styles mark), or within `NEAR` seconds of one when
  most of what it matches is next to the same part elsewhere (a title card, the closing logo):
  that OP or ED;
- after the ED: a preview of the next episode;
- in the story, matching the story of two other episodes or more: series furniture (an
  eyecatch); matching the story of an earlier episode: a recap or flashback; matching only
  later episodes, or only their OP, ED or preview: the original, which stays unmarked.

A kept shot between two shots of one kind in its episode joins them. Nothing is deleted. The
marks are derived from the frames and the OP/ED ranges, so they live in the cache
(`cache/repeats.json`) and are recomputed when either changes. Telling a preview from a recap
needs the lyric-marked ED.
"""

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.episodes import list_episodes
from tokimeki.library.lines import list_parts
from tokimeki.library.paths import SeriesPaths
from tokimeki.library.records import Episode, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import list_episode_frames, list_shots
from tokimeki.media.images import load_image
from tokimeki.media.phash import distances, phash

log = logging.getLogger("tokimeki")

VERSION = 3
SAME = 8
"""Most bits (of 64) two hashes of the same drawing differ by; different drawings differ by 12
or more."""
FLAT = 0.04
"""Frames whose grey thumbnail varies less than this (black, white, a plain sky) match any
other flat frame, so they are not compared."""
SHARE = 0.5
"""A shot recurs when this share of its compared frames is found in another episode."""
FURNITURE = 3
"""Episodes a drawing must be told in to be series furniture (an eyecatch) rather than a
recap."""
NEAR = 10.0
"""Seconds around a sung OP or ED where its unsung shots (a title card, the closing logo) are."""
THUMB = 160
KINDS = ("op", "ed", "recurring", "preview", "recap")

type Json = dict[str, object]


@dataclass(frozen=True, slots=True)
class Repeat:
    kind: str
    """One of `KINDS`."""
    like: tuple[int, ...]
    """The shots of other episodes it matches."""


def hashes_path(paths: SeriesPaths, episode_id: int) -> Path:
    return paths.cache / "phash" / f"{episode_id}.npz"


def marks_path(paths: SeriesPaths) -> Path:
    return paths.cache / "repeats.json"


@dataclass(frozen=True, slots=True)
class Hashes:
    shots: NDArray[np.int64]
    frames: NDArray[np.int64]
    hashes: NDArray[np.uint64]
    spread: NDArray[np.float64]
    stamp: str


def _kept_frames(ctx: Context, episode: Episode) -> list[tuple[int, int]]:
    """`(shot id, frame index)` of the episode's sampled frames, which only kept shots own."""
    return [(f.shot_id, f.frame_index) for f in list_episode_frames(ctx.conn, episode.id)]


def _stamp(frames: Sequence[tuple[int, int]]) -> str:
    return hashlib.sha1(json.dumps(list(frames)).encode()).hexdigest()[:16]


def episode_hashes(ctx: Context, episode: Episode) -> Hashes:
    """The hashes of the episode's cached frames, computed once while its frames stay the same."""
    frames = _kept_frames(ctx, episode)
    stamp = _stamp(frames)
    path = hashes_path(ctx.paths, episode.id)
    if path.exists():
        with np.load(path) as stored:
            if str(stored["stamp"]) == stamp:
                return Hashes(
                    stored["shots"], stored["frames"], stored["hashes"], stored["spread"], stamp
                )
    shots: list[int] = []
    indices: list[int] = []
    values: list[int] = []
    spreads: list[float] = []
    for shot_id, index in frames:
        image = ctx.paths.frame_path(episode.id, index)
        if not image.exists():
            continue
        value, spread = phash(load_image(image, THUMB))
        shots.append(shot_id)
        indices.append(index)
        values.append(value)
        spreads.append(spread)
    found = Hashes(
        np.array(shots, dtype=np.int64),
        np.array(indices, dtype=np.int64),
        np.array(values, dtype=np.uint64),
        np.array(spreads, dtype=np.float64),
        stamp,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.npz")
    np.savez(
        partial,
        shots=found.shots,
        frames=found.frames,
        hashes=found.hashes,
        spread=found.spread,
        stamp=np.array(stamp),
    )
    partial.replace(path)
    log.info("%s: hashed %d frames", episode.path, len(values))
    return found


def _matches(hashes: Mapping[int, Hashes]) -> dict[int, set[int]]:
    """Per recurring shot, the shots of other episodes its frames are found in."""
    ids = sorted(hashes)
    shot = np.concatenate([hashes[e].shots for e in ids] or [np.zeros(0, np.int64)])
    value = np.concatenate([hashes[e].hashes for e in ids] or [np.zeros(0, np.uint64)])
    spread = np.concatenate([hashes[e].spread for e in ids] or [np.zeros(0)])
    episode = np.concatenate([np.full(len(hashes[e].shots), e) for e in ids] or [np.zeros(0)])
    usable = np.flatnonzero(spread >= FLAT)
    shot, value, episode = shot[usable], value[usable], episode[usable]
    found: dict[int, set[int]] = {}
    for e in ids:
        rows = np.flatnonzero(episode == e)
        others = np.flatnonzero(episode != e)
        if len(rows) == 0 or len(others) == 0:
            continue
        near = distances(value[rows], value[others]) <= SAME
        hit = near.any(axis=1)
        for s in np.unique(shot[rows]):
            mine = shot[rows] == s
            if hit[mine].mean() >= SHARE:
                partners = shot[others][near[mine].any(axis=0)]
                found[int(s)] = {int(p) for p in partners}
    return found


def find_repeats(ctx: Context, episodes: Sequence[Episode]) -> dict[int, Repeat]:
    """Every recurring shot of `episodes`, by shot id, with its kind."""
    hashes = {e.id: episode_hashes(ctx, e) for e in episodes}
    linked = _matches(hashes)
    order = {e.id: i for i, e in enumerate(sorted(episodes, key=lambda e: e.path))}
    where: dict[int, tuple[Episode, float, float]] = {}
    shots_of: dict[int, list[int]] = {}
    for e in episodes:
        kept = list_shots(ctx.conn, e.id, ShotStatus.KEPT)
        shots_of[e.id] = [s.id for s in kept]
        for s in kept:
            where[s.id] = (e, e.seconds(s.start_frame), e.seconds(s.end_frame))
    parts = {e.id: list_parts(ctx.conn, e.id) for e in episodes}

    def part(shot: int, margin: float) -> str:
        e, start, end = where[shot]
        return next(
            (
                p.kind.value
                for p in parts[e.id]
                if p.start - margin < end and start < p.end + margin
            ),
            "",
        )

    def region(shot: int) -> str:
        """`op`/`ed` inside the sung part, or next to it when most shots it matches are too
        (a title card, the closing logo); `preview` after the ED; else `body`."""
        inside = part(shot, 0.0)
        if inside:
            return inside
        near = part(shot, NEAR)
        others = [p for p in linked.get(shot, ()) if p in where]
        if near and 2 * sum(part(p, NEAR) == near for p in others) > len(others):
            return near
        e, start, _ = where[shot]
        if any(p.kind.value == "ed" and start >= p.end - 1e-6 for p in parts[e.id]):
            return "preview"
        return "body"

    marks: dict[int, Repeat] = {}
    for s, partners in linked.items():
        if s not in where:
            continue
        kind = region(s)
        if kind == "body":
            told = [where[p][0].id for p in partners if p in where and region(p) == "body"]
            if len(set(told)) >= FURNITURE - 1:
                kind = "recurring"
            elif any(order[e] < order[where[s][0].id] for e in told):
                kind = "recap"
            else:
                continue
        marks[s] = Repeat(kind, tuple(sorted(partners)))
    for e in episodes:
        kept = shots_of[e.id]
        for before, shot, after in zip(kept, kept[1:], kept[2:], strict=False):
            if shot in marks or before not in marks or after not in marks:
                continue
            if marks[before].kind == marks[after].kind:
                marks[shot] = Repeat(marks[before].kind, ())
    return marks


def _to_json(stamps: Mapping[int, str], marks: Mapping[int, Repeat]) -> Json:
    return {
        "version": VERSION,
        "episodes": {str(e): s for e, s in sorted(stamps.items())},
        "shots": {str(s): {"kind": r.kind, "like": list(r.like)} for s, r in sorted(marks.items())},
    }


def _from_json(data: Json) -> dict[int, Repeat]:
    shots = cast(dict[str, Json], data.get("shots", {}))
    return {
        int(s): Repeat(str(r["kind"]), tuple(cast(list[int], r["like"]))) for s, r in shots.items()
    }


def _episode_stamp(ctx: Context, episode: Episode, frames: Sequence[tuple[int, int]]) -> str:
    """What an episode's marks depend on: its kept frames and its OP/ED."""
    parts = [(p.kind.value, p.start, p.end) for p in list_parts(ctx.conn, episode.id)]
    return f"{_stamp(frames)}-{hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:8]}"


def update(ctx: Context) -> dict[int, Repeat]:
    """The series' recurring shots, recomputed when an episode's kept frames or OP/ED have
    changed since they were stored."""
    frames = {e.id: _kept_frames(ctx, e) for e in list_episodes(ctx.conn)}
    episodes = [e for e in list_episodes(ctx.conn) if frames[e.id]]
    stamps = {e.id: _episode_stamp(ctx, e, frames[e.id]) for e in episodes}
    path = marks_path(ctx.paths)
    if path.exists():
        data = cast(Json, json.loads(path.read_text(encoding="utf-8")))
        known = {int(e): s for e, s in cast(dict[str, str], data.get("episodes", {})).items()}
        if data.get("version") == VERSION and known == stamps:
            return _from_json(data)
    marks = find_repeats(ctx, episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(_to_json(stamps, marks), indent=1), encoding="utf-8")
    partial.replace(path)
    return marks


def summary(ctx: Context, marks: Mapping[int, Repeat]) -> Json:
    """Counts per kind and per episode, and the shots the lyric styles' OP/ED did not cover."""
    by_episode: dict[str, dict[str, int]] = {}
    beyond: dict[str, int] = {}
    for e in list_episodes(ctx.conn):
        kept = list_shots(ctx.conn, e.id, ShotStatus.KEPT)
        parts = list_parts(ctx.conn, e.id)
        counts: dict[str, int] = {}
        for s in kept:
            if s.id not in marks:
                continue
            kind = marks[s.id].kind
            counts[kind] = counts.get(kind, 0) + 1
            start, end = e.seconds(s.start_frame), e.seconds(s.end_frame)
            if not any(p.start < end and start < p.end for p in parts):
                beyond[kind] = beyond.get(kind, 0) + 1
        if kept:
            by_episode[e.path] = {"kept": len(kept), **counts}
    kinds = {k: sum(1 for r in marks.values() if r.kind == k) for k in KINDS}
    return {
        "marked": len(marks),
        "kinds": kinds,
        "outside_lyric_op_ed": beyond,
        "episodes": by_episode,
    }
