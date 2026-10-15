"""Who says each line: speaker embeddings of the clean voices, clustered, matched to the cast.

Speakers are series-wide like face clusters: a new episode's voices join the speakers they
sound like, and only the rest form new speakers. Clusters are kept tight (one character, often
one manner of speaking) and then named from the cast:

- seeds: a line said while one named character, and nobody else, fills the screen in a
  close-up the whole time. A seed alone is wrong about a quarter of the time (inner monologue
  over her face, the listener's reaction shot), but a cluster's seeds vote, and a voice
  cluster is rarely split between two characters;
- without seeds, who is on screen unusually often while the cluster speaks.

That is a suggestion; `accept` turns suggestions into names (merging clusters of one
character), and the user corrects by ear with `name`, `merge` and `move`.
"""

import logging
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tokimeki.library.cast import episode_cast, list_clusters
from tokimeki.library.db import transaction
from tokimeki.library.episodes import clear_stage, get_episode, mark_stage_done, stage_done
from tokimeki.library.records import Episode, ShotCast, ShotStatus
from tokimeki.library.series import Context
from tokimeki.library.shots import list_shots
from tokimeki.models.gpu import loaded
from tokimeki.stages.clustering import average_linkage, evenly_spaced
from tokimeki.voice.clips import CLIP_RATE, Cut, episode_cuts, samples
from tokimeki.voice.embedding import MODEL_FILE, SpeakerEncoder, cosine_distances
from tokimeki.voice.store import (
    LineVoice,
    NewLineVoice,
    Speaker,
    VoiceStatus,
    apply_speaker_fixes,
    assign_speaker,
    clean_voices,
    clear_line_voices,
    create_speaker,
    episode_voices,
    find_speaker,
    fix_line_speaker,
    fixed_lines,
    get_speaker,
    get_voice,
    list_speakers,
    merge_speakers,
    name_speaker,
    replace_line_voices,
    unassign_unnamed_speakers,
    voice_embeddings,
)

NAME = "speakers"
AFTER = ("lines", "voice")
"""Stages whose output the speakers stage reads: a newer run of either makes it stale."""

CLUSTER_THRESHOLD = 0.72
"""Average cosine distance up to which voices stay one speaker. On ep01's 170 hand-checked
lines, 0.70-0.75 keeps clusters 98% pure; at 0.8 the Deviluke sisters start to merge."""
MIN_SPEAKER_LINES = 2
FILL_NEIGHBOURS = 3
FILL_DISTANCE = 0.6
"""A lone voice joins a speaker when its nearest voices all are that speaker's, this close."""
MAX_EXEMPLARS = 48
MIXED_MIN = 2.4
"""Seconds of voice from which a line is checked for a second voice."""
MIXED_DISTANCE = 0.8
"""Cosine distance between a line's halves above which they are two speakers."""
SEED_HEIGHT = 0.2
"""Face height (share of the frame) from which a shot counts as a close-up for seeding."""
SEED_PRESENCE = 0.6
SEED_COVER = 0.9
SEED_VOTES = 2
SEED_MAJORITY = 0.6
MIN_SHARE = 0.3
MIN_LIFT = 1.3
MATCH_LINES = 3
"""Lines a speaker needs before who is on screen says anything about it."""

log = logging.getLogger("tokimeki")


def finished_at(conn: sqlite3.Connection, episode_id: int, stage: str) -> str | None:
    row: tuple[str] | None = conn.execute(
        "SELECT finished_at FROM stage_runs WHERE episode_id = ? AND stage = ?",
        (episode_id, stage),
    ).fetchone()
    return None if row is None else row[0]


def fresh(conn: sqlite3.Connection, episode_id: int, stage: str, after: Sequence[str]) -> bool:
    """`stage` ran for the episode after every stage it reads from."""
    done = finished_at(conn, episode_id, stage)
    if done is None:
        return False
    return all((finished_at(conn, episode_id, s) or "") <= done for s in after)


def ready(conn: sqlite3.Connection, episode_id: int) -> bool:
    return all(stage_done(conn, episode_id, s) for s in AFTER)


def run(ctx: Context, episodes: Sequence[Episode]) -> int:
    """Cut, embed and cluster the voices of episodes not done yet; returns new speakers."""
    todo = [e for e in episodes if ready(ctx.conn, e.id) and not fresh(ctx.conn, e.id, NAME, AFTER)]
    if not todo:
        return 0
    with loaded("speaker embedding", SpeakerEncoder) as encoder:
        for episode in todo:
            cuts, stem = episode_cuts(ctx, episode)
            voices = embed_cuts(cuts, stem, encoder)
            with transaction(ctx.conn):
                keep = named_assignments(ctx.conn, episode.id)
                replace_line_voices(ctx.conn, episode.id, voices)
                for speaker_id, line_ids in keep.items():
                    assign_speaker(ctx.conn, clean_ids(ctx.conn, line_ids), speaker_id)
                mark_stage_done(
                    ctx.conn, episode.id, NAME, {"model": MODEL_FILE, "min_voice": MIXED_MIN}
                )
            counts = {s: sum(v.status == s for v in voices) for s in VoiceStatus}
            log.info(
                "%s: %s",
                episode.path,
                ", ".join(f"{n} {s.value}" for s, n in counts.items() if n),
            )
    with transaction(ctx.conn):
        return assign_speakers(ctx.conn)


def embed_cuts(
    cuts: Sequence[Cut], stem: NDArray[np.float32], encoder: SpeakerEncoder
) -> list[NewLineVoice]:
    """Embeddings of the clean cuts; long ones whose halves sound like two people are mixed."""
    out: list[NewLineVoice] = []
    for cut in cuts:
        if cut.status != VoiceStatus.CLEAN:
            out.append(NewLineVoice(cut.line.id, cut.start, cut.end, cut.status, None))
            continue
        audio = samples(stem, CLIP_RATE, cut.start, cut.end)
        embedding = encoder.embed(audio)
        status = cut.status
        if cut.duration >= MIXED_MIN:
            half = len(audio) // 2
            halves = np.stack([encoder.embed(audio[:half]), encoder.embed(audio[half:])])
            if cosine_distances(halves[:1], halves[1:])[0, 0] > MIXED_DISTANCE:
                status = VoiceStatus.MIXED
        out.append(NewLineVoice(cut.line.id, cut.start, cut.end, status, embedding))
    return out


def named_assignments(conn: sqlite3.Connection, episode_id: int) -> dict[int, list[int]]:
    """Line ids of the episode per named speaker, to restore after its voices are redone."""
    rows: list[tuple[int, int]] = conn.execute(
        "SELECT v.speaker_id, v.line_id FROM line_voices AS v"
        " JOIN lines AS l ON l.id = v.line_id JOIN speakers AS s ON s.id = v.speaker_id"
        " WHERE l.episode_id = ? AND s.name IS NOT NULL",
        (episode_id,),
    ).fetchall()
    out: dict[int, list[int]] = {}
    for speaker_id, line_id in rows:
        out.setdefault(speaker_id, []).append(line_id)
    return out


def clean_ids(conn: sqlite3.Connection, line_ids: Sequence[int]) -> list[int]:
    rows: list[tuple[int]] = conn.execute(
        f"SELECT line_id FROM line_voices WHERE status = 'clean'"
        f" AND line_id IN ({', '.join('?' * len(line_ids))})",
        list(line_ids),
    ).fetchall()
    return [row[0] for row in rows]


def assign_speakers(conn: sqlite3.Connection, threshold: float = CLUSTER_THRESHOLD) -> int:
    """Put every clean, unassigned and unfixed voice with a speaker; returns new speakers."""
    pinned = fixed_lines(conn)
    free = [v.line_id for v in clean_voices(conn) if v.speaker_id is None]
    free = [line_id for line_id in free if line_id not in pinned]
    if not free:
        apply_speaker_fixes(conn)
        return 0
    known = {
        s.id: voice_embeddings(conn, evenly_spaced(ids, MAX_EXEMPLARS))
        for s in list_speakers(conn)
        if (ids := [v.line_id for v in clean_voices(conn, s.id)])
    }
    created = _agglomerate(conn, free, voice_embeddings(conn, free), threshold, known)
    _fill(conn, pinned)
    apply_speaker_fixes(conn)
    return len(created)


def _fill(conn: sqlite3.Connection, pinned: set[int]) -> None:
    """Give a voice left alone by clustering to the speaker all its nearest voices have."""
    voices = clean_voices(conn)
    lone = [v.line_id for v in voices if v.speaker_id is None and v.line_id not in pinned]
    placed = [v for v in voices if v.speaker_id is not None]
    if not lone or len(placed) < FILL_NEIGHBOURS:
        return
    distances = cosine_distances(
        voice_embeddings(conn, lone), voice_embeddings(conn, [v.line_id for v in placed])
    )
    for line_id, row in zip(lone, distances, strict=True):
        nearest = np.argsort(row)[:FILL_NEIGHBOURS]
        owners = {placed[int(j)].speaker_id for j in nearest}
        if len(owners) == 1 and float(row[nearest].mean()) <= FILL_DISTANCE:
            assign_speaker(conn, [line_id], owners.pop())


def _agglomerate(
    conn: sqlite3.Connection,
    line_ids: Sequence[int],
    embeddings: NDArray[np.float32],
    threshold: float,
    known: Mapping[int, NDArray[np.float32]],
) -> list[int]:
    """Cluster voices together with the existing speakers, each a fixed group of exemplars.

    Voices grouped with a speaker join it; other groups of at least `MIN_SPEAKER_LINES`
    become new speakers. Returns the new speaker ids, largest first.
    """
    speakers = list(known)
    k, n = len(speakers), len(line_ids)
    distances = np.full((k + n, k + n), np.inf, dtype=np.float32)
    distances[k:, k:] = cosine_distances(embeddings, embeddings)
    for j, speaker_id in enumerate(speakers):
        column = cosine_distances(embeddings, known[speaker_id]).mean(axis=1)
        distances[k:, j] = column
        distances[j, k:] = column
    weights = np.array([len(known[s]) for s in speakers] + [1] * n, dtype=np.float64)
    labels = average_linkage(distances, threshold, weights)
    groups = sorted((np.flatnonzero(labels == label) for label in np.unique(labels)), key=len)
    created: list[int] = []
    for members in reversed(groups):
        anchors = [speakers[int(m)] for m in members if m < k]
        lines = [line_ids[int(m) - k] for m in members if m >= k]
        if anchors:
            assign_speaker(conn, lines, anchors[0])
        elif len(lines) >= MIN_SPEAKER_LINES:
            speaker_id = create_speaker(conn)
            assign_speaker(conn, lines, speaker_id)
            created.append(speaker_id)
    return created


def reset(ctx: Context, episode: Episode) -> None:
    with transaction(ctx.conn):
        clear_line_voices(ctx.conn, episode.id)
        clear_stage(ctx.conn, episode.id, NAME)


def recluster(ctx: Context, threshold: float = CLUSTER_THRESHOLD) -> int:
    """Cluster again every voice outside a named speaker, from the stored embeddings (no GPU)."""
    with transaction(ctx.conn):
        unassign_unnamed_speakers(ctx.conn)
        return assign_speakers(ctx.conn, threshold)


def rename(ctx: Context, ref: str, name: str | None) -> Speaker:
    with transaction(ctx.conn):
        speaker = find_speaker(ctx.conn, ref)
        name_speaker(ctx.conn, speaker.id, name)
        return get_speaker(ctx.conn, speaker.id)


def merge(ctx: Context, sources: Sequence[str], target: str) -> Speaker:
    with transaction(ctx.conn):
        into = find_speaker(ctx.conn, target)
        for ref in sources:
            merge_speakers(ctx.conn, find_speaker(ctx.conn, ref).id, into.id)
        return get_speaker(ctx.conn, into.id)


def move(ctx: Context, line_ids: Sequence[int], target: str | None) -> list[str]:
    """Pin lines to a named speaker (None: to nobody), through every later clustering."""
    conn = ctx.conn
    done: list[str] = []
    with transaction(conn):
        to = find_speaker(conn, target) if target is not None else None
        if to is not None and to.name is None:
            raise ValueError(f"speaker #{to.id} has no name; name it first (fixes stay on names)")
        for line_id in line_ids:
            voice = get_voice(conn, line_id)
            if voice.status != VoiceStatus.CLEAN:
                raise ValueError(f"line {line_id} is {voice.status.value}, not a clean voice")
            before = get_speaker(conn, voice.speaker_id).label if voice.speaker_id else "nobody"
            fix_line_speaker(conn, line_id, to.id if to else None)
            done.append(f"line {line_id}: {before} -> {to.label if to else 'nobody'}")
    return done


@dataclass(frozen=True, slots=True)
class Match:
    name: str
    share: float
    """Mean on-screen weight of the character over the speaker's lines."""
    lift: float
    """`share` over the character's mean weight over every clean line."""


@dataclass(frozen=True, slots=True)
class ShotView:
    start: float
    end: float
    cast: list[tuple[str | None, ShotCast]]
    """Every face cluster in the shot, with its name (None: unnamed)."""


def shot_views(conn: sqlite3.Connection, episode: Episode) -> list[ShotView]:
    names = {c.id: c.name for c in list_clusters(conn)}
    by_shot: dict[int, list[tuple[str | None, ShotCast]]] = {}
    for sc in episode_cast(conn, episode.id):
        by_shot.setdefault(sc.shot_id, []).append((names.get(sc.cluster_id), sc))
    return [
        ShotView(
            episode.seconds(s.start_frame), episode.seconds(s.end_frame), by_shot.get(s.id, [])
        )
        for s in list_shots(conn, episode.id, ShotStatus.KEPT)
    ]


def _overlap(view: ShotView, voice: LineVoice) -> float:
    return min(view.end, voice.end) - max(view.start, voice.start)


def seed(voice: LineVoice, views: Sequence[ShotView]) -> str | None:
    """The named character alone on screen in a close-up while the line is said, if any."""
    who: set[str] = set()
    covered = 0.0
    for view in views:
        seconds = _overlap(view, voice)
        if seconds <= 0:
            continue
        here = [(name, sc) for name, sc in view.cast if sc.presence >= 0.2]
        if len(here) != 1:
            return None
        name, sc = here[0]
        if name is None or sc.face_height < SEED_HEIGHT or sc.presence < SEED_PRESENCE:
            return None
        who.add(name)
        covered += seconds
    if len(who) != 1 or covered < SEED_COVER * voice.duration:
        return None
    return who.pop()


def on_screen(voice: LineVoice, views: Sequence[ShotView]) -> dict[str, float]:
    """The named characters on screen while the line is said: per character, the share of
    the voice its shots cover times the character's presence in them."""
    weights: dict[str, float] = {}
    for view in views:
        seconds = _overlap(view, voice)
        if seconds <= 0:
            continue
        for name, sc in view.cast:
            if name is not None:
                weights[name] = weights.get(name, 0.0) + seconds / voice.duration * sc.presence
    return weights


@dataclass(frozen=True, slots=True)
class Evidence:
    lines: int = 0
    seeds: dict[str, int] = field(default_factory=dict[str, int])
    """Seed lines among the speaker's, per character."""
    matches: list[Match] = field(default_factory=list[Match])
    """Characters on screen unusually often while it speaks, best first."""

    @property
    def suggestion(self) -> str | None:
        """The name to propose: the seeds' clear majority, else a clear on-screen match."""
        votes = Counter(self.seeds)
        if votes:
            name, count = votes.most_common(1)[0]
            if count >= SEED_VOTES and count >= SEED_MAJORITY * votes.total():
                return name
        best = self.matches[0] if self.matches else None
        if self.lines < MATCH_LINES or best is None:
            return None
        return best.name if best.share >= MIN_SHARE and best.lift >= MIN_LIFT else None


def matches(
    lines: Mapping[int, Mapping[str, float]], members: Mapping[int, Sequence[int]]
) -> dict[int, list[Match]]:
    """Per speaker (its line ids in `members`), the characters on screen unusually often
    while it speaks, best first. `lines` is `on_screen` for every clean line."""
    names = sorted({name for weights in lines.values() for name in weights})
    if not lines:
        return {}
    base = {n: sum(w.get(n, 0.0) for w in lines.values()) / len(lines) for n in names}
    out: dict[int, list[Match]] = {}
    for speaker_id, ids in members.items():
        present = [lines[i] for i in ids if i in lines]
        if not present:
            out[speaker_id] = []
            continue
        found = [
            Match(n, round(share, 3), round(share / base[n], 2) if base[n] > 0 else 0.0)
            for n in names
            if (share := sum(w.get(n, 0.0) for w in present) / len(present)) > 0
        ]
        out[speaker_id] = sorted(found, key=lambda m: m.share * np.log(max(m.lift, 1e-6)))[::-1]
    return out


def evidence(ctx: Context) -> dict[int, Evidence]:
    """Seeds and on-screen matches of every speaker, over every episode with voices."""
    conn = ctx.conn
    weights: dict[int, dict[str, float]] = {}
    members: dict[int, list[int]] = {}
    seeds: dict[int, Counter[str]] = {}
    for episode_id in _episodes_with_voices(conn):
        episode = get_episode(conn, episode_id)
        views = shot_views(conn, episode)
        for v in episode_voices(conn, episode_id):
            if v.status != VoiceStatus.CLEAN:
                continue
            weights[v.line_id] = on_screen(v, views)
            if v.speaker_id is None:
                continue
            members.setdefault(v.speaker_id, []).append(v.line_id)
            if (name := seed(v, views)) is not None:
                seeds.setdefault(v.speaker_id, Counter())[name] += 1
    found = matches(weights, members)
    return {
        s: Evidence(len(ids), dict(seeds.get(s, Counter())), found.get(s, []))
        for s, ids in members.items()
    }


def accept(ctx: Context, refs: Sequence[str] = ()) -> list[str]:
    """Name unnamed speakers (all, or those in `refs`) by their suggestion; a speaker whose
    character already has one is merged into it."""
    conn = ctx.conn
    found = evidence(ctx)
    done: list[str] = []
    with transaction(conn):
        chosen = [find_speaker(conn, r) for r in refs] if refs else list_speakers(conn)
        for speaker in chosen:
            name = found.get(speaker.id, Evidence()).suggestion
            if speaker.name is not None or name is None:
                continue
            owner = next((s for s in list_speakers(conn) if s.name == name), None)
            if owner is None:
                name_speaker(conn, speaker.id, name)
                done.append(f"#{speaker.id} -> {name}")
            else:
                merge_speakers(conn, speaker.id, owner.id)
                done.append(f"#{speaker.id} -> {name} (merged into #{owner.id})")
    return done


def _episodes_with_voices(conn: sqlite3.Connection) -> list[int]:
    rows: list[tuple[int]] = conn.execute(
        "SELECT DISTINCT l.episode_id FROM line_voices AS v JOIN lines AS l ON l.id = v.line_id"
        " ORDER BY l.episode_id"
    ).fetchall()
    return [row[0] for row in rows]
