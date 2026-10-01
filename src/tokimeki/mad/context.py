"""Everything an arranger needs to write a plan, as one compact JSON object.

Only kept shots outside the OP/ED are listed, so only they can reach whoever reads it.
Keyframes are paths to cached frames of those shots, for agents that can look at images.
Voice lines are the dialogue of her scenes that kept shots cover, with who is on screen.
"""

from collections.abc import Mapping, Sequence

from tokimeki.library.records import Episode
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.mix import DUCK_DB
from tokimeki.mad.motion import ShotMotion, motions
from tokimeki.mad.plan import DEFAULT_FPS, MAX_SPEED, MIN_SPEED
from tokimeki.mad.refine import EDGE_FRAMES
from tokimeki.mad.validate import BEAT_TOLERANCE, MIN_SLOT
from tokimeki.mad.voices import usable_lines
from tokimeki.song.analysis import SongAnalysis, slots_over
from tokimeki.song.gaps import gaps, quiet_bars
from tokimeki.song.lyrics import pair
from tokimeki.stages.base import Context
from tokimeki.stages.scenes import summaries

type Json = dict[str, object]

PEAKS_PER_SHOT = 2
TAGS_PER_PEAK = 4


def _r(x: float, digits: int = 2) -> float:
    return round(x, digits)


ONSETS_PER_SHOT = 3


def _motion(m: ShotMotion | None) -> Json | None:
    if m is None:
        return None
    strongest = sorted(m.onsets, key=lambda o: -o.strength)[:ONSETS_PER_SHOT]
    return {
        "still": m.still,
        "moving": _r(m.moving),
        "onsets": [
            [_r(o.time), _r(o.strength, 3)] for o in sorted(strongest, key=lambda o: o.time)
        ],
    }


def _shot(ctx: Context, c: Candidate, motion: ShotMotion | None) -> Json:
    seen = [f for f in c.frames if f.face > 0] or list(c.frames)
    peaks: list[Json] = []
    for f in sorted(seen, key=lambda f: -f.cuteness)[:PEAKS_PER_SHOT]:
        tags = sorted(f.tags.items(), key=lambda item: -item[1])[:TAGS_PER_PEAK]
        peaks.append({"t": _r(f.time), "cute": _r(f.cuteness), "tags": {t: _r(v) for t, v in tags}})
    frame = round(c.peak.time * c.episode.fps)
    return {
        "shot": c.shot.id,
        "start": _r(c.start),
        "end": _r(c.end),
        "len": _r(c.duration),
        "framing": c.framing,
        "presence": _r(c.presence),
        "face": _r(c.face_height),
        "with": list(c.others),
        "peaks": peaks,
        "motion": _motion(motion),
        "keyframe": str(ctx.paths.frame_path(c.episode.id, frame)),
    }


def _song(
    analysis: SongAnalysis,
    budget: int,
    span: tuple[float, float] | None,
    fixed: Mapping[str, int] | None,
) -> Json:
    excerpt, fitted = slots_over(analysis, span, budget, fixed)
    start, end = excerpt.start, excerpt.end
    return {
        "id": analysis.id,
        "title": analysis.title,
        "bpm": _r(analysis.bpm, 1),
        "duration": _r(analysis.source.duration),
        "vocal_line_from": analysis.vocal_source,
        "excerpt": [_r(start, 3), _r(end, 3)],
        "sections": [
            {"label": s.label, "start": _r(s.start, 3), "end": _r(s.end, 3)}
            for s in analysis.sections
        ],
        "beats": [_r(b, 3) for b in analysis.beats if start - 1e-6 <= b <= end + 1e-6],
        "lyrics": [
            {"start": _r(p.start), "end": _r(p.end), **p.texts}
            for p in pair(analysis.lyrics)
            if start - 1e-6 <= p.start < end
        ],
        "gaps": [
            {"start": _r(g.start, 3), "end": _r(g.end, 3), "len": _r(g.length), "where": g.where}
            for g in gaps(analysis, start, end)
        ],
        "quiet_bars": [_r(b, 3) for b in quiet_bars(analysis, start, end)],
        "slots": [
            {"start": _r(s.start, 3), "end": _r(s.end, 3), "section": s.section, "beats": s.beats}
            for s in fitted
        ],
        "slots_note": "a suggestion fitted to the number of candidate shots (or --max-slots): a "
        "cut a bar in verses, every two beats in the chorus, slower where shots run short; any "
        "beats will do",
    }


def build_context(
    ctx: Context,
    character: str,
    episodes: Sequence[Episode],
    analysis: SongAnalysis | None,
    min_presence: float,
    boost: Mapping[str, float],
    span: tuple[float, float] | None = None,
    max_slots: int | None = None,
    fixed: Mapping[str, int] | None = None,
) -> Json:
    candidates = find_candidates(ctx, character, episodes, boost, min_presence)
    moves = motions(ctx.paths, [(c.episode, c.shot) for c in candidates])
    by_scene: dict[tuple[str, int], list[Candidate]] = {}
    for c in candidates:
        by_scene.setdefault((c.episode.path, c.scene), []).append(c)
    scenes: list[Json] = []
    voice_lines: list[Json] = []
    for episode in episodes:
        scene_of: dict[int, int] = {}
        for summary in summaries(ctx, episode):
            members = by_scene.get((episode.path, summary.scene.index))
            if not members:
                continue
            scene_of.update((shot, summary.scene.index) for shot in summary.scene.shot_ids)
            scenes.append(
                {
                    "episode": episode.path,
                    "scene": summary.scene.index,
                    "start": _r(summary.start),
                    "end": _r(summary.end),
                    "cast": {name: _r(share) for name, share in summary.cast},
                    "lines": [[_r(x.start), x.text.replace("\n", " ")] for x in summary.lines],
                    "shots": [_shot(ctx, c, moves.get(c.shot.id)) for c in members],
                }
            )
        for x in usable_lines(ctx, episode):
            scene = next((scene_of[s] for s in x.shots if s in scene_of), None)
            if scene is None:
                continue
            voice_lines.append(
                {
                    "line": x.line.id,
                    "episode": episode.path,
                    "scene": scene,
                    "start": _r(x.line.start),
                    "end": _r(x.line.end),
                    "len": _r(x.line.end - x.line.start),
                    "text": " ".join(x.line.text.split()),
                    "shots": x.shots,
                    "on_screen": x.on_screen,
                }
            )
    out: Json = {
        "schema": "tokimeki.context/1",
        "series": str(ctx.paths.root),
        "character": character,
        "episodes": [
            {"path": e.path, "fps": f"{e.fps.numerator}/{e.fps.denominator}"} for e in episodes
        ],
        "counts": {"shots": len(candidates), "scenes": len(scenes)},
        "rules": {
            "speed": [MIN_SPEED, MAX_SPEED],
            "min_slot": MIN_SLOT,
            "beat_tolerance": round(BEAT_TOLERANCE, 4),
            "edge_frames": EDGE_FRAMES,
            "fps": DEFAULT_FPS,
            "kept_shots_only": True,
            "each_shot_once": True,
            "voices": "inside song gaps (no sung lyrics) unless force; never from dropped shots;"
            f" the song ducks {DUCK_DB:g} dB under each",
        },
        "legend": {
            "presence": "share of the shot's sampled frames the character is in",
            "face": "her largest face height as a share of the frame (close-up >= 0.35)",
            "peaks": "her cutest sampled moments: episode second, score, WD14 expression tags",
            "keyframe": "the cached frame at the best peak",
            "motion": "still: hardly moves (drift suits it); moving: share of frames that move;"
            " onsets: [episode second, share of pixels changed] where movement starts, which"
            " refine lands on the cut or a slot's accent",
            "voice_lines": "lines a voice may come from (subtitle times; refine trims to the"
            " voice); speakers are unknown, on_screen says who is seen while it is said",
        },
        "scenes": scenes,
        "voice_lines": voice_lines,
    }
    if analysis is not None:
        budget = min(len(candidates), max_slots) if max_slots else len(candidates)
        out["song"] = _song(analysis, budget, span, fixed)
    return out
