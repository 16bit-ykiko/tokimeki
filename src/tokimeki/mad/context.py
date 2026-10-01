"""Everything an arranger needs to write a plan, as one compact JSON object.

Only kept shots outside the OP/ED are listed, so only they can reach whoever reads it.
Keyframes are paths to cached frames of those shots, for agents that can look at images.
"""

from collections.abc import Mapping, Sequence

from tokimeki.library.records import Episode
from tokimeki.mad.candidates import Candidate, find_candidates
from tokimeki.mad.plan import DEFAULT_FPS, MAX_SPEED, MIN_SPEED
from tokimeki.mad.refine import EDGE_FRAMES
from tokimeki.mad.validate import BEAT_TOLERANCE, MIN_SLOT
from tokimeki.song.analysis import SongAnalysis
from tokimeki.song.lyrics import pair
from tokimeki.song.slots import excerpt_between, make_slots
from tokimeki.stages.base import Context
from tokimeki.stages.scenes import summaries

type Json = dict[str, object]

PEAKS_PER_SHOT = 2
TAGS_PER_PEAK = 4


def _r(x: float, digits: int = 2) -> float:
    return round(x, digits)


def _shot(ctx: Context, c: Candidate) -> Json:
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
        "keyframe": str(ctx.paths.frame_path(c.episode.id, frame)),
    }


def _song(analysis: SongAnalysis, budget: int) -> Json:
    start, end = analysis.excerpt
    excerpt = excerpt_between(
        analysis.bars, analysis.source.duration, analysis.sections, start, end
    )
    fitted = make_slots(analysis.beats, analysis.bars, analysis.bpm, excerpt, budget)
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
        "slots": [
            {"start": _r(s.start, 3), "end": _r(s.end, 3), "section": s.section, "beats": s.beats}
            for s in fitted
        ],
        "slots_note": "a suggestion fitted to the number of candidate shots: a cut a bar in "
        "verses, every two beats in the chorus, slower where shots run short; any beats will do",
    }


def build_context(
    ctx: Context,
    character: str,
    episodes: Sequence[Episode],
    analysis: SongAnalysis | None,
    min_presence: float,
    boost: Mapping[str, float],
) -> Json:
    candidates = find_candidates(ctx, character, episodes, boost, min_presence)
    by_scene: dict[tuple[str, int], list[Candidate]] = {}
    for c in candidates:
        by_scene.setdefault((c.episode.path, c.scene), []).append(c)
    scenes: list[Json] = []
    for episode in episodes:
        for summary in summaries(ctx, episode):
            members = by_scene.get((episode.path, summary.scene.index))
            if not members:
                continue
            scenes.append(
                {
                    "episode": episode.path,
                    "scene": summary.scene.index,
                    "start": _r(summary.start),
                    "end": _r(summary.end),
                    "cast": {name: _r(share) for name, share in summary.cast},
                    "lines": [[_r(x.start), x.text.replace("\n", " ")] for x in summary.lines],
                    "shots": [_shot(ctx, c) for c in members],
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
        },
        "legend": {
            "presence": "share of the shot's sampled frames the character is in",
            "face": "her largest face height as a share of the frame (close-up >= 0.35)",
            "peaks": "her cutest sampled moments: episode second, score, WD14 expression tags",
            "keyframe": "the cached frame at the best peak",
        },
        "scenes": scenes,
    }
    if analysis is not None:
        out["song"] = _song(analysis, len(candidates))
    return out
