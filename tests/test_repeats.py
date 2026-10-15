import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from tokimeki.anime import repeats
from tokimeki.library.episodes import register_episode
from tokimeki.library.lines import replace_lines
from tokimeki.library.records import Part, PartKind, Rating
from tokimeki.library.series import Context, open_series
from tokimeki.library.shots import RatedFrame, keep_shot, replace_shots
from tokimeki.mad.plan import Plan, SlotPlan
from tokimeki.mad.validate import validate
from tokimeki.media.phash import distances, phash

SHOT = 48
"""Frames per shot (2 s at 24 fps); three are sampled."""


def drawing(seed: int, shift: int = 0) -> Image.Image:
    """A picture of its own: coarse random shapes, a few frames apart as `shift` grows."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (9, 16, 3), dtype=np.uint8)
    image = Image.fromarray(blocks).resize((640, 360), Image.Resampling.BICUBIC)
    return image.crop((shift, 0, 640, 360)).resize((640, 360))


def test_hashes_survive_reencoding_and_tell_drawings_apart(tmp_path: Path) -> None:
    a = drawing(1)
    a.save(tmp_path / "a.jpg", quality=40)
    again = phash(Image.open(tmp_path / "a.jpg").resize((160, 90)))[0]
    values = np.array([phash(a)[0], again, phash(drawing(1, 3))[0], phash(drawing(2))[0]])
    d = distances(values[:1].astype(np.uint64), values.astype(np.uint64))[0]
    assert d[1] <= repeats.SAME and d[2] <= repeats.SAME and d[3] > 12
    assert phash(Image.new("RGB", (64, 36), (20, 20, 20)))[1] < repeats.FLAT


def _episode(
    ctx: Context, name: str, shots: list[tuple[float, int]], parts: dict[float, PartKind]
) -> None:
    """An episode of 2 s shots at `(second, picture)`; `parts` puts an OP or ED over the shot
    at that second."""
    episode = register_episode(ctx.conn, name, 640, 360, Fraction(24), 24 * 720)
    spans = [(round(t * 24), round(t * 24) + SHOT) for t, _ in shots]
    for shot, (_, picture) in zip(replace_shots(ctx.conn, episode.id, spans), shots, strict=True):
        frames = [shot.start_frame + k for k in (6, 18, 30)]
        keep_shot(ctx.conn, shot.id, [RatedFrame(f, Rating(1, 0, 0, 0), []) for f in frames])
        for k, f in enumerate(frames):
            path = ctx.paths.frame_path(episode.id, f)
            path.parent.mkdir(parents=True, exist_ok=True)
            drawing(picture, k).save(path, quality=85)
    marked = [Part(episode.id, kind, t + 0.5, t + 1.5) for t, kind in parts.items()]
    replace_lines(ctx.conn, episode.id, [], marked)


def test_recurring_shots_by_kind_and_originals_kept(tmp_path: Path) -> None:
    ctx = open_series(tmp_path)
    title, op, eyecatch, ed = 100, 101, 102, 103
    parts = {104.0: PartKind.OPENING, 600.0: PartKind.ENDING}
    furniture = [(100.0, title), (104.0, op), (400.0, eyecatch), (600.0, ed)]
    _episode(ctx, "ep01.mkv", sorted([(30.0, 1), *furniture, (595.0, 41), (700.0, 21)]), parts)
    _episode(ctx, "ep02.mkv", sorted([(30.0, 1), *furniture, (108.0, 41), (300.0, 21)]), parts)
    _episode(ctx, "ep03.mkv", sorted([(30.0, 31), *furniture, (200.0, 21), (500.0, title)]), parts)
    ctx.conn.commit()
    marks = repeats.update(ctx)
    kinds = {shot: mark.kind for shot, mark in marks.items()}
    assert kinds == {
        2: "op", 3: "op", 4: "recurring", 6: "ed", 7: "preview",
        8: "recap", 9: "op", 10: "op", 11: "recap", 13: "recurring", 14: "ed",
        16: "op", 17: "op", 18: "recap", 19: "recurring", 21: "ed",
    }  # fmt: skip
    assert marks[7].like == (12, 18) and marks[8].like == (1,)
    summary = repeats.summary(ctx, marks)
    assert summary["outside_lyric_op_ed"] == {"op": 3, "recurring": 3, "preview": 1, "recap": 3}
    stored = json.loads(repeats.marks_path(ctx.paths).read_text())
    assert stored["shots"]["7"]["kind"] == "preview"
    assert repeats.update(ctx) == marks
    plan = Plan("p", str(tmp_path), "", "x", [SlotPlan(0.0, 1.0, 7), SlotPlan(1.0, 2.0, 12)])
    issues = validate(ctx.conn, plan, None, marks)
    assert [(i.slot, i.code) for i in issues if i.code.startswith("shot.")] == [
        (0, "shot.recurring")
    ]
    replace_lines(ctx.conn, 1, [], [Part(1, PartKind.OPENING, 104.5, 105.5)])
    assert repeats.update(ctx)[7].kind == "recurring"
