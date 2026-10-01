import sqlite3
from fractions import Fraction
from pathlib import Path

import fakes
import numpy as np
import pytest
from clips import make_clip
from test_lines import ASS

from tokimeki.library import episodes as episode_db
from tokimeki.library import shots as shot_db
from tokimeki.library.db import open_library
from tokimeki.library.records import PartKind, Rating, Tag, TagCategory
from tokimeki.library.scenes import list_scenes, replace_scenes
from tokimeki.library.tags import frame_tag_scores, tag_stats
from tokimeki.stages import pipeline, scenes
from tokimeki.stages.base import open_series, register_episodes
from tokimeki.stages.scenes import ShotFacts, colour_histogram, continues, group

RED = colour_histogram(np.full((4, 4, 3), (250, 10, 10), dtype=np.uint8))
BLUE = colour_histogram(np.full((4, 4, 3), (10, 10, 250), dtype=np.uint8))


def fact(
    index: int,
    lines: frozenset[int] = frozenset(),
    first: float | None = None,
    last: float | None = None,
    *,
    kept: bool = True,
    colours: np.ndarray | None = None,
    part: PartKind | None = None,
    cast: frozenset[str] = frozenset(),
) -> ShotFacts:
    return ShotFacts(
        index, index, index * 2.0, index * 2.0 + 2.0, kept, part, lines, first, last, cast,
        BLUE if colours is None else colours,
    )  # fmt: skip


def test_dialogue_carries_a_scene_across_a_change_of_look() -> None:
    assert continues(fact(0, frozenset({7}), colours=RED), fact(1, frozenset({7})))
    assert continues(
        fact(0, frozenset({1}), 0.0, 1.0, colours=RED), fact(1, frozenset({2}), 2.0, 3.0)
    )
    assert not continues(
        fact(0, frozenset({1}), 0.0, 1.0, colours=RED), fact(1, frozenset({2}), 3.0, 4.0)
    )


def test_looks_merge_only_when_clearly_alike() -> None:
    assert continues(fact(0), fact(1))
    assert not continues(fact(0, colours=RED), fact(1))
    mixed = (RED + BLUE) / 2
    assert not continues(fact(0, colours=mixed), fact(1))
    assert not continues(fact(0, colours=mixed, cast=frozenset({"梦梦"})), fact(1))


def test_scenes_break_at_dropped_shots_and_parts() -> None:
    assert not continues(fact(0), fact(1, kept=False))
    assert not continues(fact(0), fact(2))
    assert not continues(fact(0), fact(1, part=PartKind.OPENING))
    groups = group([fact(0), fact(1), fact(2, kept=False), fact(3), fact(4, colours=RED)])
    assert [[f.index for f in g] for g in groups] == [[0, 1], [3], [4]]


def test_long_scenes_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scenes, "MAX_SCENE_SECONDS", 5.0)
    assert [len(g) for g in group([fact(i) for i in range(5)])] == [2, 2, 1]


def test_scenes_hold_only_kept_shots_and_tags_summarise() -> None:
    conn = open_library(":memory:")
    episode = episode_db.register_episode(conn, "ep.mkv", 1280, 720, Fraction(24), 100)
    a, b = shot_db.replace_shots(conn, episode.id, [(0, 50), (50, 100)])
    safe = Rating(1, 0, 0, 0)
    smile = Tag("smile", TagCategory.GENERAL, 0.8)
    shot_db.keep_shot(
        conn, a.id, [shot_db.RatedFrame(10, safe, [smile]), shot_db.RatedFrame(20, safe, [])]
    )
    with pytest.raises(sqlite3.IntegrityError, match="only kept shots"):
        replace_scenes(conn, episode.id, [[a.id, b.id]])
    (scene,) = replace_scenes(conn, episode.id, [[a.id]])
    assert list_scenes(conn, episode.id) == [scene]
    (stat,) = tag_stats(conn, [a.id])
    assert (stat.tag, stat.peak, stat.share) == ("smile", pytest.approx(0.8), 0.5)
    assert frame_tag_scores(conn, a.id) == {10: {"smile": pytest.approx(0.8)}, 20: {}}


def test_stage_builds_scenes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"], seconds=3)
    (tmp_path / "ep01.ass").write_text(ASS, encoding="utf-8")
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    pipeline.run(ctx, register_episodes(ctx))
    (episode,) = register_episodes(ctx)
    (summary,) = scenes.summaries(ctx, episode)
    assert len(summary.scene.shot_ids) == 1
    assert [x.text for x in summary.lines][:1] == ["果酱可以吗, 茉茉"]
    assert summary.expressions[0].tag == "smile"
