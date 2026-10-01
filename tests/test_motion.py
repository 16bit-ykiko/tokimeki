from pathlib import Path

import fakes
import numpy as np
import pytest
from clips import make_clip
from test_plan import candidate, song

from tokimeki.mad.motion import Onset, ShotMotion, shot_motion
from tokimeki.mad.plan import Plan, SlotPlan
from tokimeki.mad.refine import refine, sync
from tokimeki.media.decode import decode_luma
from tokimeki.models.motion import MotionMeter
from tokimeki.stages import motion, pipeline
from tokimeki.stages.base import open_series, register_episodes


def test_stage_measures_kept_frames_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    measures = motion.load_motion(ctx.paths, episodes[0].id)
    assert measures is not None and measures.shape[1] == 2
    kept = ~np.isnan(measures[:, 0])
    assert 0 < kept.sum() < len(kept)
    first = int(np.flatnonzero(kept)[0])
    assert measures[first].tolist() == [0.0, 0.0]
    assert measures[fakes.JUMP, 1] == 1.0
    assert measures[kept, 1].sum() == 1.0
    pipeline.redo(ctx, episodes, motion.NAME)
    assert motion.load_motion(ctx.paths, episodes[0].id) is None


def test_onsets_follow_calm_and_ignore_held_drawings() -> None:
    changed = np.zeros(60)
    changed[1:4] = 0.2
    changed[[10, 12, 14]] = 0.1
    changed[30:33] = [0.05, 0.3, 0.0]
    measures = np.stack([np.full(60, 0.001), changed], axis=1).astype(np.float32)
    found = shot_motion(measures, 100.0, 24.0)
    assert [round(o.time * 24 - 2400) for o in found.onsets] == [10, 30]
    assert [o.strength for o in found.onsets] == pytest.approx([0.1, 0.3])
    assert found.still
    measures[:, 0] = 0.01
    assert not shot_motion(measures, 100.0, 24.0).still


def test_sync_lands_an_onset_on_the_cut_or_an_accent() -> None:
    c = candidate(1, 6.0, {"smile": 0.9})
    slot = SlotPlan(0.0, 2.0, 1)
    near = ShotMotion((Onset(248 / 24, 0.2),), 0.1, 0.001)
    found = sync(slot, 10.0833, 2.0, c, near, [])
    assert found is not None and found.target == "the cut"
    assert found.start == pytest.approx(247 / 24)
    late = ShotMotion((Onset(11.0, 0.2),), 0.1, 0.001)
    assert sync(slot, 10.0833, 2.0, c, late, []) is None
    found = sync(slot, 10.0833, 2.0, c, late, [(0.75, 1.0)])
    assert found is not None and found.target == "the accent at 0.75s"
    assert found.start == pytest.approx(10.25)


def test_refine_reports_what_moved() -> None:
    c = candidate(1, 6.0, {"smile": 0.9})
    plan = Plan("t", "/s", "梦梦", "test-song", [SlotPlan(0.0, 2.0, 1)])
    analysis = song(0.25, 4.0)
    still = {1: ShotMotion((Onset(248 / 24, 0.2),), 0.1, 0.001)}
    refined, notes = refine(plan, analysis, {1: c}, still)
    assert refined.slots[0].source_in == pytest.approx(247 / 24, abs=1e-4)
    assert any("on the cut" in n for n in notes)
    frozen = Plan("t", "/s", "梦梦", "test-song", [SlotPlan(0.0, 2.0, 1, sync=False)])
    assert refine(frozen, analysis, {1: c}, still)[0].slots[0].source_in == pytest.approx(
        242 / 24, abs=1e-4
    )


@pytest.mark.gpu
def test_luma_decode_and_meter_run_on_the_gpu(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "c.mkv", ["color=c=black", "color=c=white"], seconds=1)
    chunks = list(decode_luma(clip, 320, 180, [(10, 20), (30, 40)], chunk=4))
    frames = np.concatenate(chunks)
    assert frames.shape == (20, 90, 160)
    assert frames[:10].mean() < 40 and frames[10:].mean() > 200
    meter = MotionMeter()
    try:
        measures = meter.differences(frames, None)
    finally:
        meter.close()
    assert measures.shape == (20, 2)
    assert measures[10, 1] == 1.0 and measures[[0, 5, 15], 1].tolist() == [0.0, 0.0, 0.0]
