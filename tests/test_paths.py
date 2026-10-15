import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_plan import song

from tokimeki.library.paths import SeriesPaths
from tokimeki.mad.plan import Plan, SlotPlan, mad_dir, read_plan, write_plan
from tokimeki.mad.song.analysis import SongSource, analysis_path, load_analysis, save_analysis


def test_layout_is_hidden_next_to_episodes(tmp_path: Path) -> None:
    paths = SeriesPaths(tmp_path)
    assert paths.db == tmp_path / ".tokimeki" / "library.db"
    assert paths.frame_path(3, 1234) == tmp_path / ".tokimeki/cache/frames/3/001234.jpg"
    assert paths.report_dir == tmp_path / ".tokimeki" / "report"


def test_finds_videos_but_not_hidden_or_other_files(tmp_path: Path) -> None:
    for name in ["s1/ep02.mkv", "s1/ep01.MKV", "ep03.mp4", "notes.txt", "ep01.ass"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    (tmp_path / ".tokimeki/cache").mkdir(parents=True)
    (tmp_path / ".tokimeki/cache/clip.mkv").touch()
    assert SeriesPaths(tmp_path).find_episodes() == ["ep03.mp4", "s1/ep01.MKV", "s1/ep02.mkv"]


def test_plans_and_songs_keep_paths_under_the_data_root_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "anime"
    series = root / "show"
    monkeypatch.setenv("TOKIMEKI_DATA", str(root))
    monkeypatch.setenv("TOKIMEKI_HOME", str(tmp_path / "home"))
    plan = Plan("p", str(series), "c", "x", [SlotPlan(0.0, 1.0, 3, episode="s1/ep01.mkv")])
    path = mad_dir(series, "p") / "plan.json"
    write_plan(plan, path)
    stored = json.loads(path.read_text())
    assert stored["series"] == "show" and stored["slots"][0]["episode"] == "s1/ep01.mkv"
    assert read_plan(path) == plan
    old = {**stored, "schema": "tokimeki.plan/3", "series": str(series)}
    assert Plan.from_dict(old).series == str(series)
    elsewhere = Plan("p", str(tmp_path / "other"), "c", "x", plan.slots)
    assert elsewhere.to_dict()["series"] == str(tmp_path / "other")

    media = series / "s1" / "ep01.mkv"
    lyrics = f"{series / 's1' / 'ep01.ass'}@1#opjp@ja"
    analysis = replace(
        song(),
        source=SongSource(media, 1231.05, 89.39, "楽園PROJECT", stream=1),
        inputs={"lyrics": lyrics, "instrumental": ""},
    )
    v2 = json.loads(analysis.to_json())
    v2.update(version=2, source={**v2["source"], "path": str(media)}, inputs=analysis.inputs)
    analysis_path("test-song").parent.mkdir(parents=True)
    analysis_path("test-song").write_text(json.dumps(v2), encoding="utf-8")
    loaded = load_analysis("test-song")
    assert loaded.source.path == media
    assert loaded.inputs["lyrics"] == "show/s1/ep01.ass@1#opjp@ja"
    save_analysis(loaded)
    text = analysis_path("test-song").read_text()
    assert str(root) not in text and json.loads(text)["source"]["path"] == "show/s1/ep01.mkv"
    monkeypatch.setenv("TOKIMEKI_DATA", str(tmp_path / "moved"))
    assert load_analysis("test-song").source.path == tmp_path / "moved/show/s1/ep01.mkv"
    assert read_plan(path).series == str(tmp_path / "moved/show")
