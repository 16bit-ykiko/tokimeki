import json
from pathlib import Path
from typing import cast

import fakes
import pytest
from clips import make_clip
from httpx2 import Response
from starlette.testclient import TestClient

from tokimeki.anime import pipeline
from tokimeki.anime.pipeline import register_episodes
from tokimeki.library.series import open_series
from tokimeki.mad.plan import Plan, SlotPlan, mad_dir, write_plan
from tokimeki.server.app import create_app, find_series

type Json = dict[str, object]


@pytest.fixture
def series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TOKIMEKI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TOKIMEKI_DATA", str(tmp_path))
    root = tmp_path / "show"
    root.mkdir()
    make_clip(root / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(root)
    pipeline.run(ctx, register_episodes(ctx))
    out = mad_dir(root, "draft")
    write_plan(Plan("draft", str(root), "", "nope", [SlotPlan(0.0, 1.0, 2)]), out / "plan.json")
    (out / "preview.mp4").write_bytes(bytes(range(256)) * 4)
    return root


def _json(response: Response) -> Json:
    return cast(Json, response.json())


def test_api_lists_episodes_and_shots_without_more_of_dropped_ones(series: Path) -> None:
    assert find_series(series.parent) == {"show": series}
    client = TestClient(create_app({"show": series}))
    (episode,) = cast(list[Json], _json(client.get("/api/show/episodes"))["episodes"])
    assert episode["path"] == "ep01.mkv" and "filter" in cast(list[str], episode["stages"])
    assert episode["media"] == {
        "video": "/media/show/1/video.mp4",
        "voice": "/media/show/1/voice.flac",
    }
    shots = cast(list[Json], _json(client.get("/api/show/episodes/1"))["shots"])
    assert [s["status"] for s in shots] == ["dropped", "kept"]
    assert set(shots[0]) == {"id", "start", "end", "status"}
    assert client.get("/api/nowhere/episodes").status_code == 404
    assert client.get("/api/show/episodes/9").status_code == 400
    assert "$schema" in _json(client.get("/api/schema/plan"))


def test_plans_are_checked_through_the_api(series: Path) -> None:
    client = TestClient(create_app({"show": series}))
    report = _json(client.post("/api/show/mads/draft/validate"))
    codes = {cast(str, i["code"]) for i in cast(list[Json], report["issues"])}
    assert not report["ok"] and "song.unknown" in codes
    plan = json.loads((mad_dir(series, "draft") / "plan.json").read_text())
    assert plan["series"] == "show"
    assert _json(client.post("/api/plan/validate", json=plan))["ok"] is False
    bad = client.post("/api/show/plan/context", json={"character": 3})
    assert bad.status_code == 400 and "character" in str(_json(bad)["error"])
    assert client.post("/api/show/mads/../validate").status_code == 404
    assert [m["name"] for m in cast(list[Json], _json(client.get("/api/show/mads"))["mads"])] == [
        "draft"
    ]


def test_files_answer_range_requests(series: Path) -> None:
    client = TestClient(create_app({"show": series}))
    full = client.get("/mads/show/draft/preview.mp4")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    part = client.get("/mads/show/draft/preview.mp4", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == bytes(range(10, 20))
    assert part.headers["content-range"] == "bytes 10-19/1024"
    assert client.get("/mads/show/draft/x.mp4", headers={"Range": "bytes=0-1"}).status_code == 404
    late = client.get("/mads/show/draft/preview.mp4", headers={"Range": "bytes=5000-"})
    assert late.status_code == 416
    voice = client.get("/media/show/1/voice.flac", headers={"Range": "bytes=0-3"})
    assert voice.status_code == 206 and voice.content == b"fLaC"
    assert client.get("/media/show/1/video.mp4").status_code == 404
    assert client.get("/mads/show/draft/../../library.db").status_code == 404
    assert client.get("/mads/show/draft", follow_redirects=False).status_code == 307


def test_voice_stays_on_loopback(series: Path) -> None:
    client = TestClient(create_app({"show": series}, voice=False))
    assert client.get("/media/show/1/voice.flac").status_code == 404
    assert client.get("/api/show/speakers").status_code == 404
    episode = cast(list[Json], _json(client.get("/api/show/episodes"))["episodes"])[0]
    assert "voice" not in cast(Json, episode["media"])
