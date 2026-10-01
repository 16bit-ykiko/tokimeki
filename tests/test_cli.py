from pathlib import Path

import fakes
import pytest
from clips import make_clip

from tokimeki.cli import main


def test_ingest_then_name_merge_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    series = str(tmp_path)
    assert main(["ingest", series]) == 0
    assert "report:" in capsys.readouterr().out

    assert main(["cast", "name", series, "1", "Momo"]) == 0
    assert main(["cast", "list", series]) == 0
    listing = capsys.readouterr().out
    assert "Momo" in listing and "momo_velia_deviluke 100%" in listing

    assert main(["status", series]) == 0
    assert "stages: shots filter cast" in capsys.readouterr().out
    assert main(["report", series]) == 0
    assert "Momo" in (tmp_path / ".tokimeki/report/index.html").read_text()
