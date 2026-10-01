from pathlib import Path

from tokimeki.paths import SeriesPaths


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
