from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = frozenset({".mkv", ".mp4", ".m2ts", ".ts", ".webm", ".avi"})
DATA_DIR_NAME = ".tokimeki"


@dataclass(frozen=True)
class SeriesPaths:
    """Where a series lives on disk.

    The episodes stay where the user put them and are only read. Everything tokimeki
    derives goes into a hidden directory next to them: the library database, a cache of
    regenerable intermediates, and the report.
    """

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / DATA_DIR_NAME

    @property
    def db(self) -> Path:
        return self.data_dir / "library.db"

    @property
    def cache(self) -> Path:
        return self.data_dir / "cache"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "report"

    def frames_dir(self, episode_id: int) -> Path:
        return self.cache / "frames" / str(episode_id)

    def frame_path(self, episode_id: int, frame_index: int) -> Path:
        return self.frames_dir(episode_id) / f"{frame_index:06d}.jpg"

    def episode_file(self, relative: str) -> Path:
        return self.root / relative

    def find_episodes(self) -> list[str]:
        """Video files under the series root, as sorted posix paths relative to it."""
        found: list[str] = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found.append(relative.as_posix())
        return sorted(found)

    def ensure(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
