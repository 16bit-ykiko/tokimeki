from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tokimeki.library.records import Episode
from tokimeki.stages import content_filter, shots
from tokimeki.stages.base import Context


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[Context, Sequence[Episode]], None]
    reset: Callable[[Context, Episode], None]


STAGES = (
    Stage(shots.NAME, shots.run, shots.reset),
    Stage(content_filter.NAME, content_filter.run, content_filter.reset),
)
STAGE_NAMES = tuple(stage.name for stage in STAGES)


def _position(name: str) -> int:
    if name not in STAGE_NAMES:
        raise ValueError(f"unknown stage {name!r}; stages are {', '.join(STAGE_NAMES)}")
    return STAGE_NAMES.index(name)


def redo(ctx: Context, episodes: Sequence[Episode], stage: str) -> None:
    """Forget `stage` and every later stage for `episodes`, so the next run repeats them."""
    for later in reversed(STAGES[_position(stage) :]):
        for episode in episodes:
            later.reset(ctx, episode)


def run(ctx: Context, episodes: Sequence[Episode], until: str = STAGE_NAMES[-1]) -> None:
    """Run every stage up to `until`, skipping the work each stage has already finished."""
    for stage in STAGES[: _position(until) + 1]:
        stage.run(ctx, episodes)
