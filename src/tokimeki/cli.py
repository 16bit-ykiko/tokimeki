"""Find the cute moments of anime heroines across a series and cut them into MADs."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from tokimeki.library.episodes import list_episodes, stage_done
from tokimeki.library.shots import status_counts
from tokimeki.stages import pipeline
from tokimeki.stages.base import open_series, register_episodes
from tokimeki.stages.selfcheck import run_checks

type Handler = Callable[[argparse.Namespace], int]


def _series(args: argparse.Namespace) -> Path:
    series: str = args.series
    return Path(series).expanduser()


def cmd_gpu_check(_: argparse.Namespace) -> int:
    results = run_checks()
    for check in results:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name:12} {check.detail}")
    return 0 if all(check.ok for check in results) else 1


def cmd_status(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    register_episodes(ctx)
    for episode in list_episodes(ctx.conn):
        done = [s for s in pipeline.STAGE_NAMES if stage_done(ctx.conn, episode.id, s)]
        counts = status_counts(ctx.conn, episode.id)
        shots = ", ".join(f"{status.value} {n}" for status, n in counts.items())
        print(f"{episode.id:3} {episode.path}  stages: {' '.join(done) or '-'}  shots: {shots}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    episodes = register_episodes(ctx)
    only: list[str] | None = args.episode
    if only:
        episodes = [e for e in episodes if any(part in e.path for part in only)]
    if not episodes:
        print(f"no episodes found in {ctx.paths.root}", file=sys.stderr)
        return 1
    redo: str | None = args.redo
    if redo is not None:
        pipeline.redo(ctx, episodes, redo)
    until: str = args.until
    pipeline.run(ctx, episodes, until)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokimeki", description=__doc__)
    sub = parser.add_subparsers(required=True, metavar="command")

    def command(name: str, handler: Handler, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(handler=handler)
        return p

    command("gpu-check", cmd_gpu_check, "check that NVDEC, PyTorch and onnxruntime use the GPU")
    status = command("status", cmd_status, "list a series' episodes and finished stages")
    status.add_argument("series", help="the series directory holding the episodes")

    ingest = command("ingest", cmd_ingest, "run the pipeline over a series' episodes")
    ingest.add_argument("series", help="the series directory holding the episodes")
    ingest.add_argument(
        "--episode", action="append", help="only episodes whose path contains this (repeatable)"
    )
    ingest.add_argument(
        "--redo", choices=pipeline.STAGE_NAMES, help="forget this stage and the later ones first"
    )
    ingest.add_argument(
        "--until",
        choices=pipeline.STAGE_NAMES,
        default=pipeline.STAGE_NAMES[-1],
        help="stop after this stage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
