"""Find the cute moments of anime heroines across a series and cut them into MADs."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from tokimeki.mad import make as mad
from tokimeki.mad.plan import Preferences
from tokimeki.report import build_report
from tokimeki.song.analysis import analyse_song
from tokimeki.song.slots import first_chorus, make_slots
from tokimeki.stages import cast, pipeline, scenes
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
    for entry in pipeline.status(ctx):
        stages = " ".join(entry.finished) or "-"
        shots = ", ".join(f"{status.value} {n}" for status, n in entry.shots.items())
        print(f"{entry.episode.id:3} {entry.episode.path}  stages: {stages}  shots: {shots}")
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
    no_report: bool = args.no_report
    if not no_report:
        print(f"report: {build_report(ctx.conn, ctx.paths)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    print(f"report: {build_report(ctx.conn, ctx.paths)}")
    return 0


def cmd_scenes(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    only: list[str] | None = args.episode
    for episode in register_episodes(ctx):
        if only and not any(part in episode.path for part in only):
            continue
        print(episode.path)
        for s in scenes.summaries(ctx, episode):
            who = ", ".join(f"{name} {share:.0%}" for name, share in s.cast) or "-"
            faces = ", ".join(f"{t.tag} {t.peak:.2f}" for t in s.expressions[:4]) or "-"
            first = s.lines[0].text.replace("\n", " ") if s.lines else ""
            print(
                f"  #{s.scene.index:3} {s.start:7.1f}-{s.end:7.1f}s {len(s.scene.shot_ids):2} shots"
                f"  {who}  [{faces}]  {len(s.lines)} lines  {first[:24]}"
            )
    return 0


def cmd_song(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    song_path: str = args.song
    track: int | None = args.track
    budget: int = args.shots
    analysis = analyse_song(ctx.paths, Path(song_path).expanduser(), track)
    print(f"{analysis.song.title}: {analysis.bpm:.1f} bpm, {len(analysis.bars)} bars")
    for s in analysis.sections:
        print(f"  {s.label:10} bars {s.first_bar:3}-{s.end_bar:3}  {s.start:6.1f}-{s.end:6.1f}s")
    excerpt = first_chorus(analysis)
    slots = make_slots(analysis, excerpt, budget)
    print(
        f"excerpt {excerpt.start:.1f}-{excerpt.end:.1f}s ({excerpt.duration:.1f}s):"
        f" {len(slots)} slots for up to {budget} shots"
    )
    return 0


def _pairs(values: list[str] | None, kind: type[float] | type[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values or []:
        key, _, number = value.partition("=")
        out[key.strip()] = kind(number)
    return out


def _preferences(args: argparse.Namespace) -> Preferences:
    song: str = args.song
    start: float | None = args.start
    end: float | None = args.end
    return Preferences(
        character=str(args.character),
        song=str(Path(song).expanduser()),
        track=args.track,
        excerpt=(start, end) if start is not None and end is not None else None,
        beats_per_slot={k: int(v) for k, v in _pairs(args.beats, int).items()},
        boost=_pairs(args.boost, float),
        min_presence=float(args.min_presence),
        guidance=str(args.guidance),
    )


def cmd_mad_plan(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    name: str = args.name
    episodes = register_episodes(ctx)
    only: list[str] | None = args.episode
    if only:
        episodes = [e for e in episodes if any(part in e.path for part in only)]
    prefs = _preferences(args)
    if args.prompt:
        print(f"prompt: {mad.write_prompt(ctx, name, prefs, episodes)}")
    plan = mad.make_plan(ctx, name, prefs, episodes, str(args.arranger))
    print(f"plan: {mad.save_plan(ctx, plan)}")
    for clip in plan.clips:
        slot = plan.slots[clip.slot]
        where = f"{slot.index:3} {slot.section:10} {slot.start:6.2f}s"
        print(f"  {where}  shot {clip.shot:5}  {clip.reason}")
    return 0


def cmd_cast_list(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    for cluster, hints in cast.clusters_with_hints(ctx):
        hint = ", ".join(f"{tag} {share:.0%}" for tag, share in hints)
        name = cluster.name or "(unnamed)"
        print(f"{cluster.id:4}  {name:24} {cluster.face_count:5} faces  {hint}")
    return 0


def cmd_cast_name(args: argparse.Namespace) -> int:
    cluster: int = args.cluster
    name: str = args.name
    cast.rename(open_series(_series(args)), cluster, name.strip() or None)
    return 0


def cmd_cast_merge(args: argparse.Namespace) -> int:
    sources: list[int] = args.source
    target: int = args.target
    cast.merge(open_series(_series(args)), sources, target)
    return 0


def cmd_cast_recluster(args: argparse.Namespace) -> int:
    ctx = open_series(_series(args))
    created = cast.recluster(ctx, register_episodes(ctx))
    print(f"{created} clusters besides the named ones")
    return 0


def cmd_cast_split(args: argparse.Namespace) -> int:
    cluster: int = args.cluster
    created = cast.split(open_series(_series(args)), cluster)
    print(f"cluster {cluster} split; new clusters: {', '.join(map(str, created)) or 'none'}")
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
    ingest.add_argument("--no-report", action="store_true", help="do not rebuild the report")

    scene_list = command("scenes", cmd_scenes, "list scenes with cast, expressions and lines")
    scene_list.add_argument("series", help="the series directory holding the episodes")
    scene_list.add_argument("--episode", action="append", help="only episodes containing this")

    song = command("song", cmd_song, "analyse a song: beats, bars, sections, excerpt, slots")
    song.add_argument("series", help="the series directory holding the episodes")
    song.add_argument("song", help="an audio file, or a .cue sheet of a CD image")
    song.add_argument("--track", type=int, help="track number in the .cue sheet (default 1)")
    song.add_argument("--shots", type=int, default=60, help="how many shots can fill slots")

    mad_parser = sub.add_parser("mad", help="plan and render a MAD")
    mad_sub = mad_parser.add_subparsers(required=True, metavar="action")
    plan = mad_sub.add_parser("plan", help="choose the excerpt, slots and shots; write plan.json")
    plan.set_defaults(handler=cmd_mad_plan)
    plan.add_argument("series", help="the series directory holding the episodes")
    plan.add_argument("name", help="the MAD's name (its directory under .tokimeki/mads/)")
    plan.add_argument("--song", required=True, help="an audio file, or a .cue sheet")
    plan.add_argument("--track", type=int, help="track number in the .cue sheet")
    plan.add_argument("--character", default="梦梦", help="the named cast cluster to feature")
    plan.add_argument("--episode", action="append", help="only episodes containing this")
    plan.add_argument("--start", type=float, help="excerpt start in song seconds")
    plan.add_argument("--end", type=float, help="excerpt end in song seconds")
    plan.add_argument("--beats", action="append", help="beats per slot, e.g. chorus=2")
    plan.add_argument("--boost", action="append", help="weigh an expression tag, e.g. blush=2")
    plan.add_argument("--min-presence", type=float, default=0.34, help="share of a shot she is in")
    plan.add_argument("--guidance", default="", help="free text for a model arranger")
    plan.add_argument("--arranger", default=mad.HEURISTIC, help="heuristic or model:<provider>")
    plan.add_argument("--prompt", action="store_true", help="also write the model prompt")

    report = command("report", cmd_report, "rebuild the static HTML report of a series")
    report.add_argument("series", help="the series directory holding the episodes")

    cast_parser = sub.add_parser("cast", help="list, name, merge or split character clusters")
    cast_sub = cast_parser.add_subparsers(required=True, metavar="action")

    def cast_command(name: str, handler: Handler, help_: str) -> argparse.ArgumentParser:
        p = cast_sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(handler=handler)
        p.add_argument("series", help="the series directory holding the episodes")
        return p

    cast_command("list", cmd_cast_list, "list clusters with ids, names and WD14 name hints")
    name = cast_command("name", cmd_cast_name, "name a cluster (an empty name clears it)")
    name.add_argument("cluster", type=int)
    name.add_argument("name")
    merge = cast_command("merge", cmd_cast_merge, "move the faces of clusters into another")
    merge.add_argument("source", type=int, nargs="+")
    merge.add_argument("target", type=int)
    cast_command(
        "recluster", cmd_cast_recluster, "cluster all faces outside named clusters again (no GPU)"
    )
    split = cast_command("split", cmd_cast_split, "re-cluster one cluster more tightly")
    split.add_argument("cluster", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
