"""Find the cute moments of anime heroines across a series and cut them into MADs."""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from tokimeki import api
from tokimeki.report import build_report
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


def _emit(run: Callable[[], object]) -> int:
    """Print an API result as JSON; errors too, with exit status 1."""
    try:
        result = run()
    except api.ApiError as error:
        print(
            json.dumps(
                {"error": str(error), "details": error.details}, ensure_ascii=False, indent=1
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


def _boosts(values: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values or []:
        tag, _, weight = value.partition("=")
        out[tag.strip()] = float(weight)
    return out


def cmd_song_analyze(args: argparse.Namespace) -> int:
    audio: str = args.audio
    track: int | None = args.track
    lyrics: list[str] = args.lyrics or []
    instrumental: str | None = args.instrumental
    span: str | None = args.range
    backing_track: int | None = args.instrumental_track
    stream: int | None = args.stream
    within: str | None = args.within
    title: str | None = args.title
    return _emit(
        lambda: api.song_analyze(
            audio, track, lyrics, instrumental, span, backing_track, stream, within, title
        )
    )


def cmd_song_show(args: argparse.Namespace) -> int:
    song: str = args.song
    return _emit(lambda: api.song_show(song))


def cmd_plan_schema(_: argparse.Namespace) -> int:
    return _emit(api.plan_schema)


def cmd_plan_context(args: argparse.Namespace) -> int:
    series: str = args.series
    character: str = args.character
    episodes: list[str] | None = args.episode
    song: str | None = args.song
    presence: float = args.min_presence
    boost = _boosts(args.boost)
    return _emit(lambda: api.plan_context(series, character, episodes, song, presence, boost))


def cmd_plan_auto(args: argparse.Namespace) -> int:
    series: str = args.series
    character: str = args.character
    song: str = args.song
    name: str = args.name
    episodes: list[str] | None = args.episode
    presence: float = args.min_presence
    boost = _boosts(args.boost)
    output: str | None = args.output
    return _emit(
        lambda: api.plan_auto(series, character, song, name, episodes, presence, boost, output)
    )


def cmd_plan_validate(args: argparse.Namespace) -> int:
    plan: str = args.plan
    status = 0

    def run() -> object:
        nonlocal status
        report = api.plan_validate(plan)
        status = 0 if report["ok"] else 1
        return report

    return _emit(run) or status


def cmd_plan_refine(args: argparse.Namespace) -> int:
    plan: str = args.plan
    output: str | None = args.output
    return _emit(lambda: api.plan_refine(plan, output))


def cmd_render(args: argparse.Namespace) -> int:
    plan: str = args.plan
    quality: str = args.quality
    otio: bool = args.otio
    subs: str = args.subs
    return _emit(lambda: api.render_plan(plan, quality, otio, subs))


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

    song = sub.add_parser("song", help="analyse songs (JSON out)")
    song_sub = song.add_subparsers(required=True, metavar="action")
    analyze = song_sub.add_parser("analyze", help="beats, bars, sections, energy, excerpt, slots")
    analyze.set_defaults(handler=cmd_song_analyze)
    analyze.add_argument("audio", help="any media file: audio, a .cue image, an episode …")
    analyze.add_argument("--track", type=int, help="track of a .cue sheet (default 1)")
    analyze.add_argument(
        "--within", help="rough window of the song in the file, e.g. 20:37-21:58; edges found"
    )
    analyze.add_argument("--stream", type=int, help="audio stream index (default: main track)")
    analyze.add_argument("--title", help="the song's title (default: from the file)")
    analyze.add_argument(
        "--lyrics",
        action="append",
        help="PATH[#STYLE,STYLE][@LANG]: LRC, or ASS lines of those styles (repeatable)",
    )
    analyze.add_argument(
        "--instrumental", help="hint: the song without vocals (default: separate the mix)"
    )
    analyze.add_argument(
        "--instrumental-track", type=int, help="hint: the instrumental's track of the same .cue"
    )
    analyze.add_argument("--range", help="the excerpt, e.g. 0:58-1:21 (default: through chorus 1)")
    show = song_sub.add_parser("show", help="print a stored analysis")
    show.set_defaults(handler=cmd_song_show)
    show.add_argument("song", help="song id")

    plan = sub.add_parser("plan", help="write and check edit plans (JSON in and out)")
    plan_sub = plan.add_subparsers(required=True, metavar="action")
    schema = plan_sub.add_parser("schema", help="the plan's JSON Schema")
    schema.set_defaults(handler=cmd_plan_schema)
    context = plan_sub.add_parser("context", help="everything an arranger needs, as one JSON")
    context.set_defaults(handler=cmd_plan_context)
    auto = plan_sub.add_parser("auto", help="a draft plan from the heuristic arranger")
    auto.set_defaults(handler=cmd_plan_auto)
    for p in (context, auto):
        p.add_argument("series", help="the series directory holding the episodes")
        p.add_argument("--character", required=True, help="a named cast cluster, e.g. 梦梦")
        p.add_argument("--episode", action="append", help="only episodes containing this")
        p.add_argument("--min-presence", type=float, default=0.34, help="share of a shot she is in")
        p.add_argument("--boost", action="append", help="weigh an expression tag, e.g. blush=2")
    context.add_argument("--song", help="song id from `song analyze`, to include its slots")
    auto.add_argument("--song", required=True, help="song id from `song analyze`")
    auto.add_argument("--name", required=True, help="the MAD's name")
    auto.add_argument("-o", "--output", help="where to write the plan (default: the data dir)")
    check = plan_sub.add_parser("validate", help="every problem with a plan; exit 1 on errors")
    check.set_defaults(handler=cmd_plan_validate)
    check.add_argument("plan", help="plan.json")
    refine = plan_sub.add_parser("refine", help="snap slots to beats, fill windows around peaks")
    refine.set_defaults(handler=cmd_plan_refine)
    refine.add_argument("plan", help="plan.json")
    refine.add_argument("-o", "--output", help="write here instead of over the plan")

    render = command("render", cmd_render, "render a plan into the series' data dir")
    render.add_argument("plan", help="plan.json")
    quality = render.add_mutually_exclusive_group()
    for q in ("preview", "final", "both"):
        quality.add_argument(f"--{q}", dest="quality", action="store_const", const=q)
    render.set_defaults(quality="preview")
    render.add_argument("--otio", action="store_true", help="also write timeline.otio")
    render.add_argument(
        "--subs", choices=("lyrics", "none"), default="lyrics", help="burn in lyric subtitles"
    )

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
