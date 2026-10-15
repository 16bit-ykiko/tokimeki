"""The local HTTP service: the `tokimeki.api` functions as JSON endpoints, the episodes'
browser media and the MAD outputs with HTTP Range (so a browser can scrub), and the static
pages (the voice pages, the series report).

It serves the series under the data root by directory name. On any address but loopback the
voice pages and the voice stems are left out: voice data never leaves the machine. Requests
must name the host it listens on (no DNS rebinding) and POSTs must be JSON (no cross-site form
posts). Calls that may load a GPU model or run ffmpeg on the GPU go one at a time. Routes
follow docs/design-data-and-labeling.md where that costs nothing: JSON under `/api`, episode
media under `/media`.
"""

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import BaseRoute, Route

from tokimeki import api
from tokimeki.library.paths import DATA_DIR_NAME
from tokimeki.mad.plan import mad_dir

type Json = dict[str, object]


def find_series(root: Path) -> dict[str, Path]:
    """The series under `root`: directories holding a library, by name."""
    if not root.is_dir():
        return {}
    return {
        p.name: p for p in sorted(root.iterdir()) if (p / DATA_DIR_NAME / "library.db").is_file()
    }


class _Body:
    """A JSON object from a request, read field by field with its type checked."""

    def __init__(self, data: object) -> None:
        if not isinstance(data, dict):
            raise HTTPException(400, "the body must be a JSON object")
        self.data = cast(Json, data)

    def _bad(self, key: str, expected: str) -> HTTPException:
        return HTTPException(400, f"{key}: expected {expected}")

    def text(self, key: str) -> str | None:
        value = self.data.get(key)
        if value is not None and not isinstance(value, str):
            raise self._bad(key, "a string")
        return value

    def required(self, key: str) -> str:
        value = self.text(key)
        if not value:
            raise self._bad(key, "a string")
        return value

    def number(self, key: str) -> float | None:
        value = self.data.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._bad(key, "a number")
        return float(value)

    def integer(self, key: str) -> int | None:
        value = self.data.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise self._bad(key, "an integer")
        return value

    def flag(self, key: str, default: bool) -> bool:
        value = self.data.get(key, default)
        if not isinstance(value, bool):
            raise self._bad(key, "true or false")
        return value

    def texts(self, key: str) -> list[str] | None:
        value = self.data.get(key)
        if value is None:
            return None
        items = cast(list[object], value) if isinstance(value, list) else None
        if items is None or not all(isinstance(x, str) for x in items):
            raise self._bad(key, "a list of strings")
        return cast(list[str], items)

    def weights(self, key: str) -> dict[str, float] | None:
        value = self.data.get(key)
        if value is None:
            return None
        items = cast(dict[str, object], value) if isinstance(value, dict) else None
        if items is None or not all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in items.values()
        ):
            raise self._bad(key, "an object of numbers")
        return {k: float(cast(float, v)) for k, v in items.items()}

    def counts(self, key: str) -> dict[str, int] | None:
        weights = self.weights(key)
        if weights is None:
            return None
        if not all(v.is_integer() for v in weights.values()):
            raise self._bad(key, "an object of whole numbers")
        return {k: int(v) for k, v in weights.items()}


async def _body(request: Request) -> _Body:
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "send JSON (Content-Type: application/json)")
    raw = await request.body()
    try:
        return _Body(json.loads(raw) if raw else {})
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"the body is not JSON: {error}") from error


_GPU = threading.Lock()
"""Held by calls that may run a GPU model or NVDEC/NVENC: one at a time, as the GPU rules say."""


async def _call(run: Callable[[], object], gpu: bool = False) -> Response:
    """Run an API function off the event loop; its errors become 400 with their details."""

    def guarded() -> object:
        if not gpu:
            return run()
        with _GPU:
            return run()

    try:
        return JSONResponse(await run_in_threadpool(guarded))
    except api.ApiError as error:
        return JSONResponse({"error": str(error), "details": error.details}, status_code=400)


def _file(base: Path, relative: str) -> FileResponse:
    """A file under `base` (a directory's index.html), answering Range requests."""
    root = base.resolve()
    target = (root / relative).resolve()
    if target.is_dir():
        target = target / "index.html"
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, f"no {relative or 'index'} here")
    return FileResponse(target)


async def _error(request: Request, error: Exception) -> Response:
    assert isinstance(error, HTTPException)
    return JSONResponse({"error": error.detail}, status_code=error.status_code)


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def create_app(
    series: Mapping[str, Path], voice: bool = True, hosts: Sequence[str] = LOOPBACK
) -> Starlette:
    """The service over `series` (name to directory), answering requests for `hosts`; `voice`
    serves the voice pages and stems, for loopback only."""

    def path_of(request: Request) -> str:
        name = str(request.path_params["series"])
        if name not in series:
            raise HTTPException(404, f"no series {name!r}; served: {', '.join(series) or 'none'}")
        return str(series[name])

    def episode_of(request: Request) -> int:
        return int(request.path_params["episode"])

    def media(request: Request, episode: object) -> Json:
        name = str(request.path_params["series"])
        urls: Json = {"video": f"/media/{name}/{episode}/video.mp4"}
        if voice:
            urls["voice"] = f"/media/{name}/{episode}/voice.flac"
        return urls

    async def index(_: Request) -> Response:
        return JSONResponse(
            {"series": [{"name": n, "path": str(p)} for n, p in series.items()], "voice": voice}
        )

    async def plan_schema(_: Request) -> Response:
        return await _call(api.plan_schema)

    async def song_show(request: Request) -> Response:
        song = str(request.path_params["song"])
        return await _call(lambda: api.song_show(song))

    async def song_analyze(request: Request) -> Response:
        b = await _body(request)
        audio, track, lyrics = b.required("audio"), b.integer("track"), b.texts("lyrics") or []
        backing, span, backing_track = (
            b.text("instrumental"),
            b.text("range"),
            b.integer("instrumental_track"),
        )
        stream, within, title = b.integer("stream"), b.text("within"), b.text("title")
        return await _call(
            lambda: api.song_analyze(
                audio, track, lyrics, backing, span, backing_track, stream, within, title
            ),
            gpu=True,
        )

    async def plan_validate(request: Request) -> Response:
        path, b = path_of(request), await _body(request)
        return await _call(lambda: api.plan_validate({**b.data, "series": path}))

    async def episodes(request: Request) -> Response:
        path = path_of(request)

        def run() -> Json:
            found = api.series_episodes(path)
            for e in cast(list[Json], found["episodes"]):
                e["media"] = media(request, e["id"])
            return found

        return await _call(run)

    async def episode(request: Request) -> Response:
        path, number = path_of(request), episode_of(request)

        def run() -> Json:
            return {**api.series_episode(path, number), "media": media(request, number)}

        return await _call(run)

    async def proxies(request: Request) -> Response:
        path, b = path_of(request), await _body(request)
        only = b.texts("episodes")
        return await _call(lambda: api.make_proxies(path, only), gpu=True)

    async def repeats(request: Request) -> Response:
        path = path_of(request)
        return await _call(lambda: api.series_repeats(path))

    async def plan_context(request: Request) -> Response:
        path, b = path_of(request), await _body(request)
        character, only, song = b.required("character"), b.texts("episodes"), b.text("song")
        presence = b.number("min_presence")
        boost, span, slots, beats = (
            b.weights("boost"),
            b.text("range"),
            b.integer("max_slots"),
            b.counts("beats"),
        )
        return await _call(
            lambda: api.plan_context(
                path,
                character,
                only,
                song,
                0.34 if presence is None else presence,
                boost,
                span,
                slots,
                beats,
            )
        )

    async def plan_auto(request: Request) -> Response:
        path, b = path_of(request), await _body(request)
        character, song, name = b.required("character"), b.required("song"), b.required("name")
        only, presence, boost = b.texts("episodes"), b.number("min_presence"), b.weights("boost")
        span, slots, beats = b.text("range"), b.integer("max_slots"), b.counts("beats")
        voices, fx = b.flag("voices", True), b.flag("fx", True)
        return await _call(
            lambda: api.plan_auto(
                path,
                character,
                song,
                name,
                only,
                0.34 if presence is None else presence,
                boost,
                None,
                span,
                slots,
                beats,
                voices,
                fx,
            ),
            gpu=True,
        )

    async def mads(request: Request) -> Response:
        path = path_of(request)
        return await _call(lambda: api.series_mads(path))

    def plan_file(request: Request) -> str:
        name = str(request.path_params["name"])
        plan = mad_dir(Path(path_of(request)), name) / "plan.json"
        if "/" in name or name.startswith(".") or not plan.is_file():
            raise HTTPException(404, f"no MAD {name!r}")
        return str(plan)

    async def mad_validate(request: Request) -> Response:
        plan, _ = plan_file(request), await _body(request)
        return await _call(lambda: api.plan_validate(plan))

    async def mad_refine(request: Request) -> Response:
        plan, _ = plan_file(request), await _body(request)
        return await _call(lambda: api.plan_refine(plan), gpu=True)

    async def mad_render(request: Request) -> Response:
        plan, b = plan_file(request), await _body(request)
        quality, otio, subs = b.text("quality") or "preview", b.flag("otio", False), b.text("subs")
        return await _call(lambda: api.render_plan(plan, quality, otio, subs or "all"), gpu=True)

    async def speakers(request: Request) -> Response:
        path = path_of(request)
        return await _call(lambda: api.voice_speakers(path))

    async def voice_page(request: Request) -> Response:
        path, _ = path_of(request), await _body(request)
        return await _call(lambda: api.voice_page(path))

    async def episode_file(request: Request, kind: str) -> Response:
        path, number = path_of(request), episode_of(request)
        try:
            found = await run_in_threadpool(lambda: api.episode_media(path, number))
        except api.ApiError as error:
            raise HTTPException(404, str(error)) from error
        file = found[kind]
        if not isinstance(file, str):
            hint = "; make it with `tokimeki proxy`" if kind == "video" else ""
            raise HTTPException(404, f"episode {number} has no {kind} yet{hint}")
        return FileResponse(file)

    async def video(request: Request) -> Response:
        return await episode_file(request, "video")

    async def stem(request: Request) -> Response:
        return await episode_file(request, "voice")

    def files(where: Callable[[Path], Path]) -> Callable[[Request], Response]:
        def serve(request: Request) -> Response:
            if "path" not in request.path_params:
                return RedirectResponse(f"{request.url.path}/")
            base = where(Path(path_of(request)) / DATA_DIR_NAME)
            return _file(base, str(request.path_params["path"]))

        return serve

    def mad_files(request: Request) -> Response:
        name = str(request.path_params["name"])
        base = mad_dir(Path(path_of(request)), name)
        if "/" in name or name.startswith(".") or not base.is_dir():
            raise HTTPException(404, f"no MAD {name!r}")
        if "path" not in request.path_params:
            return RedirectResponse(f"{request.url.path}/")
        return _file(base, str(request.path_params["path"]))

    routes: list[BaseRoute] = [
        Route("/", index),
        Route("/api/series", index),
        Route("/api/schema/plan", plan_schema),
        Route("/api/songs/analyze", song_analyze, methods=["POST"]),
        Route("/api/songs/{song}", song_show),
        Route("/api/{series}/plan/validate", plan_validate, methods=["POST"]),
        Route("/api/{series}/episodes", episodes),
        Route("/api/{series}/episodes/{episode:int}", episode),
        Route("/api/{series}/proxies", proxies, methods=["POST"]),
        Route("/api/{series}/repeats", repeats),
        Route("/api/{series}/plan/context", plan_context, methods=["POST"]),
        Route("/api/{series}/plan/auto", plan_auto, methods=["POST"]),
        Route("/api/{series}/mads", mads),
        Route("/api/{series}/mads/{name}/validate", mad_validate, methods=["POST"]),
        Route("/api/{series}/mads/{name}/refine", mad_refine, methods=["POST"]),
        Route("/api/{series}/mads/{name}/render", mad_render, methods=["POST"]),
        Route("/media/{series}/{episode:int}/video.mp4", video),
        Route("/mads/{series}/{name}/{path:path}", mad_files),
        Route("/mads/{series}/{name}", mad_files),
        Route("/report/{series}/{path:path}", files(lambda d: d / "report")),
        Route("/report/{series}", files(lambda d: d / "report")),
    ]
    if voice:
        routes += [
            Route("/api/{series}/speakers", speakers),
            Route("/api/{series}/voice/page", voice_page, methods=["POST"]),
            Route("/media/{series}/{episode:int}/voice.flac", stem),
            Route("/voice/{series}/{path:path}", files(lambda d: d / "voice")),
            Route("/voice/{series}", files(lambda d: d / "voice")),
        ]
    return Starlette(
        routes=routes,
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))],
        exception_handlers={HTTPException: _error},
    )


def serve(series: Mapping[str, Path], host: str, port: int) -> None:
    """Serve until interrupted; the voice routes only on loopback."""
    loopback = host in LOOPBACK
    hosts = LOOPBACK if loopback else ("*",) if host in ("0.0.0.0", "::") else (host,)
    app = create_app(series, voice=loopback, hosts=hosts)
    uvicorn.run(app, host=host, port=port, log_level="warning")
