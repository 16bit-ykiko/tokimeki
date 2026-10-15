# Server

`tokimeki serve` runs a small HTTP service (Starlette on uvicorn) over the same functions the CLI and agents use (`tokimeki/api.py`): JSON in and out, the episodes' media and the MAD outputs with HTTP Range so a browser can scrub, and the static pages. There is no web UI yet; `web/` comes after the data and labeling design is settled.

```bash
pixi run tokimeki serve                         # every series under the data root, on 127.0.0.1:8780
pixi run tokimeki serve ~/anime/<series> --port 8780
pixi run tokimeki proxy ~/anime/<series> --episode 01   # a browser-playable copy to scrub (GPU)
```

It listens on 127.0.0.1. Given another `--host`, it leaves out the voice pages and the voice stems: voice data never leaves the machine. A series is addressed by its directory name, an episode by its id in the library.

| Route | What |
|---|---|
| `GET /api/series` | the series served |
| `GET /api/{series}/episodes` | episodes: path, frame rate, length, finished stages, OP/ED, media URLs |
| `GET /api/{series}/episodes/{id}` | one episode with every shot: range, kept or dropped, named cast, how it recurs |
| `GET /api/{series}/repeats` | shots that recur across episodes (`tokimeki repeats`) |
| `POST /api/{series}/proxies` | make the proxies (`{"episodes": ["01"]}`; GPU) |
| `POST /api/{series}/plan/context` · `/plan/auto` | `plan context` and `plan auto`, the CLI's options as JSON fields (`range`, `max_slots`, `beats`, …) |
| `GET /api/{series}/mads` | the MADs and their files |
| `POST /api/{series}/mads/{name}/validate` · `/refine` · `/render` | on its `plan.json`; render takes `quality`, `otio`, `subs` |
| `POST /api/plan/validate` | a plan given as the body |
| `GET /api/schema/plan` | the plan's JSON Schema |
| `GET /api/songs/{id}` · `POST /api/songs/analyze` | `song show`, `song analyze` |
| `GET /api/{series}/speakers` · `POST /api/{series}/voice/page` | `voice speakers`, `voice page` (loopback only) |
| `GET /media/{series}/{id}/video.mp4` | the episode's proxy |
| `GET /media/{series}/{id}/voice.flac` | its voice stem (loopback only) |
| `GET /mads/{series}/{name}/…` | the MAD's outputs: `report.html`, `preview.mp4`, `final.mp4`, `plan.json`, … |
| `GET /voice/{series}/…` | the voice pages, e.g. `/voice/{series}/page/` (loopback only) |
| `GET /report/{series}/` | the series report |

An API error answers 400 with `{"error", "details"}`, as the CLI prints it. Long calls (render, song analysis, proxies) hold their request until done.

**Proxies.** The episodes as released (HEVC Main10 in MKV with FLAC) do not play in a browser, and they hold the shots the content filter dropped. `tokimeki proxy` writes `cache/proxy/<episode id>.mp4`: H.264 at 540p (NVDEC and `scale_cuda` in, NVENC out) with a keyframe every half second, AAC stereo of the main track, `+faststart`, and black and silent outside kept shots. It is made again when the kept shots change. The server never serves an episode file itself.
