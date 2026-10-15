# Architecture

## Code layout

Dependencies point one way: `cli` → `api` → `mad`, `voice` → `anime` → `cast`, `render` → `models`, `media`, `library`. The shared packages (`cast`, `library`, `media`, `models`, `render`) never import a product line (`anime`, `voice`, `mad`); `voice` and `mad` both build on `anime`, never on each other. A test fails if a lower layer imports a higher one or a package has no layer.

```
src/tokimeki/
  cli.py        the `tokimeki` command
  api.py        the agent-facing API: JSON in and out (the CLI prints it; an MCP server can wrap it)
  selfcheck.py  `tokimeki gpu-check`: NVDEC, PyTorch and onnxruntime each prove they run on the GPU

  cast/         who each character is: face clusters from CCIP embeddings, their names, per-shot
                fixes and doubtful shots (no GPU); the average-linkage clustering speakers use too
  library/      SQLite: schema and migrations, typed records, queries; where a series keeps its
                library, cache and report (`paths.py`), and an open series (`series.py`)
  media/        ffprobe, NVDEC decoding and frame extraction, audio, subtitles, cue sheets
  models/       the only place third-party models and untyped libraries are touched:
                typed wrappers, GPU lifecycle (one model at a time), no CPU fallback
  render/       preview and final encoder settings, subtitle fonts and burn-in, the voice-over-song mix

  anime/        the ingest pipeline, idempotent and resumable: shots, content filter, faces (the `cast`
                stage), lines, scenes, voice stems (`stems.py`, the `voice` stage), motion; shots
                that recur across episodes (`repeats.py`);
                `report/` is the static HTML report of a series
  voice/        speakers (clips, embeddings, clusters, naming) and their listening page, with their
                speaker-embedding model (on `models/ort.py`) and voice tables
  mad/          `song/` analysis (beats, bars, vocal line, sections, excerpt, slots, lyrics); edit plans
                (context, schema, validation, refinement, draft arranger); the MAD's clips, sound,
                subtitles and timeline (`render.py`, on `render/`); its report
```

A MAD's clips are rendered in `mad/render.py`, not `render/`: each is cut from an episode by a plan's slot. Planned: transcripts, voice banks and voice models in `voice/`; `comic/`, `server/`, `clients/` and `web/` below.

## Target layout

Applied; `comic/`, `server/`, `clients/` and `web/` are not built yet. Shared packages are named by what they do, with no umbrella package; the product lines sit on top of them.

Where the modules went, for porting branches made before the move (stage names, CLI commands and JSON are unchanged):

| Old | New |
|---|---|
| `paths.py` | `library/paths.py` |
| `stages/base.py`: `Context`, `open_series` | `library/series.py` |
| `stages/base.py`: `register_episodes` | `anime/pipeline.py` |
| `stages/cast.py`: detection and CCIP, `run`, `reset`, `head_box`, `NAME` | `anime/faces.py` |
| `stages/cast.py`: clusters, `rename`, `merge`, `recluster`, `split`, `move`, `fixes`, `doubtful` | `cast/clusters.py` |
| `stages/clustering.py` | `cast/clustering.py` |
| `stages/voice.py` | `anime/stems.py` |
| `stages/selfcheck.py` | `selfcheck.py` |
| `stages/{content_filter,frames,lines,motion,pipeline,scenes,shots}.py` | `anime/` |
| `report/` | `anime/report/` |
| `song/` | `mad/song/` |
| `mad/mix.py`, `mad/fonts.py` | `render/mix.py`, `render/fonts.py` |
| `mad/render.py`: `RenderSettings`, `PREVIEW`, `FINAL`, `burn_subtitles`, `_ffmpeg` (now `ffmpeg`) | `render/video.py` |

```
src/tokimeki/
  cast/      who each character is: face embeddings, names and aliases, per-shot fixes (anime and comic)
  library/   SQLite: schema and migrations, typed records, queries
  media/     decoding, audio, subtitles, cue sheets
  models/    typed GPU model wrappers, one model on the GPU at a time
  render/    encoding, mixing, subtitle burn-in (MADs and motion comics)

  anime/     the ingest pipeline: shots, content filter, faces, lines, scenes, voice stems, motion, report
  voice/     speakers, transcripts, voice banks, training (external engines), inference
  mad/       song analysis, edit plans, arrangement, MAD output
  comic/     screenshot to script (vision model plus the cast) to voiced audio
  server/    the HTTP service the phone sends screenshots to
  api.py     the agent-facing JSON API (an MCP server can wrap it)
  cli.py     the `tokimeki` command
clients/     phone-side capture: the iOS Shortcut recipe, later an Android floating button
web/         local web UI in TypeScript (React, Vite, strict), served by `server/`
```

The flow: `anime` ingests a series → `voice` builds banks and trains models → `server` takes a screenshot from a client, `comic` scripts it with the cast and voices it. `mad` builds on `anime`, `cast` and `render`.

Computation stays in Python; interaction moves to `web/`: browsing the scene library, naming and fixing the cast and speakers, listening to voice samples, reviewing MAD plans. The UI talks only to the JSON API that `server/` exposes over `api.py`, and its types are generated from the same JSON Schemas (the plan schema first), so both sides cannot drift. pixi installs Node.js and pnpm, and `pixi run check` adds `tsc --noEmit` and lint. It is built after the voice work; until then pages stay static HTML.

## Models

Weights come from the Hugging Face Hub on first use (into the HF cache); the code that touches them lives in `src/tokimeki/models/` (the speaker embedding in `voice/embedding.py`).

| Stage | Model | Runs on |
|---|---|---|
| Shots | TransNetV2 (`transnetv2-pytorch`, bundled weights) | PyTorch, CUDA |
| Content filter, tags | WD14 SwinV2 v3 (`SmilingWolf/wd-swinv2-tagger-v3`) | onnxruntime, CUDA |
| Faces | `deepghs/anime_face_detection`, `face_detect_v1.4_s` (YOLOv8) | onnxruntime, CUDA |
| Characters | `deepghs/ccip_onnx`, `ccip-caformer-24-randaug-pruned` | onnxruntime, CUDA |
| Subtitle timing | Silero VAD 6 (sequence ONNX from the `silero-vad` package) | onnxruntime, CUDA |
| Beats | Beat This! `final0` (network vendored in `models/beat_this`, MIT) | PyTorch, CUDA |
| Vocal line, episode voices | MDX-Net Kim_Vocal_2 (`seanghay/uvr_models`, from UVR) | onnxruntime, CUDA (STFT in PyTorch) |
| Motion | none: luma differences of 160×90 frames | NVDEC + PyTorch, CUDA |
| Speakers | litagin anime speaker embedding, `char` (ECAPA-TDNN, ONNX export `aloxaf/anime-speaker-embedding-char-onnx`) | onnxruntime, CUDA |

The deepghs and WD14 ONNX files are wrapped directly instead of going through `dghs-imgutils`, the library they were published with:

- it pins `numpy<2` and pulls in opencv-contrib, bchlib and more, which clash with the CUDA builds of PyTorch and onnxruntime from conda-forge;
- its sessions always list the CPU provider as a fallback, so a missing CUDA provider silently runs on the CPU; here a session that does not start on CUDA is an error;
- it sets no thread or GPU-memory limits and caches sessions internally, so a model cannot be freed after its batch; here each model is loaded, run and closed under one lifecycle (`models/gpu.py`).

The pre- and post-processing follow imgutils (WD14 padding and BGR order, YOLO decoding and NMS, CCIP normalisation). CCIP's pairwise metric model computes exactly (1 − cosine similarity) / 2, so it is done in numpy (`models/ccip.py`) and clustering needs no GPU. WD14 EVA02-Large rates better than SwinV2 but is 2.3× slower and peaks at 7.4 of 8 GB, so SwinV2 is the default (`models/wd14.py: MODEL_REPO`).
