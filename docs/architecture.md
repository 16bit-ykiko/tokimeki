# Architecture

## Code layout

Dependencies point one way: `cli` → `api` → `mad` → `stages`, `song` → `models`, `media`, `library` (and `report` → `library`, `media`). A test fails if a lower layer imports a higher one.

```
src/tokimeki/
  cli.py        the `tokimeki` command
  api.py        the agent-facing API: JSON in and out (the CLI prints it; an MCP server can wrap it)
  paths.py      where a series keeps its library, cache and report
  library/      SQLite: schema and migrations, typed records, queries
  media/        ffprobe, NVDEC decoding and frame extraction, audio, subtitles, cue sheets
  models/       the only place third-party models and untyped libraries are touched:
                typed wrappers, GPU lifecycle (one model at a time), no CPU fallback
  stages/       pipeline stages (shots, content filter, cast, lines, scenes); idempotent, resumable
  song/         song analysis: beats, bars, vocal line, sections, excerpt, slots, lyrics
  mad/          edit plans: context, schema, validation, refinement, draft arranger, render, report
  report/       the static HTML report of a series
```

Planned: `voice/` (speakers, transcripts, voice banks, voice models) and `comic/` (page scripts and voicing, behind the screenshot service), on the same layering.

## Target layout

Decided, not yet applied: the existing modules move once the voice work on `voice/bank` is merged. Shared packages are named by what they do, with no umbrella package; the product lines sit on top of them.

```
src/tokimeki/
  cast/      who each character is: face embeddings, names and aliases, per-shot fixes (anime and comic)
  library/   SQLite: schema and migrations, typed records, queries
  media/     decoding, audio, subtitles, cue sheets
  models/    typed GPU model wrappers, one model on the GPU at a time
  render/    clip rendering, mixing, subtitle burn-in (MADs and motion comics)

  anime/     the ingest pipeline: shots, content filter, faces, lines, scenes, voice stems, motion, report
  voice/     speakers, transcripts, voice banks, training (external engines), inference
  mad/       song analysis, edit plans, arrangement, MAD output
  comic/     screenshot to script (vision model plus the cast) to voiced audio
  server/    the HTTP service the phone sends screenshots to
  api.py     the agent-facing JSON API (an MCP server can wrap it)
  cli.py     the `tokimeki` command
clients/     phone-side capture: the iOS Shortcut recipe, later an Android floating button
```

The flow: `anime` ingests a series → `voice` builds banks and trains models → `server` takes a screenshot from a client, `comic` scripts it with the cast and voices it. `mad` builds on `anime`, `cast` and `render`.

## Models

Weights come from the Hugging Face Hub on first use (into the HF cache); the code that touches them lives in `src/tokimeki/models/`.

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

The deepghs and WD14 ONNX files are wrapped directly instead of going through `dghs-imgutils`, the library they were published with:

- it pins `numpy<2` and pulls in opencv-contrib, bchlib and more, which clash with the CUDA builds of PyTorch and onnxruntime from conda-forge;
- its sessions always list the CPU provider as a fallback, so a missing CUDA provider silently runs on the CPU; here a session that does not start on CUDA is an error;
- it sets no thread or GPU-memory limits and caches sessions internally, so a model cannot be freed after its batch; here each model is loaded, run and closed under one lifecycle (`models/gpu.py`).

The pre- and post-processing follow imgutils (WD14 padding and BGR order, YOLO decoding and NMS, CCIP normalisation). CCIP's pairwise metric model computes exactly (1 − cosine similarity) / 2, so it is done in numpy (`models/ccip.py`) and clustering needs no GPU. WD14 EVA02-Large rates better than SwinV2 but is 2.3× slower and peaks at 7.4 of 8 GB, so SwinV2 is the default (`models/wd14.py: MODEL_REPO`).
