# tokimeki

Find the cute moments of anime heroines across whole series, and cut them into MADs.

Stage 1 is a **scene library**: give it the episodes of a series, get back every scene with who is in it, what happens, how it feels, and a preview, searchable by character, mood and line. Stage 2 cuts a MAD to a song from that library: a model arranges which scene goes where, algorithms place the cuts on the beat, and you steer it in plain words until it is right, then polish in an editor.

The aim is an AI rough cut you refine, not a finished video with no human in the loop. What makes a good MAD (cuts on the beat, picture answering the lyrics, pacing) comes from that loop.

First series: _To LOVE-Ru Darkness_, seasons 1–2 (24 episodes).

## Stage 1: scene library

1. **Shots.** NVDEC decodes every frame on the GPU, scaled down there to 48×27; TransNetV2 (PyTorch, CUDA) splits the episode into shots. Each shot is sampled every 0.5 s, at least 3 frames, into the frame cache at up to 720p. _TODO:_ OP/ED and recaps repeat every episode; perceptual hashes across episodes will drop them (within one episode they cannot be told apart from the story).
2. **Content filter.** WD14 (SwinV2 v3, on CUDA) rates every sampled frame; one frame whose `questionable` + `explicit` scores reach `UNSAFE_THRESHOLD` (0.2, conservative; `stages/content_filter.py`) drops the whole shot. This runs before anything else sees the frames: a dropped shot keeps only its time range and the dropped flag, its frames are deleted from the cache, and it never reaches face detection, the report or a cloud model. There is no censor-and-keep path. Kept frames store their rating scores and WD14 general tags (for expressions later) and character tags.
3. **Characters.** Anime face detection (deepghs YOLOv8) on kept frames, then CCIP embeddings of a square head crop (hair tells anime characters apart better than the face alone), both ONNX on CUDA (see [Models](#models)). Clusters are series-wide: average-linkage agglomerative clustering (CCIP difference, cut at `CLUSTER_THRESHOLD` = 0.15) runs over a new episode's faces together with the existing clusters, each standing in as a fixed group of exemplars, so faces join the clusters they match and the rest form new ones. You name each cluster once from a few thumbnails and the name carries over to later episodes. (DBSCAN, CCIP's suggested method, chained a whole episode into one cluster through ambiguous faces; average linkage does not.) WD14 character tags on single-face frames are shown as name hints. Per shot: who, face size (close-up or wide), share of the sampled frames they appear in (the `shot_cast` view).

   ```bash
   pixi run tokimeki cast list  ~/anime/to-love-ru-darkness            # ids, names, face counts, WD14 hints
   pixi run tokimeki cast name  ~/anime/to-love-ru-darkness 3 "Momo"
   pixi run tokimeki cast merge ~/anime/to-love-ru-darkness 7 9 3      # move clusters 7 and 9 into 3
   pixi run tokimeki cast split ~/anime/to-love-ru-darkness 3          # re-cluster 3 more tightly
   pixi run tokimeki cast recluster ~/anime/to-love-ru-darkness        # redo all unnamed clusters (no GPU, seconds)
   ```
4. **Lines and speakers.** External ASS subtitles when the release has them; otherwise faster-whisper (Japanese). Speakers from the subtitle's actor field, or by clustering voices and matching clusters to characters.
5. **Scenes.** In a romantic comedy the cute moment is usually an exchange and a reaction, not one shot. Consecutive shots are merged into scenes by dialogue continuity and visual similarity.
6. **Understanding.**
   - Cheap, local, everywhere: WD14 expression tags (`smile`, `blush`, `:d`, `pout`, `wink`, `>_<` …) per frame; the original BGM's mood (comedic, warm, tense) from an audio embedding such as CLAP, as a weak extra signal.
   - Cheap, cloud, per scene: a low-cost model (Gemini Flash, Haiku, DeepSeek …) reads the lines and a few keyframes and writes a one-line summary with mood and type tags (embarrassed, jealous, clingy, confession …). Text-heavy, a few cents per episode.
   - Personal taste: you mark scenes 👍/👎 in the library; a small ranker learns from it.
7. **Library.** One SQLite database per series (shots, scenes, cast, tags, summaries, lines) and a simple web page to filter and preview.

## Stage 2: MAD

1. **Song.** Beats, bars and sections (verse, chorus, bridge) with `allin1` or beat_this; energy curve and accents; lyrics timeline from an LRC file or Whisper. Output: a timeline of slots (e.g. a cut per bar in verses, every two beats in the chorus).
2. **Arrangement (model).** From the slots, the lyrics and the library's text, a model writes the edit plan: which scene goes in which slot and why. Everyday moments early, the cutest in the chorus, the emotional ones in the bridge, a signature smile to close; no repeats, close-ups alternating with wide shots, picture answering the lyrics.
3. **Cut placement (algorithm).** Within each chosen shot, pick the stretch whose expression peaks (per-frame tag scores), and land motion onsets (head turns, blinks, jumps; frame difference or optical flow) on the beat. Speed 0.9–1.1× to fit; anime animated on twos and threes hides it.
4. **Sound.** The song is the main track; lines worth keeping are separated with Demucs and placed in the song's gaps, with the music ducked.
5. **Picture.** Mostly hard cuts; an occasional flash or push-in on strong beats; slow pan/zoom on still shots; a 9:16 crop from the face boxes if wanted.
6. **Render and iterate.** The edit plan JSON is the single source of truth. Each clip is rendered on its own with ffmpeg (NVENC), cached by a hash of its parameters, then concatenated and mixed; a change re-renders only the clips it touches. Low-res previews first; you say what to change ("more embarrassed ones in the second chorus", "too choppy here"), the plan is edited, the preview re-rendered. Final: full-quality render plus an OpenTimelineIO export for DaVinci Resolve (or a CapCut/剪映 draft) for hand polish — one way: edits made there do not come back.

## Sources

You provide the episodes; nothing here downloads them. Prefer releases without burned-in subtitles (text at the bottom of the frame is hard to work around) and with external ASS subtitles, Japanese if possible.

## Usage

Put the episodes of a series in one directory (subdirectories are fine), then:

```bash
pixi run tokimeki gpu-check                                # once: is everything on the GPU?
pixi run tokimeki ingest ~/anime/to-love-ru-darkness       # run every stage; reruns skip finished work
pixi run tokimeki ingest ~/anime/to-love-ru-darkness --episode 01 --redo shots   # redo one stage (and the later ones)
pixi run tokimeki status ~/anime/to-love-ru-darkness
pixi run tokimeki report ~/anime/to-love-ru-darkness       # ingest rebuilds it too
explorer.exe "$(wslpath -w ~/anime/to-love-ru-darkness/.tokimeki/report/index.html)"
```

The report is one static page in the data directory: totals (shots, kept, dropped and the drop rate, with no images of dropped shots), the character clusters with sample faces, ids and WD14 name hints, and every kept shot with a thumbnail, time range, cast (framing and presence) and top WD14 tags. It is rebuilt from scratch each time, so no image outlives a shot the filter later drops.

## Layout and data

- The repository holds code only. Episodes, songs and everything derived live in a data directory outside it.
- Episodes stay where you put them and are only read. Everything derived goes into a hidden `.tokimeki/` next to them:

  ```
  ~/anime/to-love-ru-darkness/        # your episodes (*.mkv …), never written to
    .tokimeki/
      library.db                      # the series library (SQLite)
      cache/frames/<episode id>/      # sampled frames as JPEG; regenerable, safe to delete
      report/index.html               # the static review page and its images
  ```
- Keep media on the WSL filesystem, not `/mnt/c`: reading through the Windows mount is slow.

## Code layout

Dependencies point one way: `cli` → `stages` / `report` → `models`, `media`, `library`.

```
src/tokimeki/
  cli.py        the `tokimeki` command
  paths.py      where a series keeps its library, cache and report
  library/      SQLite: schema and migrations, typed records, queries
  media/        ffprobe, NVDEC decoding and frame extraction, frame sampling
  models/       the only place third-party models and untyped libraries are touched:
                typed wrappers, GPU lifecycle (one model at a time), no CPU fallback
  stages/       pipeline stages (shots, content filter, cast); each idempotent and resumable
  report/       the static HTML report
```

## Hardware and cost

- Local: an RTX 3070 Ti Laptop GPU (8 GB). Shot detection, face detection, CCIP, WD14, beat analysis, Demucs and Whisper all fit; run one model at a time. NVDEC/NVENC for decoding and encoding.
- Measured on S1E01 (23:42, 1080p HEVC 10-bit): 19 min for shots, filter and cast, with the CPU fully loaded by other work and the GPU thermally throttled (SM clock ~220 MHz of 1635 during WD14 and CCIP). Roughly 3 min decoding, 8 min WD14, 6 min faces and CCIP; peak GPU memory 5 GB including the desktop.
- Cloud: only per-scene understanding (cheap model) and arrangement (a strong model, a few rounds per MAD). Batch scoring goes through an API, not chat sessions.

## Models

Weights come from the Hugging Face Hub on first use (into the HF cache); the code that touches them lives in `src/tokimeki/models/`.

| Stage | Model | Runs on |
|---|---|---|
| Shots | TransNetV2 (`transnetv2-pytorch`, bundled weights) | PyTorch, CUDA |
| Content filter, tags | WD14 SwinV2 v3 (`SmilingWolf/wd-swinv2-tagger-v3`) | onnxruntime, CUDA |
| Faces | `deepghs/anime_face_detection`, `face_detect_v1.4_s` (YOLOv8) | onnxruntime, CUDA |
| Characters | `deepghs/ccip_onnx`, `ccip-caformer-24-randaug-pruned` | onnxruntime, CUDA |

The deepghs and WD14 ONNX files are wrapped directly instead of going through `dghs-imgutils`, the library they were published with:

- it pins `numpy<2` and pulls in opencv-contrib, bchlib and more, which clash with the CUDA builds of PyTorch and onnxruntime from conda-forge;
- its sessions always list the CPU provider as a fallback, so a missing CUDA provider silently runs on the CPU; here a session that does not start on CUDA is an error;
- it sets no thread or GPU-memory limits and caches sessions internally, so a model cannot be freed after its batch; here each model is loaded, run and closed under one lifecycle (`models/gpu.py`).

The pre- and post-processing follow imgutils (WD14 padding and BGR order, YOLO decoding and NMS, CCIP normalisation). CCIP's pairwise metric model computes exactly (1 − cosine similarity) / 2, so it is done in numpy (`models/ccip.py`) and clustering needs no GPU. WD14 EVA02-Large rates better than SwinV2 but is 2.3× slower and peaks at 7.4 of 8 GB, so SwinV2 is the default (`models/wd14.py: MODEL_REPO`).

## Development

[pixi](https://pixi.sh) manages the environment: Python 3.12, a CUDA build of ffmpeg (NVDEC, `scale_cuda`), PyTorch and onnxruntime, all from conda-forge.

```bash
pixi install
pixi run tokimeki gpu-check   # NVDEC, PyTorch and onnxruntime each prove they run on the GPU
pixi run check                # ruff lint + format check + basedpyright strict
pixi run test                 # GPU tests are skipped where there is no GPU
```

- The environment is solved for CUDA 12.9 (`platforms = [{ platform = "linux-64", cuda = "12.9" }]`), not just `12`: conda-forge's CUDA build of ffmpeg requires `__cuda >= 12.8`, and with `cuda = "12"` the solver quietly picks the build without NVDEC or `scale_cuda`. `tokimeki gpu-check` fails if that ever happens.
- pixi finds the driver through `nvidia-smi`; on WSL that lives in `/usr/lib/wsl/lib`, which must be on `PATH` (or set `CONDA_OVERRIDE_CUDA=12.9`), otherwise pixi refuses to run anything.
- Nothing heavy falls back to the CPU: without NVDEC or CUDA the pipeline stops with an error. CPU thread pools are kept small (`models.CPU_THREADS`).
- One model is on the GPU at a time and may use at most `GPU_MEMORY_LIMIT` (4 GiB, `models/gpu.py`). Past the card's memory the Windows driver pages to system RAM and everything crawls; the cap turns that into an error. Setting *CUDA – Sysmem Fallback Policy* to *Prefer No Sysmem Fallback* in the NVIDIA Control Panel does the same driver-wide.
- CI type-checks in the small `lint` environment (CPU PyTorch: same annotations, no CUDA libraries).
