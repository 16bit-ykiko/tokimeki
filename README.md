# tokimeki

Find the cute moments of anime heroines across whole series, and cut them into MADs.

Stage 1 is a **scene library**: give it the episodes of a series, get back every scene with who is in it, what happens, how it feels, and a preview, searchable by character, mood and line. Stage 2 cuts a MAD to a song from that library: a model arranges which scene goes where, algorithms place the cuts on the beat, and you steer it in plain words until it is right, then polish in an editor.

The aim is an AI rough cut you refine, not a finished video with no human in the loop. What makes a good MAD (cuts on the beat, picture answering the lyrics, pacing) comes from that loop.

First series: _To LOVE-Ru Darkness_ and _Darkness 2nd_ (12 + 14 = 26 episodes).

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
   pixi run tokimeki cast doubtful ~/anime/to-love-ru-darkness 梦梦      # her shots that look most like someone else
   pixi run tokimeki cast move ~/anime/to-love-ru-darkness --shot 173 --shot 179 --to 娜娜   # or --nobody
   pixi run tokimeki cast fixes ~/anime/to-love-ru-darkness            # the moves made so far
   ```

   Look-alikes still end up together now and then (side-ponytail Nana in Momo's cluster). `cast move` records that in a shot the faces of one named character belong to another (or to nobody), moves them, and applies the fix again after every recluster, split, merge or cast redo; everything downstream (scenes, `plan context`, validation) reads the corrected cast. Fixes are kept per shot and between named clusters, so they last until the episode's shots are redone. `cast doubtful` ranks a cluster's shots by how much closer their faces sit to another named cluster than to the rest of their own, with a keyframe to look at; on S1E01 it lists 173 and 179 among its first five, next to real Momo shots, so it is a review aid, not a judge.
4. **Lines and speakers.** External ASS subtitles next to the video (same basename, `.ass`) when they exist; otherwise (TODO) faster-whisper (Japanese) on the main audio track, never a commentary track. The subtitles are checked against the speech: Silero VAD (ONNX on CUDA) over the main track, cross-correlated with "a line is showing"; a shift is applied only when the match is clear and at least 0.1 s (S1E01: +0.03 s, correlation 0.74, speech starts a median 0.034 s after a line, so VCB timing is right). Lines attach to the shots they overlap (`shot_lines` view). Speakers come from clustering voices and matching the clusters to characters; the actor field is not reliable (the Darkness subtitles put the placeholder `NTP` on every line). Lyric styles (`opjp`/`opcn`/`edjp`/`edcn` in the Darkness subtitles) mark the OP/ED time ranges for free. Subtitle timing must be checked against the video before it is trusted.
5. **Scenes.** In a romantic comedy the cute moment is usually an exchange and a reaction, not one shot. Consecutive kept shots are merged into scenes by dialogue continuity (a line spans the cut, or the speech pauses at most 1.5 s) and, without dialogue, only by clearly matching colours; scenes never span a dropped shot or the edge of the OP/ED, and stop at 90 s. Measured on S1E01, WD14 tag vectors and colour histograms barely separate "same exchange" from "next scene", so dialogue does most of the work (160 scenes from 286 kept shots). Each scene summarises its named cast (share of screen time), its lines and its expression tags (`tokimeki scenes`).
6. **Understanding.**
   - Cheap, local, everywhere: WD14 expression tags (`smile`, `blush`, `:d`, `pout`, `wink`, `>_<` …) per frame; the original BGM's mood (comedic, warm, tense) from an audio embedding such as CLAP, as a weak extra signal.
   - Cheap, cloud, per scene: a low-cost model (Gemini Flash, Haiku, DeepSeek …) reads the lines and a few keyframes and writes a one-line summary with mood and type tags (embarrassed, jealous, clingy, confession …). Text-heavy, a few cents per episode.
   - Personal taste: you mark scenes 👍/👎 in the library; a small ranker learns from it.
7. **Library.** One SQLite database per series (shots, scenes, cast, tags, summaries, lines) and a simple web page to filter and preview.

## Stage 2: MAD

1. **Song.** Any media file: an audio file, a track of a CD image (`.cue`), or a song inside an episode (`--within 20:37-21:58` gives a rough window; the exact start, the sharpest attack, and end, the drop into silence, are found from the audio; `--stream` picks the track, by default the main one, never commentary). The 楽園PROJECT OP of S1E01 is taken this way: 1231.05 s for 89.39 s. Beat This! (on CUDA) gives beats and downbeats; bars follow the downbeat phase most detections agree on (4/4). `allin1` is not used: it needs natten and madmom, which do not build for this PyTorch. Sections come from novelty in the bar self-similarity matrix plus the vocal line, and the loudest repeated sung section is the chorus. The vocal line comes from the mix itself: MDX-Net (UVR's Kim_Vocal_2, ONNX on CUDA) separates the voice and each bar's share of vocal-band energy that is voice says whether it is sung (Silero VAD does not hear singing over a dense mix). Optional hints replace it: lyrics (which also give the lyric timeline and, by lines sung more than once, the chorus) or an instrumental version (`--instrumental file`, or `--instrumental-track N` of the same `.cue`). On MORE&MORE, separation agrees with the instrumental-derived line on 95.5% of bars; both give intro 0.9–16 s and verse to 58.2 s, and separation ends the first chorus at 85.5 s rather than 81.1 s. The default excerpt runs from the top through the first chorus within 45–90 s (`--range` picks another). Suggested slots: a cut a bar in verses and every two beats in the chorus, slowed section by section (verse first, chorus last, never past 4 s a slot) to the number of usable shots. Analyses live in a song store shared by all series (`$TOKIMEKI_HOME/songs/<id>/`, default `~/.local/share/tokimeki/songs`). TODO: energy accents within a bar.
2. **Arrangement (an agent, through tools).** Whoever arranges (an agent driving the CLI, later an MCP server over `tokimeki/api.py`, or a person) gets one JSON context, writes a plan, and checks it. The plan (`tokimeki plan schema`) is the single source of truth: song id and, per slot, song start/end on beats, the shot, the source `in`/`out` (episode seconds) and a short `why`. `plan validate` reports every problem with its slot and a stable code (gaps, off-beat boundaries, too-short slots, unknown, dropped or repeated shots, OP/ED shots, windows outside the shot, speed outside 0.9–1.1×); `plan refine` puts boundaries exactly on beats and fills missing windows; `plan auto` writes a heuristic draft to start from (everyday moments in the verse, the cutest close-ups in the chorus, her best smile last, story order, framing alternating). Only kept shots outside the OP/ED appear in the context, so only they can reach a cloud model. TODO: picture answering the lyrics.
3. **Cut placement.** Each slot gets a whole number of output frames counted from the excerpt's start, so every cut lands on the frame nearest its beat. A clip with no window yet shows the stretch of its shot centred on the cutest sampled frame (expression tags weighted by `CUTE`), two frames off the shot's edges, slowed to as little as 0.9× when the shot is a little short. TODO: land motion onsets (head turns, blinks, jumps) on the beat.
4. **Sound.** The song is the main track; lines worth keeping are separated with Demucs and placed in the song's gaps, with the music ducked.
5. **Picture.** Mostly hard cuts; an occasional flash or push-in on strong beats; slow pan/zoom on still shots; a 9:16 crop from the face boxes if wanted.
5b. **Lyric subtitles.** Lyrics come as `--lyrics PATH[#STYLE,STYLE][@LANG]`, repeatable: LRC files (song clock) or ASS lines of the given styles (the media's clock, moved by the song's start), e.g. an OP's `opjp` Japanese and `opcn` Chinese. Lines in different languages that start together are paired; `plan context` lists them so the picture can answer the lyrics. Render writes `lyrics.ass` (Japanese above, Chinese below, white with a soft magenta outline, short fades, no karaoke) and `lyrics.srt` next to the video, and burns the ASS in through libass unless `--subs none` (NVDEC in, libass on the CPU, NVENC out). Fonts come from the system or, under WSL, Windows (Yu Gothic, Microsoft YaHei; Noto Sans CJK otherwise) and are copied into `.tokimeki/fonts/`, never the repository.
6. **Render and iterate.** The edit plan JSON is the single source of truth. Each clip is decoded with NVDEC, scaled and retimed on the GPU and encoded with NVENC on its own, cached in `cache/clips/` by a hash of its parameters, then the clips are joined and the song excerpt laid under them; a change re-renders only the clips it touches (`tokimeki render plan.json --preview|--final|--both [--otio]`: a 640×360 preview and the 1080p final, `timeline.otio` for an editor, and `report.html` with every cut, its frame and why it was chosen). A plan can only use kept shots: anything else is refused before rendering. Low-res previews first; you say what to change ("more embarrassed ones in the second chorus", "too choppy here"), the plan is edited, the preview re-rendered. Final: full-quality render plus an OpenTimelineIO export for DaVinci Resolve (or a CapCut/剪映 draft) for hand polish — one way: edits made there do not come back.

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

### A MAD (for agents and people)

Every command prints JSON; `plan validate` exits 1 when the plan has errors.

```bash
S=~/anime/to-love-ru-darkness
E="$S/s1/[VCB-Studio] To LOVE-Ru Darkness [01][Ma10p_1080p][x265_flac_aac]"
pixi run tokimeki song analyze "$E.mkv" --within 20:37-21:58 --title 楽園PROJECT \
  --lyrics "$S/.tokimeki/subs/zh-hant/${E##*/}.ass#opjp@ja" --lyrics "$E.ass#opcn@zh"   # any audio file or .cue --track N too
pixi run tokimeki plan context $S --character 梦梦 --episode 01 --song project-d3e6341c --range 41.76-89.39 --max-slots 20 > context.json
pixi run tokimeki plan schema                                               # the plan format
pixi run tokimeki plan auto $S --character 梦梦 --episode 01 --song project-d3e6341c --name momo-ep1 --range 41.76-89.39 --max-slots 20 --min-presence 0.5   # a draft to edit
pixi run tokimeki plan validate plan.json
pixi run tokimeki plan refine plan.json                                     # beats exact, missing windows filled
pixi run tokimeki render plan.json --preview --otio                         # then --final; --subs none for no lyrics
```

`--range` picks the excerpt for this MAD (song seconds or m:ss, snapped to bar lines; the stored one otherwise), `--max-slots` caps the cuts so only the best shots are needed (slots grow, verse first, up to 4 s), `--beats chorus=4` fixes the beats per slot of a section kind. The context lists the song (sections, beats, lyrics, suggested slots) and every candidate scene of the character: its lines with times, and per shot the time range, framing, how much of it she is in, her face size, her cutest moments with WD14 expression tags, and the path of a keyframe to look at. A plan may leave `in`/`out` out; render fills them as `plan refine` would. Outputs go to `<series>/.tokimeki/mads/<name>/`.

## Layout and data

- The repository holds code only. Episodes, songs and everything derived live in a data directory outside it.
- Episodes stay where you put them and are only read. Everything derived goes into a hidden `.tokimeki/` next to them:

  ```
  ~/anime/to-love-ru-darkness/        # your episodes (*.mkv …), never written to
    .tokimeki/
      library.db                      # the series library (SQLite)
      cache/frames/<episode id>/      # sampled frames as JPEG; regenerable, safe to delete
      cache/clips/<hash>.mp4          # rendered clips, reused while their parameters stay the same
      mads/<name>/                    # plan.json, preview.mp4, final.mp4, lyrics.ass/.srt, timeline.otio, report.html
      fonts/                          # subtitle fonts copied from the system
      subs/                           # subtitle files unpacked from archives (inputs for lyrics)
      report/index.html               # the static review page and its images
  ```
- Song analyses are shared by every series: `$TOKIMEKI_HOME/songs/<id>/analysis.json` (default `~/.local/share/tokimeki/songs`).
- Keep media on the WSL filesystem, not `/mnt/c`: reading through the Windows mount is slow.

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

## Hardware and cost

- Local: an RTX 3070 Ti Laptop GPU (8 GB). Shot detection, face detection, CCIP, WD14, beat analysis, Demucs and Whisper all fit; run one model at a time. NVDEC/NVENC for decoding and encoding.
- Measured on S1E01 (23:42, 1080p HEVC 10-bit): 19 min for shots, filter and cast, with the CPU fully loaded by other work and the GPU thermally throttled (SM clock ~220 MHz of 1635 during WD14 and CCIP). Roughly 3 min decoding, 8 min WD14, 6 min faces and CCIP; peak GPU memory 5 GB including the desktop.
- Cloud: only per-scene understanding (cheap model) and arrangement (an agent driving `tokimeki plan …`, a few rounds per MAD). Batch scoring goes through an API, not chat sessions.

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
| Vocal line | MDX-Net Kim_Vocal_2 (`seanghay/uvr_models`, from UVR) | onnxruntime, CUDA (STFT in PyTorch) |

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
- Beat This!'s PyPI package needs torchaudio, which conda-forge does not build for this PyTorch, and pixi's lock check rejects overrides that drop a dependency; so its two network files are vendored (`models/beat_this`, excluded from lint and type checks) and its mel front end is reimplemented in `models/beats.py` (matches torchaudio to 2e-5, tested).
