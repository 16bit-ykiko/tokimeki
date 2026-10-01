# tokimeki

Find the cute moments of anime heroines across whole series, and cut them into MADs.

Stage 1 is a **scene library**: give it the episodes of a series, get back every scene with who is in it, what happens, how it feels, and a preview, searchable by character, mood and line. Stage 2 cuts a MAD to a song from that library: a model arranges which scene goes where, algorithms place the cuts on the beat, and you steer it in plain words until it is right, then polish in an editor.

The aim is an AI rough cut you refine, not a finished video with no human in the loop. What makes a good MAD (cuts on the beat, picture answering the lyrics, pacing) comes from that loop.

First series: _To LOVE-Ru Darkness_, seasons 1–2 (24 episodes).

## Stage 1: scene library

1. **Shots.** TransNetV2 (or PySceneDetect) splits each episode into shots. OP/ED and recaps repeat every episode; perceptual hashes drop them.
2. **Content filter.** WD14 tags sampled frames; shots rated questionable or explicit are dropped here, before anything else sees them. They never enter the library and are never sent to a cloud model.
3. **Characters.** Anime face detection plus CCIP embeddings (`dghs-imgutils`). Faces across the whole series are clustered; you name each cluster once from a few thumbnails, which gives the cast without reference images. Per shot: who, face size (close-up or wide), share of screen time.
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

## Layout and data

- The repository holds code only. Episodes, songs and everything derived live in a data directory outside it (e.g. `~/anime/<series>/`).
- Per series: one SQLite database for the library; regenerable intermediates (sampled frames, embeddings) in a cache that can be deleted and rebuilt.
- Keep media on the WSL filesystem, not `/mnt/c`: reading through the Windows mount is slow.

## Hardware and cost

- Local: an RTX 3070 Ti Laptop GPU (8 GB). Shot detection, face detection, CCIP, WD14, beat analysis, Demucs and Whisper all fit; run one model at a time. NVDEC/NVENC for decoding and encoding. A season is roughly an hour of compute, once.
- Cloud: only per-scene understanding (cheap model) and arrangement (a strong model, a few rounds per MAD). Batch scoring goes through an API, not chat sessions.

## Development

[pixi](https://pixi.sh) manages the environment (Python 3.12, ffmpeg, and later PyTorch with CUDA from conda-forge).

```bash
pixi install
pixi run check   # ruff lint + format check + basedpyright strict
pixi run test
```
