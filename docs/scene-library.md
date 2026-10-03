# Anime, stage 1: scene library

1. **Shots.** NVDEC decodes every frame on the GPU, scaled down there to 48×27; TransNetV2 (PyTorch, CUDA) splits the episode into shots. Each shot is sampled every 0.5 s, at least 3 frames, into the frame cache at up to 720p. _TODO:_ OP/ED and recaps repeat every episode; perceptual hashes across episodes will drop them (within one episode they cannot be told apart from the story).
2. **Content filter.** WD14 (SwinV2 v3, on CUDA) rates every sampled frame; one frame whose `questionable` + `explicit` scores reach `UNSAFE_THRESHOLD` (0.2, conservative; `stages/content_filter.py`) drops the whole shot. This runs before anything else sees the frames: a dropped shot keeps only its time range and the dropped flag, its frames are deleted from the cache, and it never reaches face detection, the report or a cloud model. There is no censor-and-keep path. Kept frames store their rating scores and WD14 general tags (for expressions later) and character tags.
3. **Characters.** Anime face detection (deepghs YOLOv8) on kept frames, then CCIP embeddings of a square head crop (hair tells anime characters apart better than the face alone), both ONNX on CUDA (see [Models](architecture.md#models)). Clusters are series-wide: average-linkage agglomerative clustering (CCIP difference, cut at `CLUSTER_THRESHOLD` = 0.15) runs over a new episode's faces together with the existing clusters, each standing in as a fixed group of exemplars, so faces join the clusters they match and the rest form new ones. You name each cluster once from a few thumbnails and the name carries over to later episodes. (DBSCAN, CCIP's suggested method, chained a whole episode into one cluster through ambiguous faces; average linkage does not.) WD14 character tags on single-face frames are shown as name hints. Per shot: who, face size (close-up or wide), share of the sampled frames they appear in (the `shot_cast` view).

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

## Sources

Put the episodes of a series in one directory; nothing here downloads them. Prefer releases without burned-in subtitles (text at the bottom of the frame is hard to work around) and with external ASS subtitles (same basename as the video), Japanese if possible.

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
