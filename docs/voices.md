# Character voices

Speakers work; transcripts, voice banks and voice models are next.

## Usage

```bash
pixi run tokimeki ingest ~/anime/<series>            # the voice stems come from the `voice` stage
pixi run tokimeki voice speakers ~/anime/<series>    # cut, embed and cluster every line (GPU)
pixi run tokimeki voice list ~/anime/<series>        # speakers, minutes, seeds, suggestions
pixi run tokimeki voice page ~/anime/<series>        # the listening page, then:
pixi run tokimeki voice serve ~/anime/<series>       # http://127.0.0.1:8770/page/ (localhost only)
pixi run tokimeki voice accept ~/anime/<series>      # name speakers by their suggestion
pixi run tokimeki voice name|merge|move|fixes|recluster|lines …   # correct by ear
```

Names and moves are kept like the cast's: `recluster` and a redone `speakers` stage keep named speakers and their lines, and `move --line N --to NAME` (or `--nobody`) pins a line through every later clustering. `tokimeki voice speakers` runs after `ingest`, per episode, and skips episodes whose voices are newer than their lines and stem. The JSON API has the same data (`api.voice_speakers`), for the web UI to come.

## 1. Who speaks

- **Clips.** Each dialogue line inside kept shots and outside the OP/ED is trimmed to where its voice is in the stem (the same trimming `plan refine` uses). A line said over another line (subtitles overlapping by more than 0.15 s) is `overlap`; less than 0.6 s of voice is `short`, more than 12 s `long`, nothing in the stem `silent`; a line of 2.4 s or more whose halves sound like two people is `mixed`. Only `clean` lines get a speaker.
- **Embedding.** [litagin's anime speaker embedding](https://huggingface.co/litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm) (ECAPA-TDNN trained on 7,000 visual-novel characters, MIT), the `char` variant, through its ONNX export (`aloxaf/anime-speaker-embedding-char-onnx`, filterbank inside the graph) on onnxruntime CUDA: ep01's 221 clean lines take about 30 s. VoxCeleb models hear only "young woman" in anime voices; this one was trained for them. On ep01, with every line's true speaker, a line's nearest neighbour is the same character 94% of the time. The `va` variant (trained on voice actors, tighter per character) needs SpeechBrain to export and was not tried.
- **Clusters.** Average linkage on cosine distance, cut at 0.72 (0.70–0.75 keeps ep01's clusters 98% pure; at 0.8 Momo and Lala start to merge). The `char` model splits one character by manner of speaking (Momo's scheming monologue vs. her polite voice), so a character is often several clusters; that is fine, naming joins them. A voice left alone joins a speaker when its three nearest voices all belong to it.
- **Naming.** A *seed* is a line said while one named character, and nobody else, fills the screen in a close-up (face at least 20% of the frame, in 60% of the sampled frames) for 90% of the line. A seed alone is wrong a quarter of the time on ep01 (Rito's inner monologue over Momo's face, the listener's reaction shot), but a cluster's seeds vote: a clear majority (two or more, 60%) names it. A cluster without one is named by who is on screen unusually often while it speaks (30% of its lines, 1.3× the character's base rate), once it has three lines. `accept` turns these suggestions into names, merging clusters of one character.
- **How well.** Checked against 170 ep01 lines whose speaker the dialogue makes certain: 145 of the 146 named lines are right (99%), 86% of the lines get a name. Momo: 46/46 right, 46 of her 54 lines found; Nana 24/24, all found; Yami 13/13; Lala 18/18 of 20; Rito 28/29 of 32 (one of Mikan's lines went to him). Mikan, Haruna and the one-line characters mostly stay unnamed: too few lines to cluster.

ep01 holds 2.3 min of clean Momo speech (47 lines), 1.5 min of Rito, 1.1 min of Nana, 0.7 min each of Yami and Lala.

## 2. Transcripts

Japanese text for each line from an anime-tuned Whisper on the voice stem (the subtitles at hand are Chinese).

## 3. Voice bank

Per character: clean clips, their Japanese text and delivery tags (calm, shy, angry, shouting …), exported in the layouts TTS trainers take (`path|speaker|lang|text`).

## 4. Voice model

GPT-SoVITS v2ProPlus (MIT) fine-tuned on one bank, in its own environment behind a typed wrapper (subprocess and its local HTTP API), with a zero-shot baseline for comparison. Training and inference fit in 8 GB, one model at a time. Irodori-TTS is not used to clone: its model card forbids cloning a voice actor's voice. Voice banks, models and generated speech stay in the data directory and are never shared.
