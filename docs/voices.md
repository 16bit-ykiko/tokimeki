# Character voices

Speakers, transcripts, voice banks, a speaking voice model (GPT-SoVITS) and a singing one (Seed-VC) work; all of it stays on this machine.

## Usage

```bash
pixi run tokimeki ingest ~/anime/<series>            # the voice stems come from the `voice` stage
pixi run tokimeki voice speakers ~/anime/<series>    # cut, embed and cluster every line (GPU)
pixi run tokimeki voice list ~/anime/<series>        # speakers, minutes, seeds, suggestions
pixi run tokimeki voice page ~/anime/<series>        # the listening page, then:
pixi run tokimeki serve                              # http://127.0.0.1:8780/voice/<series>/page/ (localhost only)
pixi run tokimeki voice accept ~/anime/<series>      # name speakers by their suggestion
pixi run tokimeki voice name|merge|move|fixes|recluster|lines …   # correct by ear
pixi run tokimeki voice transcribe ~/anime/<series>  # Japanese text of every voiced line (GPU)
pixi run tokimeki voice bank ~/anime/<series> 梦梦    # her clips and texts in trainer formats
net-guard pixi run tokimeki voice install gpt-sovits # and seed-vc: engines in their own venvs
pixi run tokimeki voice train ~/anime/<series> 梦梦   # fine-tune GPT-SoVITS (takes the GPU)
pixi run tokimeki voice style ~/anime/<series> 梦梦 teasing --line 581   # a reference per style
pixi run tokimeki voice say ~/anime/<series> 梦梦 "日本語のテキスト" --style teasing -o out.wav
pixi run tokimeki voice samples ~/anime/<series> 梦梦 --new "teasing:…" --new "…"
pixi run tokimeki voice train-singing ~/anime/<series> 梦梦      # fine-tune Seed-VC on her bank
pixi run tokimeki voice cover ~/anime/<series> 梦梦 SONG --name NAME [--track N --instrumental …]
```

GPU jobs (`speakers`, `transcribe`, `train`, `say`, `samples`, `train-singing`, `cover`) take one model at a time; when several sessions share the card, run them under `flock ~/.local/share/tokimeki/gpu.lock`.

Names and moves are kept like the cast's: `recluster` and a redone `speakers` stage keep named speakers and their lines, and `move --line N --to NAME` (or `--nobody`) pins a line through every later clustering. `tokimeki voice speakers` runs after `ingest`, per episode, and skips episodes whose voices are newer than their lines and stem. The JSON API has the same data (`api.voice_speakers`), for the web UI to come.

## 1. Who speaks

- **Clips.** Each dialogue line inside kept shots and outside the OP/ED is trimmed to where its voice is in the stem (the same trimming `plan refine` uses). A line said over another line (subtitles overlapping by more than 0.15 s) is `overlap`; less than 0.6 s of voice is `short`, more than 12 s `long`, nothing in the stem `silent`; a line of 2.4 s or more whose halves sound like two people is `mixed`. Only `clean` lines get a speaker.
- **Embedding.** [litagin's anime speaker embedding](https://huggingface.co/litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm) (ECAPA-TDNN trained on 7,000 visual-novel characters, MIT), the `char` variant, through its ONNX export (`aloxaf/anime-speaker-embedding-char-onnx`, filterbank inside the graph) on onnxruntime CUDA: ep01's 221 clean lines take about 30 s. VoxCeleb models hear only "young woman" in anime voices; this one was trained for them. On ep01, with every line's true speaker, a line's nearest neighbour is the same character 94% of the time. The `va` variant (trained on voice actors, tighter per character) needs SpeechBrain to export and was not tried.
- **Clusters.** Average linkage on cosine distance, cut at 0.72 (0.70–0.75 keeps ep01's clusters 98% pure; at 0.8 Momo and Lala start to merge). The `char` model splits one character by manner of speaking (Momo's scheming monologue vs. her polite voice), so a character is often several clusters; that is fine, naming joins them. A voice left alone joins a speaker when its three nearest voices all belong to it.
- **Naming.** A *seed* is a line said while one named character, and nobody else, fills the screen in a close-up (face at least 20% of the frame, in 60% of the sampled frames) for 90% of the line. A seed alone is wrong a quarter of the time on ep01 (Rito's inner monologue over Momo's face, the listener's reaction shot), but a cluster's seeds vote: a clear majority (two or more, 60%) names it. A cluster without one is named by who is on screen unusually often while it speaks (30% of its lines, 1.3× the character's base rate), once it has three lines. `accept` turns these suggestions into names, merging clusters of one character.
- **How well.** Checked against 170 ep01 lines whose speaker the dialogue makes certain: 145 of the 146 named lines are right (99%), 86% of the lines get a name. Momo: 46/46 right, 46 of her 54 lines found; Nana 24/24, all found; Yami 13/13; Lala 18/18 of 20; Rito 28/29 of 32 (one of Mikan's lines went to him). Mikan, Haruna and the one-line characters mostly stay unnamed: too few lines to cluster.

ep01 holds 2.3 min of clean Momo speech (47 lines), 1.5 min of Rito, 1.1 min of Nana, 0.7 min each of Yami and Lala.

## 2. Transcripts

Japanese text for every line with a voice, from the voice stem (the subtitles at hand are Chinese): [Anime Whisper](https://huggingface.co/litagin/anime-whisper) (kotoba-whisper-v2.0 fine-tuned on 5,300 h of visual-novel lines, MIT), fp16 on CUDA, no initial prompt (it hallucinates with one). On its author's unseen visual-novel test it has 13% CER against whisper-large-v3's 16.5% and kotoba-whisper-v2.0's 18.8%, and it keeps the stammers, laughs and breaths a TTS bank should hear. S1 01–07 (1,709 lines) take 11 min. Names come out in the kanji of the games it was trained on (モモ as 桃, 春菜 as 春奈), which reads the same; a few lines come out garbled, mostly names (デビルーク).

## 3. Voice bank

`voice bank` writes a named speaker's clean, transcribed lines of 1–12 s as 44.1 kHz WAVs with a 10 ms fade, `gpt-sovits.list` (`absolute path|speaker|ja|text`), `esd.list` (`name|speaker|JP|text`, for Style-Bert-VITS2) and `bank.json` (lines, minutes per episode, what was left out), in `.tokimeki/voice/<character>/bank/`.

Read the bank's texts before training: a speaker cluster can hold another voice. On S1 01–07, Momo's 321 lines held Mikado-sensei (no named face, so her scenes with Momo gave Momo the seeds), Rito turned girl, and male classmates; 62 lines went out with `voice move --nobody` (wrong speaker, or a garbled transcript), leaving 248 lines, 11.9 min.

## 4. Voice models

External engines run from their own checkouts and venvs under `$TOKIMEKI_HOME/engines`, pinned to one commit and driven through subprocesses and local HTTP only (`voice/engines/`). Each venv is built on tokimeki's own Python with its site packages visible, so they run on the CUDA PyTorch already installed instead of a second copy; PyTorch, torchaudio and NumPy are kept out of their resolution, and the few torchaudio calls they make come from a small stand-in (`voice/engines/standin/`), since conda-forge builds no torchaudio for this PyTorch.

- **Speech: GPT-SoVITS v2ProPlus** (MIT). `voice train` runs the web UI's dataset steps (phonemes, HuBERT features, speaker vectors, semantic tokens) and fine-tunes SoVITS then GPT on the bank; `voice say` and `voice samples` start its `api_v2.py` on localhost with the trained weights (or the pretrained ones, `--zero-shot`, as the baseline) and speak like a reference clip of hers (a named `style`, or her most typical 3–10 s line). Irodori-TTS is not used to clone: its model card forbids cloning a voice actor's voice.
- **Singing: Seed-VC** (GPL-3.0, used as an external program only). Its singing model (Whisper-small content, RMVPE pitch, BigVGAN 44.1 kHz) converts a sung vocal into her voice zero-shot from 25 s of her clips, or after a short fine-tune on her bank (`voice train-singing`). `voice cover` separates the song's vocal with MDX-Net, converts it and mixes it back over the release's instrumental (aligned to the mix by envelope correlation) or over what separation left.

- **Offline at run time.** `voice install` is the only step that uses the network: it fetches the pinned checkout, the venv's packages and every model the engine loads by name (fast_langdetect's language model for GPT-SoVITS; BigVGAN and Whisper-small into the Hub cache Seed-VC's scripts point at). Training, synthesis and conversion run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Installed, GPT-SoVITS takes 3.9 GB and Seed-VC 2.8 GB.
- **Cost.** On Momo's bank (248 lines, 11.9 min) on the 8 GB laptop GPU: the dataset steps and 8 SoVITS epochs take 15 min at about 5.5 GB, 15 GPT epochs 3 min at 4.7 GB. Seed-VC's fine-tune of 800 steps (batch 2) takes 22 min, and converting the 89 s TV-size 楽園PROJECT 6 min (50 diffusion steps).
- **How well.** Scored on 16 lines (6 of her real lines, re-synthesised, and 10 new ones): the fine-tuned model sounds like her where zero-shot sounds like the reference's general type, with a speaker-embedding cosine to her bank's mean of 0.60 against zero-shot's 0.34 (her own clips: 0.53); both say the text, at 4–5% CER by anime-whisper (zero-shot 12% on the real lines). Its failures are at the start of a line (「ふふっ、リトさん」 came out as 「わたしのさん」).

Voice banks, models and generated speech stay in the data directory and are never shared.
