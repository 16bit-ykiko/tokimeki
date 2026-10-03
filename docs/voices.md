# Character voices (planned)

1. **Who speaks.** The `voice` stage already separates each episode's voice from music and effects. Each line's voice gets a speaker embedding on the GPU; lines are clustered by speaker and the clusters matched to named characters by who is on screen while they are spoken. You confirm a cluster from a few clips and name it once, as with faces. MADs gain from it too: a plan can ask for her lines only.
2. **Transcripts.** Japanese text for each line from faster-whisper on the voice stem (the subtitles at hand are Chinese).
3. **Voice bank.** Per character: clean clips, their Japanese text and delivery tags (calm, shy, angry, shouting …), lines with overlapping voices left out, exported in the layout TTS trainers take.
4. **Voice model.** A Japanese TTS fine-tuned on one bank: Style-Bert-VITS2 JP-Extra for a stable character voice with emotion sliders, GPT-SoVITS when only a minute or so of audio exists, Irodori-TTS for the most expressive acting (sighs, whispers, laughter). Training and inference fit in 8 GB, one model at a time.
