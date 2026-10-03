# tokimeki

Bring the heroines you love to life wherever they appear: cut their cute moments from the anime into MADs, learn their voices, and hear them read the manga and novels you are reading.

## What it is

At the centre is the **cast** of a series: who each character is (faces, names and aliases such as 梦梦 = 茉茉 = モモ) and how she sounds (a voice bank and a voice model). Three things are built on it:

1. **Anime → scene library and MADs** (working). Give it the episodes of a series and get back every scene with who is in it, what happens, how it feels, and a preview, searchable by character, mood and line. Then cut a MAD to a song from that library: an agent arranges which scene goes where, algorithms place the cuts on the beat, and you steer it in plain words until it is right, then polish in an editor.
2. **Character voices** (planned). Every line of the anime, attributed to its speaker, transcribed and cleaned, becomes a voice bank per character, and the bank trains a voice model.
3. **Manga and novels, voiced** (planned). Read in whatever app you like; a screenshot goes to a small service on this machine, a vision model turns the page into a script (who says what, and how), and each line comes back in that character's voice.

The aim is an AI draft you steer, not a finished work with no human in the loop: a rough cut you refine, a voice you tune.

## Quick start

Needs an NVIDIA GPU (8 GB is enough) and [pixi](https://pixi.sh); on WSL, `/usr/lib/wsl/lib` must be on `PATH` (see [setup](docs/setup.md)).

```bash
pixi install
pixi run tokimeki gpu-check                  # NVDEC, PyTorch and onnxruntime run on the GPU
pixi run tokimeki ingest ~/anime/<series>    # shots, filter, cast, lines, scenes, voices, motion
pixi run tokimeki report ~/anime/<series>    # a static page to review and name the cast
```

Cutting a MAD is a handful of JSON commands an agent (or you) drives: see [MAD](docs/mad.md#usage).

## Docs

- [Scene library](docs/scene-library.md): shots, content filter, characters, lines, scenes, and the review report.
- [MAD](docs/mad.md): songs, edit plans, cut placement, voices over the song, subtitles, render.
- [Character voices](docs/voices.md) (planned): speakers, transcripts, voice banks, voice models.
- [Manga and novels, voiced](docs/comics.md) (planned): the screenshot service, page scripts, voicing.
- [Setup](docs/setup.md): environment, GPU rules, hardware and cost, where data lives.
- [Architecture](docs/architecture.md): code layout and the models used.

## Personal use only

tokimeki is for personal use. You bring the episodes, manga and novels; nothing here downloads them.

**Do not distribute AI-generated works made without authorisation.** Voiced pages, cloned voices, voice banks and voice models stay on your own machine: do not publish, upload, share or sell them, for free or otherwise. A voice belongs to its voice actor, and a work to its creators.

To share tokimeki with a friend, share the code and how to run it, never the outputs or the models.
