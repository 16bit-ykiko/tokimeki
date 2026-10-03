# Setup and environment

## Environment

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

## Hardware and cost

- Local: an RTX 3070 Ti Laptop GPU (8 GB). Shot detection, face detection, CCIP, WD14, beat analysis, Demucs and Whisper all fit; run one model at a time. NVDEC/NVENC for decoding and encoding.
- Measured on S1E01 (23:42, 1080p HEVC 10-bit): 19 min for shots, filter and cast, with the CPU fully loaded by other work and the GPU thermally throttled (SM clock ~220 MHz of 1635 during WD14 and CCIP). Roughly 3 min decoding, 8 min WD14, 6 min faces and CCIP; peak GPU memory 5 GB including the desktop.
- Cloud: only per-scene understanding (cheap model), arrangement (an agent driving `tokimeki plan …`, a few rounds per MAD) and, later, reading manga pages (a vision model, one call per screenshot). Batch scoring goes through an API, not chat sessions. Voices are synthesised locally.

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
