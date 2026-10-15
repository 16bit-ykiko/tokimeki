# torchaudio stand-in for the engines

The external engines (GPT-SoVITS, Seed-VC) run on tokimeki's own PyTorch (a conda-forge CUDA
build, shared through a venv with system site packages) so no second copy of PyTorch is
installed. conda-forge has no torchaudio for that PyTorch, and the engines use only a few of
its calls: `load`, `save`, `functional.resample`, `transforms.Resample` and
`compliance.kaldi.fbank`. This package provides them; `install` copies it next to the engines
and puts it on their path with a `.pth` file.

`compliance/kaldi.py` is torchaudio's (BSD-2-Clause), as GPT-SoVITS also bundles it. The rest
is written here. None of it is imported by tokimeki, so it is left out of ruff and basedpyright.
