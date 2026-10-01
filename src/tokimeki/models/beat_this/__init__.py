"""The Beat This! network (CPJKU, MIT licence, see LICENSE), vendored from beat-this 1.1.0.

Only `model/beat_tracker.py`, `model/roformer.py` and `replace_state_dict_key` are taken,
with their imports made relative; the PyPI package would pull in torchaudio, which has no
build for this PyTorch. Not type-checked or linted (third-party code); `models/beats.py`
wraps it.
"""
