from pathlib import Path

import fakes
import numpy as np
import pytest
import torch
from clips import make_clip

from tokimeki.library.episodes import stage_done
from tokimeki.library.shots import kept_ranges
from tokimeki.media.audio import decode_audio
from tokimeki.models.separation import CHUNK, N_FFT, SAMPLE_RATE, separate_chunks
from tokimeki.stages import pipeline, voice
from tokimeki.stages.base import open_series, register_episodes


def test_chunked_separation_puts_the_audio_back_together() -> None:
    t = np.arange(CHUNK * 2 + 12345) / SAMPLE_RATE
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 660 * t)])
    stereo = stereo.astype(np.float32)
    out = separate_chunks(stereo, lambda spec: spec, torch.hann_window(N_FFT))
    assert out.shape == stereo.shape
    edge = 4096
    assert np.abs(out - stereo)[:, edge:-edge].max() < 1e-3


def test_kept_gain_ramps_inside_kept_ranges() -> None:
    gain = voice.kept_gain([(0.0, 1.0), (2.0, 2.5)], 300, 100)
    assert gain[50] == 1.0 and gain[150] == 0.0 and gain[260] == 0.0
    assert gain[200] == 0.0 and 0.0 < gain[200 + 0] + gain[201] < 1.0 + 1e-6
    assert gain[249] < 1.0 and gain[240] == 1.0


def test_stage_writes_a_stem_silent_outside_kept_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_clip(tmp_path / "ep01.mkv", ["color=c=red", "color=c=blue"])
    fakes.install(monkeypatch)
    ctx = open_series(tmp_path)
    episodes = register_episodes(ctx)
    pipeline.run(ctx, episodes)
    (episode,) = episodes
    assert stage_done(ctx.conn, episode.id, voice.NAME)
    stem = voice.stem_path(ctx.paths, episode.id)
    audio = decode_audio(stem, SAMPLE_RATE, channels=2)
    gain = voice.kept_gain(kept_ranges(ctx.conn, episode), audio.shape[1], SAMPLE_RATE)
    assert 0 < (gain == 1).sum() < len(gain)
    assert np.abs(audio[:, gain == 1] - 0.1).max() < 1e-3
    assert np.abs(audio[:, gain == 0]).max() < 1e-4

    pipeline.redo(ctx, episodes, voice.NAME)
    assert not stem.exists()
