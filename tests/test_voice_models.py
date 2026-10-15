"""Transcripts, banks, engine configs and covers on synthetic audio (see `voices`)."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from voices import ToneEncoder, series

from tokimeki.library import episodes as episode_db
from tokimeki.library.series import Context
from tokimeki.models.separation import SAMPLE_RATE
from tokimeki.voice import speakers, transcripts
from tokimeki.voice.bank import BankError, bank_lines, export_bank, faded, letters
from tokimeki.voice.cover import aligned, mix, offset
from tokimeki.voice.engines import gpt_sovits
from tokimeki.voice.engines.gpt_sovits import Engine, Request, TrainSettings
from tokimeki.voice.engines.venv import SHARED
from tokimeki.voice.samples import parse_new
from tokimeki.voice.store import find_speaker
from tokimeki.voice.store import transcripts as stored_transcripts
from tokimeki.voice.tts import VoiceError, define_style, reference


class EchoTranscriber:
    """Says how long each clip is, in Japanese."""

    repo = "fake/whisper"

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def transcribe(self, clips: Sequence[NDArray[np.float32]]) -> list[str]:
        return [f"はい、{len(c) // 1600}です。" for c in clips]

    def close(self) -> None:
        pass


def _named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Context:
    ctx = series(tmp_path)
    monkeypatch.setattr(speakers, "SpeakerEncoder", ToneEncoder)
    monkeypatch.setattr(transcripts, "Transcriber", EchoTranscriber)
    speakers.run(ctx, episode_db.list_episodes(ctx.conn))
    speakers.accept(ctx)
    return ctx


def test_transcripts_cover_every_voiced_line_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _named(tmp_path, monkeypatch)
    episodes = episode_db.list_episodes(ctx.conn)
    assert transcripts.run(ctx, episodes, "fake/whisper") == 11
    assert transcripts.run(ctx, episodes, "fake/whisper") == 0
    texts = stored_transcripts(ctx.conn)
    assert len(texts) == 11 and all(t.startswith("はい") for t in texts.values())
    assert transcripts.run(ctx, episodes, "other/whisper") == 11


def test_bank_lists_her_clean_transcribed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _named(tmp_path, monkeypatch)
    with pytest.raises(BankError, match="no clean, transcribed"):
        export_bank(ctx, "梦梦")
    transcripts.run(ctx, episode_db.list_episodes(ctx.conn), "fake/whisper")
    stats = export_bank(ctx, "梦梦")
    assert (stats.lines, stats.skipped["too short"]) == (4, 0)
    root = Path(stats.directory)
    listing = (root / "gpt-sovits.list").read_text().splitlines()
    assert len(listing) == 4 and listing[0].split("|")[1:3] == ["梦梦", "ja"]
    assert Path(listing[0].split("|")[0]).exists()
    assert (root / "esd.list").read_text().splitlines()[0].split("|")[2] == "JP"
    assert [x.line for x in bank_lines(ctx.paths, "梦梦")] == [
        int(x.split("|")[0].split("-")[-1].removesuffix(".wav")) for x in listing
    ]
    with pytest.raises(VoiceError, match="no style"):
        reference(ctx, "梦梦", "teasing", None)
    first = bank_lines(ctx.paths, "梦梦")[0].line
    define_style(ctx, "梦梦", "teasing", first)
    assert reference(ctx, "梦梦", "teasing", None).line == first
    assert reference(ctx, "梦梦", None, None).style == "typical"
    assert find_speaker(ctx.conn, "梦梦").lines == 4


def test_bank_helpers() -> None:
    assert letters("えへへ、リトさん…!") == 7
    assert letters("…!?") == 0
    ramp = faded(np.ones(1000, dtype=np.float32), 10000)
    assert ramp[0] == 0.0 and ramp[500] == 1.0 and ramp[-1] == 0.0


def test_gpt_sovits_configs_follow_the_web_ui(tmp_path: Path) -> None:
    engine = Engine(tmp_path)
    template = engine.repo / f"GPT_SoVITS/configs/s2{gpt_sovits.VERSION}.json"
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps({"train": {"epochs": 100}, "model": {}, "data": {}}))
    settings = TrainSettings(sovits_epochs=6, gpt_epochs=12, sovits_batch=3, gpt_batch=5)
    s2 = gpt_sovits.s2_config(engine, tmp_path / "work", tmp_path / "out", "梦梦", settings)
    train = s2["train"]
    assert isinstance(train, dict) and train["epochs"] == 6 and train["batch_size"] == 3
    assert s2["save_weight_dir"] == str(tmp_path / "out") and s2["version"] == "v2ProPlus"
    s1 = gpt_sovits.s1_config(engine, tmp_path / "work", tmp_path / "out", "梦梦", settings)
    assert s1["train_semantic_path"] == str(tmp_path / "work/6-name2semantic.tsv")
    request = Request("こんにちは", tmp_path / "ref.wav", "はい").payload()
    assert (request["text_lang"], request["prompt_lang"], request["media_type"]) == (
        "ja",
        "ja",
        "wav",
    )
    assert not engine.installed()
    with pytest.raises(gpt_sovits.EngineError, match="not installed"):
        engine.require()
    assert "torch" in SHARED and "numpy" in SHARED


def test_new_lines_may_name_a_style() -> None:
    assert parse_new("teasing:ふふっ、リトさん") == parse_new("teasing: ふふっ、リトさん")
    assert parse_new("teasing:ふふっ").style == "teasing"
    assert parse_new("ふふっ:リトさん").style is None


def test_instrumental_is_aligned_to_the_mix_and_the_vocal_mixed_in() -> None:
    rng = np.random.default_rng(1)
    beats = np.zeros(SAMPLE_RATE * 6, dtype=np.float32)
    for at in rng.uniform(0, 5.5, 40):
        i = int(at * SAMPLE_RATE)
        beats[i : i + 2000] = rng.uniform(0.2, 0.8)
    mixture = np.stack([beats, beats])
    late = np.pad(mixture, ((0, 0), (int(0.25 * SAMPLE_RATE), 0)))
    lag = offset(mixture, late)
    assert lag == pytest.approx(0.25, abs=0.002)
    back = aligned(late, lag, mixture.shape[1])
    assert float(np.abs(back - mixture).mean()) < 0.01
    vocal = np.full(1000, 0.5, dtype=np.float32)
    out = mix(vocal, np.zeros((2, 1000), dtype=np.float32), 0.1)
    assert out.shape == (2, 1000) and out[0, 10] == pytest.approx(0.1)
    loud = mix(vocal, np.ones((2, 1000), dtype=np.float32), 1.0)
    assert np.abs(loud).max() <= 0.98 + 1e-6


def test_gpt_config_yaml_keeps_small_floats_numbers() -> None:
    text = gpt_sovits.yaml_text({"optimizer": {"lr_init": 0.00001, "steps": 2000}, "on": True})
    assert text == "optimizer:\n  lr_init: 0.00001\n  steps: 2000\non: true\n"
    assert gpt_sovits.yaml_text({"path": "/a b/梦梦"}) == 'path: "/a b/梦梦"\n'
