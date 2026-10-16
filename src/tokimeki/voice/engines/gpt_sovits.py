"""GPT-SoVITS v2ProPlus (MIT) as an external engine.

It lives in its own checkout and venv under `$TOKIMEKI_HOME/engines` (see `venv`), pinned to
one commit, and is driven only through subprocesses: its dataset scripts and trainers
(configured by environment variables and config files, as its web UI does) and its HTTP API
(`api_v2.py`) on localhost for synthesis. Nothing of it is imported here.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from tokimeki.voice.engines.venv import (
    EngineError,
    build,
    checkout,
    fetch_to,
    run,
    site_packages,
)
from tokimeki.voice.paths import engines_dir

NAME = "gpt-sovits"
REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
VERSION = "v2ProPlus"
REQUIREMENTS = (
    "scipy",
    "librosa==0.10.2",
    "numba",
    "pytorch-lightning>=2.4",
    "ffmpeg-python",
    "tqdm",
    "cn2an",
    "pypinyin",
    "pyopenjtalk>=0.4.1",
    "g2p_en",
    "sentencepiece",
    "transformers>=4.51,<5",
    "peft<0.18.0",
    "chardet",
    "PyYAML",
    "psutil",
    "jieba_fast",
    "jieba",
    "split-lang",
    "fast_langdetect>=0.3.1",
    "wordsegment",
    "rotary_embedding_torch",
    "fastapi[standard]>=0.115.2",
    "x_transformers",
    "torchmetrics<=1.5",
    "pydantic<=2.10.6",
    "av>=11",
    "matplotlib",
    "tensorboard",
    "gradio<5",
)
"""The repository's requirements for Japanese training and synthesis: its ASR, UVR, Korean and
Cantonese extras are left out, and PyTorch comes from tokimeki's environment."""
WEIGHTS_REPO = "lj1995/GPT-SoVITS"
PRETRAINED = (
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "s1v3.ckpt",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "v2Pro/s2Gv2ProPlus.pth",
    "v2Pro/s2Dv2ProPlus.pth",
)
EXTRAS = (("XXXXRT/GPT-SoVITS-Pretrained", "pretrained_models/fast_langdetect/lid.176.bin"),)
"""Fetched ahead of time: GPT-SoVITS's text splitter downloads this one itself on first use,
from a host outside the Hub."""
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
"""Everything the engine needs is fetched by `install`; nothing is downloaded while it runs."""
LANGUAGE = "ja"
SERVER_START_TIMEOUT = 300.0


@dataclass(frozen=True, slots=True)
class Engine:
    root: Path

    @classmethod
    def default(cls) -> "Engine":
        return cls(engines_dir() / NAME)

    @property
    def repo(self) -> Path:
        return self.root / "repo"

    @property
    def python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    @property
    def pretrained(self) -> Path:
        return self.repo / "GPT_SoVITS" / "pretrained_models"

    @property
    def marker(self) -> Path:
        return self.root / "installed.json"

    def installed(self) -> bool:
        if not self.marker.exists() or not self.python.exists():
            return False
        data: dict[str, str] = json.loads(self.marker.read_text())
        return data.get("commit") == COMMIT

    def require(self) -> None:
        if not self.installed():
            raise EngineError(
                f"GPT-SoVITS is not installed at {self.root}; run `tokimeki voice install`"
            )


def install(engine: Engine) -> None:
    """Check out the pinned commit, build its environment and fetch the v2ProPlus weights."""
    log = engine.root / "install.log"
    checkout(REPO_URL, COMMIT, engine.repo, log)
    build(engine.root / "venv", REQUIREMENTS, log)
    # The web UI puts these on the path of every script it starts; so does this file.
    folders = ("", "GPT_SoVITS", "GPT_SoVITS/BigVGAN", "tools")
    paths = "".join(f"{engine.repo / f}\n" for f in folders)
    (site_packages(engine.root / "venv") / "gpt-sovits.pth").write_text(paths)
    for name in PRETRAINED:
        fetch_to(WEIGHTS_REPO, name, engine.pretrained)
    for repo, name in EXTRAS:
        fetch_to(repo, name, engine.repo / "GPT_SoVITS")
    # pyopenjtalk fetches its dictionary on first use; do it now rather than mid-training.
    run([engine.python, "-c", "import pyopenjtalk; pyopenjtalk.g2p('テスト')"], log)
    engine.marker.write_text(json.dumps({"commit": COMMIT, "version": VERSION}))


GPT_STEPS = 850
"""GPT steps a fine-tune should take, whatever the bank's size. On 梦梦's bank 15 epochs of
10.7 min (about 850 steps) spoke cleanly; the same 15 epochs of 19.8 min cut lines short and
babbled twice as often; 6 epochs of 26 min did best of all (see docs/voices.md)."""


@dataclass(frozen=True, slots=True)
class TrainSettings:
    sovits_epochs: int = 8
    gpt_epochs: int | None = None
    """None: as many as give the GPT about `GPT_STEPS` steps over the bank (4 to 15)."""
    sovits_batch: int = 4
    gpt_batch: int = 4

    def fitted(self, lines: int) -> "TrainSettings":
        """These settings with the GPT's epochs fixed for a bank of `lines`."""
        if self.gpt_epochs is not None:
            return self
        epochs = round(GPT_STEPS * self.gpt_batch / max(lines, 1))
        return replace(self, gpt_epochs=max(4, min(15, epochs)))


@dataclass(frozen=True, slots=True)
class Weights:
    gpt: Path
    sovits: Path


def pretrained_weights(engine: Engine) -> Weights:
    """The base model: what zero-shot synthesis runs on, and what fine-tuning starts from."""
    return Weights(engine.pretrained / "s1v3.ckpt", engine.pretrained / "v2Pro/s2Gv2ProPlus.pth")


def _dataset_env(
    engine: Engine, listing: Path, wavs: Path, work: Path, name: str
) -> dict[str, str]:
    p = engine.pretrained
    return {
        "inp_text": str(listing),
        "inp_wav_dir": str(wavs),
        "exp_name": name,
        "opt_dir": str(work),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "version": VERSION,
        "bert_pretrained_dir": str(p / "chinese-roberta-wwm-ext-large"),
        "cnhubert_base_dir": str(p / "chinese-hubert-base"),
        "sv_path": str(p / "sv/pretrained_eres2netv2w24s4ep4.ckpt"),
        "pretrained_s2G": str(p / "v2Pro/s2Gv2ProPlus.pth"),
        "s2config_path": str(engine.repo / f"GPT_SoVITS/configs/s2{VERSION}.json"),
        "hz": "25hz",
        **OFFLINE,
    }


def prepare(engine: Engine, listing: Path, wavs: Path, work: Path, name: str) -> None:
    """Phonemes, SSL features, speaker vectors and semantic tokens of a bank (`listing` in
    GPT-SoVITS list format), as the web UI's one-click "1abc" step writes them to `work`."""
    engine.require()
    env = _dataset_env(engine, listing, wavs, work, name)
    log = work / "train.log"
    scripts = engine.repo / "GPT_SoVITS" / "prepare_datasets"
    text = work / "2-name2text.txt"
    if not text.exists():
        run([engine.python, "-s", scripts / "1-get-text.py"], log, engine.repo, env)
        (work / "2-name2text-0.txt").replace(text)
    for script in ("2-get-hubert-wav32k.py", "2-get-sv.py"):
        run([engine.python, "-s", scripts / script], log, engine.repo, env)
    semantic = work / "6-name2semantic.tsv"
    if not semantic.exists():
        run([engine.python, "-s", scripts / "3-get-semantic.py"], log, engine.repo, env)
        part = work / "6-name2semantic-0.tsv"
        semantic.write_text("item_name\tsemantic_audio\n" + part.read_text().strip("\n") + "\n")
        part.unlink()


def s2_config(
    engine: Engine, work: Path, out: Path, name: str, settings: TrainSettings
) -> dict[str, object]:
    """The repository's v2ProPlus SoVITS config with the web UI's training settings."""
    template = engine.repo / f"GPT_SoVITS/configs/s2{VERSION}.json"
    data = cast(dict[str, object], json.loads(template.read_text()))
    sections = {k: cast(dict[str, object], data[k]) for k in ("train", "model", "data")}
    p = engine.pretrained
    sections["train"].update(
        {
            "batch_size": settings.sovits_batch,
            "epochs": settings.sovits_epochs,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": str(p / "v2Pro/s2Gv2ProPlus.pth"),
            "pretrained_s2D": str(p / "v2Pro/s2Dv2ProPlus.pth"),
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": settings.sovits_epochs,
            "gpu_numbers": "0",
            "grad_ckpt": False,
            "lora_rank": 32,
        }
    )
    sections["model"]["version"] = VERSION
    sections["data"]["exp_dir"] = str(work)
    data.update(
        {"s2_ckpt_dir": str(work), "save_weight_dir": str(out), "name": name, "version": VERSION}
    )
    return data


def s1_config(
    engine: Engine, work: Path, out: Path, name: str, settings: TrainSettings
) -> dict[str, object]:
    """`s1longer-v2.yaml` with the web UI's settings."""
    return {
        "train": {
            "seed": 1234,
            "epochs": settings.gpt_epochs,
            "batch_size": settings.gpt_batch,
            "save_every_n_epoch": settings.gpt_epochs,
            "precision": "16-mixed",
            "gradient_clip": 1.0,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": str(out),
            "exp_name": name,
        },
        "optimizer": {
            "lr": 0.01,
            "lr_init": 0.00001,
            "lr_end": 0.0001,
            "warmup_steps": 2000,
            "decay_steps": 40000,
        },
        "data": {"max_eval_sample": 8, "max_sec": 54, "num_workers": 2, "pad_val": 1024},
        "model": {
            "vocab_size": 1025,
            "phoneme_vocab_size": 732,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "linear_units": 2048,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1024,
            "random_bert": 0,
        },
        "inference": {"top_k": 15},
        "pretrained_s1": str(engine.pretrained / "s1v3.ckpt"),
        "train_semantic_path": str(work / "6-name2semantic.tsv"),
        "train_phoneme_path": str(work / "2-name2text.txt"),
        "output_dir": str(work / "logs_s1"),
    }


def _newest(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise EngineError(f"training wrote no {pattern} into {directory}")
    return found[-1]


def _trained_since(directory: Path, pattern: str, since: Path) -> Path | None:
    """Weights matching `pattern` written after `since` changed, if any: a stage to skip."""
    found = [p for p in directory.glob(pattern) if p.stat().st_mtime > since.stat().st_mtime]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def yaml_text(data: object, indent: int = 0) -> str:
    """Nested mappings of strings, numbers and booleans as block YAML. Floats are written
    with a decimal point, since PyYAML reads `1e-05` as a string."""
    pad = "  " * indent
    if not isinstance(data, dict):
        raise EngineError("only mappings are written as YAML here")
    out = ""
    for key, value in cast(dict[str, object], data).items():
        if isinstance(value, dict):
            out += f"{pad}{key}:\n{yaml_text(cast(dict[str, object], value), indent + 1)}"
        else:
            out += f"{pad}{key}: {_scalar(value)}\n"
    return out


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if "e" not in text else f"{value:.12f}".rstrip("0")
    return json.dumps(str(value), ensure_ascii=False)


def train(
    engine: Engine, work: Path, out: Path, name: str, settings: TrainSettings, listing: Path
) -> Weights:
    """Fine-tune SoVITS, then GPT, on a prepared dataset; the weights go to `out`. A stage
    whose weights were written after `listing` (the bank) last changed is not run again."""
    engine.require()
    settings = settings.fitted(len(listing.read_text().splitlines()))
    log = work / "train.log"
    env = {"_CUDA_VISIBLE_DEVICES": "0", "version": VERSION, "hz": "25hz", **OFFLINE}
    sovits_dir, gpt_dir = out / "sovits", out / "gpt"
    # The trainers save into these but do not create them (the web UI does).
    for folder in (sovits_dir, gpt_dir, work / f"logs_s2_{VERSION}", work / "logs_s1"):
        folder.mkdir(parents=True, exist_ok=True)
    sovits = _trained_since(sovits_dir, f"{name}_e{settings.sovits_epochs}_*.pth", listing)
    if sovits is None:
        s2 = work / "s2.json"
        s2.write_text(json.dumps(s2_config(engine, work, sovits_dir, name, settings), indent=1))
        run([engine.python, "-s", "GPT_SoVITS/s2_train.py", "--config", s2], log, engine.repo, env)
        sovits = _newest(sovits_dir, f"{name}_e{settings.sovits_epochs}_*.pth")
    gpt = _trained_since(gpt_dir, f"{name}-e{settings.gpt_epochs}.ckpt", listing)
    if gpt is None:
        s1 = work / "s1.yaml"
        s1.write_text(yaml_text(s1_config(engine, work, gpt_dir, name, settings)))
        s1_train = [engine.python, "-s", "GPT_SoVITS/s1_train.py", "--config_file", s1]
        run(s1_train, log, engine.repo, env)
        gpt = _newest(gpt_dir, f"{name}-e{settings.gpt_epochs}.ckpt")
    return Weights(gpt, sovits)


@dataclass(frozen=True, slots=True)
class Request:
    text: str
    ref_audio: Path
    prompt_text: str
    aux_refs: tuple[Path, ...] = ()
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed: float = 1.0
    seed: int = -1

    def payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "text_lang": LANGUAGE,
            "ref_audio_path": str(self.ref_audio),
            "aux_ref_audio_paths": [str(p) for p in self.aux_refs],
            "prompt_text": self.prompt_text,
            "prompt_lang": LANGUAGE,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": self.speed,
            "seed": self.seed,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "media_type": "wav",
            "streaming_mode": False,
        }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class Server:
    """`api_v2.py` serving one GPT/SoVITS weight pair on a localhost port, until closed."""

    def __init__(self, engine: Engine, weights: Weights, work: Path) -> None:
        engine.require()
        work.mkdir(parents=True, exist_ok=True)
        config = work / "tts_infer.yaml"
        p = engine.pretrained
        custom = {
            "bert_base_path": str(p / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(p / "chinese-hubert-base"),
            "device": "cuda",
            "is_half": True,
            "t2s_weights_path": str(weights.gpt),
            "version": VERSION,
            "vits_weights_path": str(weights.sovits),
        }
        config.write_text(json.dumps({"custom": custom}, indent=1))
        self.port = free_port()
        self.log = work / "server.log"
        self._out = self.log.open("w")
        self._process: subprocess.Popen[bytes] | None = subprocess.Popen(
            [
                str(engine.python),
                "-s",
                "api_v2.py",
                "-a",
                "127.0.0.1",
                "-p",
                str(self.port),
                "-c",
                str(config),
            ],
            cwd=engine.repo,
            env={**os.environ, **OFFLINE},
            stdout=self._out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self._wait()

    def _wait(self) -> None:
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                self.close()
                tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-30:])
                raise EngineError(f"the GPT-SoVITS server stopped while starting:\n{tail}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return
            except OSError:
                time.sleep(1.0)
        self.close()
        raise EngineError(f"the GPT-SoVITS server did not start within {SERVER_START_TIMEOUT:.0f}s")

    def synthesize(self, request: Request) -> bytes:
        """WAV bytes of the request's text in the voice of its reference."""
        body = json.dumps(request.payload()).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as error:
            raise EngineError(
                f"GPT-SoVITS refused the request: {error.read().decode(errors='replace')}"
            ) from error

    def close(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        self._out.close()
