"""Seed-VC (GPL-3.0) as an external engine for singing voice conversion.

Its singing model (DiT on Whisper-small content features with an RMVPE pitch track, BigVGAN
at 44.1 kHz) converts a sung vocal into the voice of a reference clip zero-shot, and a short
fine-tune on one character's bank tightens the timbre. It runs from its own checkout and venv
(see `venv`) through its `train.py` and `inference.py`; nothing of it is imported or vendored.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokimeki.voice.engines.venv import EngineError, build, checkout, run
from tokimeki.voice.paths import engines_dir

NAME = "seed-vc"
REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
REQUIREMENTS = (
    "accelerate",
    "huggingface-hub>=0.28.1",
    "munch==4.0.0",
    "einops==0.8.0",
    "descript-audio-codec==1.0.0",
    "pydub==0.25.1",
    "hydra-core==1.3.2",
    "pyyaml",
    "python-dotenv",
    "scipy",
    "librosa==0.10.2.post1",
    "soundfile",
    "transformers>=4.46,<5",
)
CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
SINGING = ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
CHECKPOINTS = (
    SINGING,
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
    ("Plachta/Seed-VC", "se_db.pt"),
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt"),
    ("funasr/campplus", "campplus_cn_common.bin"),
    ("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth"),
    ("myshell-ai/OpenVoiceV2", "converter/config.json"),
)
"""Files Seed-VC reads from its `./checkpoints` Hub cache."""
SNAPSHOTS = (
    ("nvidia/bigvgan_v2_44khz_128band_512x", ("config.json", "bigvgan_generator.pt")),
    ("openai/whisper-small", ("*.json", "*.txt", "model.safetensors")),
)
"""Models it loads by name; its scripts point the Hub cache at `./checkpoints/hf_cache`."""
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
"""Everything is fetched by `install`; nothing is downloaded while it trains or converts."""
_PREFETCH = """
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
files, snapshots = sys.argv[1], sys.argv[2]
for repo, name in (x.split("::") for x in files.split(",")):
    hf_hub_download(repo_id=repo, filename=name, cache_dir="./checkpoints")
for spec in snapshots.split(","):
    repo, patterns = spec.split("::")
    snapshot_download(repo, allow_patterns=patterns.split("|"), cache_dir="./checkpoints/hf_cache")
"""


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
                f"Seed-VC is not installed at {self.root}; run `tokimeki voice install seed-vc`"
            )

    def singing_model(self) -> Path:
        repo, name = SINGING
        found = sorted(
            (self.repo / "checkpoints").glob(
                f"models--{repo.replace('/', '--')}/snapshots/*/{name}"
            )
        )
        if not found:
            raise EngineError(f"Seed-VC's singing model {name} is missing; reinstall the engine")
        return found[-1]


def install(engine: Engine) -> None:
    """Check out the pinned commit, build its environment and fetch the singing models."""
    log = engine.root / "install.log"
    checkout(REPO_URL, COMMIT, engine.repo, log)
    build(engine.root / "venv", REQUIREMENTS, log)
    files = ",".join(f"{repo}::{name}" for repo, name in CHECKPOINTS)
    snaps = ",".join(f"{repo}::{'|'.join(p)}" for repo, p in SNAPSHOTS)
    run([engine.python, "-c", _PREFETCH, files, snaps], log, engine.repo)
    engine.marker.write_text(json.dumps({"commit": COMMIT}))


def fine_tune(engine: Engine, wavs: Path, name: str, steps: int, out: Path) -> Path:
    """Fine-tune the singing model on a directory of one voice's clips; the model goes to `out`."""
    engine.require()
    log = out.parent / "train.log"
    runs = engine.repo / "runs" / name
    if runs.exists():
        shutil.rmtree(runs)
    args = [engine.python, "train.py", "--config", CONFIG, "--dataset-dir", wavs]
    args += ["--pretrained-ckpt", engine.singing_model(), "--run-name", name]
    args += ["--batch-size", "2", "--max-steps", str(steps), "--save-every", str(steps)]
    run(args, log, engine.repo, OFFLINE)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runs / "ft_model.pth", out)
    return out


@dataclass(frozen=True, slots=True)
class Conversion:
    source: Path
    reference: Path
    semitones: int = 0
    steps: int = 50
    cfg_rate: float = 0.7


def convert(engine: Engine, job: Conversion, checkpoint: Path | None, out: Path) -> Path:
    """Sing `job.source` (a dry vocal) in the voice of `job.reference`; a WAV at 44.1 kHz."""
    engine.require()
    work = out.parent / f".{out.stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    args = [engine.python, "inference.py", "--source", job.source, "--target", job.reference]
    args += ["--output", work, "--diffusion-steps", str(job.steps), "--length-adjust", "1.0"]
    args += ["--inference-cfg-rate", str(job.cfg_rate), "--f0-condition", "True"]
    args += ["--auto-f0-adjust", "False", "--semi-tone-shift", str(job.semitones), "--fp16", "True"]
    args += ["--checkpoint", checkpoint or engine.singing_model(), "--config", CONFIG]
    run(args, out.parent / "convert.log", engine.repo, OFFLINE)
    found = sorted(work.glob("*.wav"))
    if not found:
        raise EngineError(f"Seed-VC wrote nothing for {job.source.name}")
    found[0].replace(out)
    shutil.rmtree(work)
    return out
