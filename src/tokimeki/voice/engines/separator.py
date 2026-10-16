"""audio-separator (MIT) as an external engine: RoFormer models that split a song's vocal far
more cleanly than MDX-Net, then its lead from the backing vocals, then the lead's reverb.

A cover converts only the dry lead, so Seed-VC hears one voice without the room or the
harmonies (which it turns into buzz). It runs from its own venv (see `venv`) through a small
driver; nothing of it is imported here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from tokimeki.voice.engines.venv import EngineError, build, run
from tokimeki.voice.paths import engines_dir

NAME = "audio-separator"
VERSION = "0.47.0"
REQUIREMENTS = (f"audio-separator=={VERSION}", "audioread")
VOCALS = "vocals_mel_band_roformer.ckpt"
"""Kim's MelBand RoFormer: vocals from the mix."""
KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
"""MelBand RoFormer karaoke: the lead from the backing vocals."""
DEREVERB = "deverb_bs_roformer_8_384dim_10depth.ckpt"
"""BS-RoFormer de-reverb: the lead without its room."""
STEPS = (
    (VOCALS, "song", {"vocals": "vocals.wav", "other": "instrumental.wav"}),
    (KARAOKE, "vocals.wav", {"vocals": "lead.wav", "instrumental": "backing.wav"}),
    (DEREVERB, "lead.wav", {"noreverb": "lead_dry.wav"}),
)
"""Each step: the model, what it separates (the song or an earlier step's stem), and the
stems kept under plain names."""
_DRIVER = """
import json, shutil, sys
from pathlib import Path
from audio_separator.separator import Separator
song, out, models = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
steps = json.loads(sys.argv[4])
work = out / "work"
work.mkdir(parents=True, exist_ok=True)
for model, source, names in steps:
    separator = Separator(model_file_dir=models, output_dir=str(work), output_format="WAV")
    separator.load_model(model_filename=model)
    for name in separator.separate(str(song if source == "song" else out / source)):
        stem = Path(name).stem.split("_(", 1)[1].split(")", 1)[0].lower()
        if stem in names:
            shutil.move(work / Path(name).name, out / names[stem])
shutil.rmtree(work)
"""


@dataclass(frozen=True, slots=True)
class Engine:
    root: Path

    @classmethod
    def default(cls) -> "Engine":
        return cls(engines_dir() / NAME)

    @property
    def python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def marker(self) -> Path:
        return self.root / "installed.json"

    def installed(self) -> bool:
        if not self.marker.exists() or not self.python.exists():
            return False
        data: dict[str, str] = json.loads(self.marker.read_text())
        return data.get("version") == VERSION


def install(engine: Engine) -> None:
    """Build its environment and fetch the three models (about 2.1 GB)."""
    log = engine.root / "install.log"
    build(engine.root / "venv", REQUIREMENTS, log)
    engine.models.mkdir(parents=True, exist_ok=True)
    cli = engine.python.parent / "audio-separator"
    for model in (VOCALS, KARAOKE, DEREVERB):
        run([cli, "--download_model_only", "-m", model, "--model_file_dir", engine.models], log)
    engine.marker.write_text(json.dumps({"version": VERSION}))


@dataclass(frozen=True, slots=True)
class Stems:
    """WAVs at the song's rate: the mix without vocals, the lead, the lead without its reverb,
    and the backing vocals."""

    instrumental: Path
    lead: Path
    lead_dry: Path
    backing: Path


def separate(engine: Engine, song: Path, out: Path) -> Stems:
    if not engine.installed():
        raise EngineError(f"{NAME} is not installed; run `tokimeki voice install {NAME}`")
    out.mkdir(parents=True, exist_ok=True)
    steps = json.dumps([[model, source, names] for model, source, names in STEPS])
    run([engine.python, "-c", _DRIVER, song, out, engine.models, steps], out / "separate.log")
    stems = Stems(
        out / "instrumental.wav", out / "lead.wav", out / "lead_dry.wav", out / "backing.wav"
    )
    missing = [
        p.name
        for p in (stems.instrumental, stems.lead, stems.lead_dry, stems.backing)
        if not p.exists()
    ]
    if missing:
        raise EngineError(f"{NAME} wrote no {', '.join(missing)} for {song.name}")
    return stems
