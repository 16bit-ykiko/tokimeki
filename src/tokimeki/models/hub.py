import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

_hub = importlib.import_module("huggingface_hub")
_download = cast(Callable[..., str], _hub.hf_hub_download)


def fetch(repo_id: str, filename: str) -> Path:
    """A model file from the Hugging Face Hub, downloaded once into the local HF cache."""
    return Path(_download(repo_id=repo_id, filename=filename))
