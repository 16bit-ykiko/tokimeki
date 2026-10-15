"""The package's dependencies point one way: cli -> api -> mad, voice -> anime -> cast -> models,
media, library."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "tokimeki"

FORBIDDEN = {
    "models": {
        "cast",
        "library",
        "anime",
        "cli",
        "media",
        "song",
        "mad",
        "api",
        "voice",
    },
    "media": {
        "cast",
        "library",
        "anime",
        "cli",
        "models",
        "song",
        "mad",
        "api",
        "voice",
    },
    "library": {
        "cast",
        "anime",
        "cli",
        "models",
        "media",
        "song",
        "mad",
        "api",
        "voice",
    },
    "cast": {"anime", "cli", "song", "mad", "api", "voice"},
    "anime": {"cli", "song", "mad", "api", "voice"},
    "song": {"library", "anime", "cli", "mad", "api", "voice"},
    "mad": {"cli", "api", "voice"},
    "voice": {"cli", "song", "mad", "api"},
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name.split(".")[1] for name in found if name.startswith("tokimeki.")}


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_imports(layer: str) -> None:
    for path in (PACKAGE / layer).rglob("*.py"):
        bad = _imports(path) & FORBIDDEN[layer]
        assert not bad, f"{path.relative_to(PACKAGE)} imports {sorted(bad)}"
