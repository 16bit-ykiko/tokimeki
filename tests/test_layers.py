"""The package's dependencies point one way: cli -> stages/report -> models, media, library."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "tokimeki"

FORBIDDEN = {
    "models": {"library", "stages", "report", "cli", "media"},
    "media": {"library", "stages", "report", "cli", "models"},
    "library": {"stages", "report", "cli", "models", "media"},
    "report": {"stages", "cli", "models"},
    "stages": {"report", "cli", "song"},
    "song": {"library", "stages", "report", "cli"},
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
