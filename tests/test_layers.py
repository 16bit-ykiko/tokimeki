"""The package's dependencies point one way: cli -> server -> api -> mad, voice -> anime -> cast,
render -> models, media, library. Shared packages never import a product line."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "tokimeki"

TOP = {"api", "cli", "selfcheck"}
PRODUCTS = {"anime", "voice", "mad", "comic", "server"}
ABOVE = PRODUCTS | TOP

FORBIDDEN = {
    "models": {"media", "library", "cast", "render"} | ABOVE,
    "media": {"models", "library", "cast", "render"} | ABOVE,
    "library": {"models", "media", "cast", "render"} | ABOVE,
    "cast": {"render"} | ABOVE,
    "render": {"cast"} | ABOVE,
    "anime": {"voice", "mad", "comic", "server"} | TOP,
    "voice": {"mad", "comic", "server"} | TOP,
    "mad": {"voice", "comic", "server"} | TOP,
    "server": {"cli", "selfcheck"},
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module == "tokimeki":
            found.update(f"tokimeki.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name.split(".")[1] for name in found if name.startswith("tokimeki.")}


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_imports(layer: str) -> None:
    for path in (PACKAGE / layer).rglob("*.py"):
        bad = _imports(path) & FORBIDDEN[layer]
        assert not bad, f"{path.relative_to(PACKAGE)} imports {sorted(bad)}"


def test_every_package_has_a_layer() -> None:
    packages = {p.parent.name for p in PACKAGE.glob("*/__init__.py")}
    assert packages <= FORBIDDEN.keys()
