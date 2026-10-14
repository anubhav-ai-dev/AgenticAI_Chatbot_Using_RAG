"""Guards the package layout: layer direction, import cycles, env access, entry point."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

# package → backend packages it may import (dependencies point downward)
ALLOWED: dict[str, set[str]] = {
    "utils": {"utils"},
    "core": {"core", "utils", "models"},           # core.database imports models lazily in init()
    "models": {"models", "core", "utils"},
    "client": {"client", "core", "utils"},
    "services": {"services", "client", "core", "models", "utils"},
    "api": {"api", "services", "core", "models", "utils"},
}


def _modules() -> dict[str, Path]:
    out = {}
    for p in BACKEND.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        mod = ".".join(("backend",) + p.relative_to(BACKEND).with_suffix("").parts)
        out[mod[: -len(".__init__")] if mod.endswith(".__init__") else mod] = p
    return out


def _imports(path: Path, module_level_only: bool) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if module_level_only and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.ImportFrom) and child.module and child.module.startswith("backend"):
                found.add(child.module)
            elif isinstance(child, ast.Import):
                found.update(a.name for a in child.names if a.name.startswith("backend"))
            visit(child)

    visit(tree)
    return found


def _package(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "main"


def test_backend_root_only_contains_main():
    assert sorted(p.name for p in BACKEND.glob("*.py")) == ["__init__.py", "main.py"]
    for pkg in ("core", "utils", "client", "services", "models", "api"):
        assert (BACKEND / pkg / "__init__.py").exists(), pkg


def test_layer_direction():
    violations = []
    for mod, path in _modules().items():
        pkg = _package(mod)
        if pkg == "main" or mod == "backend.core.container":
            continue                                   # entry point / composition root may import all
        for imp in _imports(path, module_level_only=False):
            target = _package(imp)
            if target not in ALLOWED[pkg]:
                violations.append(f"{mod} -> {imp}")
            if imp == "backend.core.container" and pkg != "api":
                violations.append(f"{mod} imports the DI container (only api may)")
    assert not violations, "\n".join(violations)


def test_no_module_level_import_cycles():
    mods = _modules()
    graph = {m: {i for i in _imports(p, module_level_only=True) if i in mods} for m, p in mods.items()}
    state: dict[str, int] = {}
    cycles: list[list[str]] = []

    def dfs(node: str, trail: list[str]) -> None:
        state[node] = 1
        for nxt in graph.get(node, ()):
            if state.get(nxt) == 1:
                cycles.append(trail + [node, nxt])
            elif state.get(nxt) is None:
                dfs(nxt, trail + [node])
        state[node] = 2

    for m in graph:
        if m not in state:
            dfs(m, [])
    assert not cycles, cycles


def test_package_initialisers_do_not_import_siblings():
    for pkg in ("core", "utils", "client", "services", "models", "api"):
        tree = ast.parse((BACKEND / pkg / "__init__.py").read_text())
        assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], pkg


def test_environment_only_read_in_config():
    offenders = []
    for mod, path in _modules().items():
        if mod == "backend.core.config":
            continue
        if re.search(r"os\.(getenv|environ)", path.read_text()):
            offenders.append(mod)
    assert not offenders, offenders


def test_no_hardcoded_model_ids_outside_config():
    pattern = re.compile(r"gpt-oss|qwen3|gemini-\d|gemma-\d")
    offenders = [m for m, p in _modules().items()
                 if m != "backend.core.config" and pattern.search(p.read_text())]
    assert not offenders, offenders


@pytest.mark.parametrize("module", sorted(m for m in _modules() if not m.endswith("__init__")))
def test_every_module_imports_in_a_clean_interpreter(module):
    r = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
