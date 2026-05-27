#!/usr/bin/env python3
"""Open-core boundary linter. See docs/open-core-boundary.md."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _find_default_root() -> Path:
    """Find the modelmeld source root. Supports both layouts:

    - Monorepo (private): `<repo>/core-engine/src/modelmeld/`
    - OSS public repo: `<repo>/src/modelmeld/`

    Returns whichever path exists. Falls through to the OSS layout if
    neither is found — caller's existence check then surfaces the
    error.
    """
    monorepo_path = REPO_ROOT / "core-engine" / "src" / "modelmeld"
    oss_path = REPO_ROOT / "src" / "modelmeld"
    if monorepo_path.is_dir():
        return monorepo_path
    return oss_path


DEFAULT_ROOT = _find_default_root()
DEFAULT_FORBIDDEN: tuple[str, ...] = ("modelmeld_enterprise",)


def iter_py_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.py")


def _matches(module: str, forbidden: Iterable[str]) -> bool:
    return any(module == f or module.startswith(f + ".") for f in forbidden)


def find_forbidden_imports(
    path: Path, forbidden: Iterable[str]
) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"syntax error: {exc.msg}")]

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches(alias.name, forbidden):
                    violations.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _matches(module, forbidden):
                violations.append((node.lineno, f"from {module} import ..."))
        # Dynamic-import patterns — catching what AST-Import doesn't.
        # An adversarial PR could otherwise bypass the static check via
        # __import__, importlib.import_module, or sys.modules subscript.
        elif isinstance(node, ast.Call):
            descr = _dynamic_import_violation(node, forbidden)
            if descr:
                violations.append((node.lineno, descr))
        elif isinstance(node, ast.Subscript):
            descr = _sys_modules_violation(node, forbidden)
            if descr:
                violations.append((node.lineno, descr))
    return violations


def _string_constant(node: ast.AST) -> str | None:
    """Return the string value of an ast.Constant, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dynamic_import_violation(
    call: ast.Call, forbidden: Iterable[str]
) -> str | None:
    """Detect `__import__("forbidden...")` or `importlib.import_module("forbidden...")`."""
    if not call.args:
        return None
    target = _string_constant(call.args[0])
    if not target or not _matches(target, forbidden):
        return None
    # __import__("...")
    if isinstance(call.func, ast.Name) and call.func.id == "__import__":
        return f"__import__({target!r})"
    # importlib.import_module("...")
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "import_module"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "importlib"
    ):
        return f"importlib.import_module({target!r})"
    return None


def _sys_modules_violation(
    sub: ast.Subscript, forbidden: Iterable[str]
) -> str | None:
    """Detect `sys.modules["forbidden..."]`."""
    if not (
        isinstance(sub.value, ast.Attribute)
        and sub.value.attr == "modules"
        and isinstance(sub.value.value, ast.Name)
        and sub.value.value.id == "sys"
    ):
        return None
    target = _string_constant(sub.slice)
    if not target or not _matches(target, forbidden):
        return None
    return f"sys.modules[{target!r}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory tree to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--forbidden",
        action="append",
        default=None,
        help=(
            "Module prefix forbidden inside --root (repeatable). "
            f"Default: {' '.join(DEFAULT_FORBIDDEN)}"
        ),
    )
    args = parser.parse_args(argv)

    forbidden = tuple(args.forbidden) if args.forbidden else DEFAULT_FORBIDDEN
    root: Path = args.root

    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        return 2

    total = 0
    for py_file in iter_py_files(root):
        for lineno, descr in find_forbidden_imports(py_file, forbidden):
            rel = py_file.relative_to(root.parent) if py_file.is_relative_to(root.parent) else py_file
            print(
                f"{rel}:{lineno}: BOUNDARY VIOLATION: {descr}",
                file=sys.stderr,
            )
            total += 1

    if total:
        print(
            f"\n{total} boundary violation(s). "
            f"/core-engine MUST NOT import from {', '.join(forbidden)}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
