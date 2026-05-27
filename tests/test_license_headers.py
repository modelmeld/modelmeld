"""Assert every source file carries the right license header.

Two invariants:

  1. Every `.py` under `modelmeld/` carries the AGPL-3.0-or-later SPDX
     header (so the OSS distribution is unambiguous about licensing on a
     per-file basis — required by some downstream procurement reviews).

  2. Every `.py` under `modelmeld_enterprise/` carries the proprietary
     "All rights reserved" header (so any file that escapes the private
     repository is identifiable as proprietary at first glance).

The applier script `scripts/apply_license_headers.py --check` does the
same enumeration; this test runs it within the test suite so a missing
header is a build break, not a thing reviewers have to remember to run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLIER = _REPO_ROOT / "scripts" / "apply_license_headers.py"
_OSS_ROOT = _REPO_ROOT / "core-engine" / "src" / "modelmeld"
_ENT_ROOT = _REPO_ROOT / "enterprise-control" / "src" / "modelmeld_enterprise"

_OSS_MARKER = "SPDX-License-Identifier: AGPL-3.0-or-later"
_ENT_MARKER = "Proprietary and confidential"

# Internal-tooling tests: rely on `scripts/apply_license_headers.py` (not
# shipped in OSS) and the monorepo layout. Skip the whole module when we're
# running from the OSS-flat layout.
if not _APPLIER.is_file():
    pytest.skip(
        "Internal license-applier tests skipped (OSS layout omits apply_license_headers.py)",
        allow_module_level=True,
    )


def _head(path: Path, n_lines: int = 8) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:n_lines])


def _walk(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_applier_script_exists() -> None:
    """The header-applier script must exist (it's the canonical fix tool
    referenced in CONTRIBUTING.md + pre-commit + this test)."""
    assert _APPLIER.is_file(), f"missing {_APPLIER}"


def test_applier_check_mode_passes() -> None:
    """Running the applier in --check mode against the live tree must
    exit 0 — that's the same gate as the applier script in CI."""
    result = subprocess.run(
        [sys.executable, str(_APPLIER), "--check"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"apply_license_headers.py --check failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.parametrize("py_file", _walk(_OSS_ROOT), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_oss_files_have_agpl_header(py_file: Path) -> None:
    """Every Python file shipped in `modelmeld` declares AGPL-3.0-or-later."""
    head = _head(py_file)
    assert _OSS_MARKER in head, (
        f"{py_file.relative_to(_REPO_ROOT)} is missing the AGPL-3.0-or-later "
        f"SPDX header. Fix: python scripts/apply_license_headers.py"
    )


@pytest.mark.parametrize("py_file", _walk(_ENT_ROOT), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_enterprise_files_have_proprietary_header(py_file: Path) -> None:
    """Every Python file in `modelmeld_enterprise` is marked proprietary."""
    head = _head(py_file)
    assert _ENT_MARKER in head, (
        f"{py_file.relative_to(_REPO_ROOT)} is missing the proprietary "
        f"header. Fix: python scripts/apply_license_headers.py"
    )


def test_oss_does_not_carry_proprietary_marker() -> None:
    """Belt-and-suspenders: no OSS file should have the proprietary marker
    (would indicate a copy-paste mistake when adding new modules)."""
    leaked = []
    for path in _walk(_OSS_ROOT):
        if _ENT_MARKER in _head(path):
            leaked.append(path.relative_to(_REPO_ROOT))
    assert not leaked, (
        f"OSS files carry the PROPRIETARY marker — copy-paste error: "
        f"{leaked}"
    )


def test_enterprise_does_not_carry_oss_marker() -> None:
    """Inverse belt-and-suspenders: enterprise files shouldn't have the
    AGPL OSS SPDX line (avoids confusion about licensing)."""
    leaked = []
    for path in _walk(_ENT_ROOT):
        if _OSS_MARKER in _head(path):
            leaked.append(path.relative_to(_REPO_ROOT))
    assert not leaked, (
        f"Enterprise files carry the AGPL OSS marker: {leaked}"
    )
