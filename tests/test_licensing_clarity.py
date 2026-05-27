"""Verify the licensing-distinction docs are in place and consistent.

Catches "we wrote the policy but never shipped the files" regressions. Each
file has a small set of required strings; the test checks they actually exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_DIR = _REPO_ROOT / "core-engine"
_DATA_DIR = _CORE_DIR / "src" / "modelmeld" / "scout" / "data"
_DOCS = _REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# Required files exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel_path",
    [
        "core-engine/LICENSE",
        "core-engine/NOTICE",
        "core-engine/README.md",
        "core-engine/src/modelmeld/scout/data/LICENSE.md",
        "core-engine/src/modelmeld/scout/data/default_registry.json",
        "docs/registry-feed.md",
        "docs/open-core-boundary.md",
    ],
)
def test_licensing_doc_exists(rel_path: str) -> None:
    assert (_REPO_ROOT / rel_path).is_file(), (
        f"Required licensing-related file missing: {rel_path}"
    )


# ---------------------------------------------------------------------------
# NOTICE file declares the data-files separation
# ---------------------------------------------------------------------------

def test_notice_declares_data_files_separate_terms() -> None:
    """NOTICE must declare the data-files-separate-terms split and
    reference the live feed by its current domain."""
    content = (_CORE_DIR / "NOTICE").read_text(encoding="utf-8")
    assert "DATA FILES" in content
    assert "separate terms" in content.lower()
    assert "CC-BY-4.0" in content
    assert "feed.modelmeld.ai" in content


def test_notice_lists_third_party_components() -> None:
    """NOTICE must list third-party dependencies with their licenses."""
    content = (_CORE_DIR / "NOTICE").read_text(encoding="utf-8")
    assert "THIRD-PARTY" in content
    # A sampling of deps we know are present
    for dep in ("FastAPI", "Pydantic", "httpx"):
        assert dep in content, f"NOTICE missing third-party dep {dep}"


# ---------------------------------------------------------------------------
# Data-dir LICENSE.md is comprehensive
# ---------------------------------------------------------------------------

def test_data_license_documents_seed_terms() -> None:
    content = (_DATA_DIR / "LICENSE.md").read_text(encoding="utf-8")
    assert "CC-BY-4.0" in content
    assert "Attribution" in content
    assert "stale" in content.lower()


def test_data_license_documents_subscription_feed_terms() -> None:
    content = (_DATA_DIR / "LICENSE.md").read_text(encoding="utf-8")
    assert "feed.modelmeld.ai" in content
    assert "subscription" in content.lower()
    assert "Ed25519" in content
    # Subscription restrictions must be enumerated
    assert "Redistribution" in content or "redistribute" in content.lower()


def test_data_license_cites_precedent_models() -> None:
    """The Maxmind-style code-vs-data pattern citation should be present so
    future maintainers understand WHY we did this."""
    content = (_DATA_DIR / "LICENSE.md").read_text(encoding="utf-8")
    assert "MaxMind" in content or "Maxmind" in content


# ---------------------------------------------------------------------------
# Seed registry carries the seed_only flag + updated notes
# ---------------------------------------------------------------------------

def test_seed_registry_marked_as_snapshot() -> None:
    """The bundled registry uses `snapshot_release_date` to mark itself
    as a frozen point-in-time copy of production-tuned data. The notes
    field explains the deflation curve and the upgrade path to the live
    feed."""
    import json
    payload = json.loads(
        (_DATA_DIR / "default_registry.json").read_text(encoding="utf-8"),
    )
    snapshot = payload.get("snapshot_release_date")
    assert isinstance(snapshot, str) and snapshot, (
        "default_registry.json must carry a snapshot_release_date"
    )
    # Loose ISO-ish date shape check
    assert len(snapshot) >= 10 and snapshot[4] == "-" and snapshot[7] == "-"
    # Notes still explain the snapshot nature and upgrade path
    notes_lower = payload.get("notes", "").lower()
    assert "snapshot" in notes_lower
    assert "feed" in notes_lower  # references the paid feed upgrade path


# ---------------------------------------------------------------------------
# open-core-boundary.md documents the three-tier distribution
# ---------------------------------------------------------------------------

def test_open_core_boundary_documents_three_tier_model() -> None:
    content = (_DOCS / "open-core-boundary.md").read_text(encoding="utf-8")
    # The bundled seed should be called out as distinct from the live feed
    assert "Bundled seed" in content or "bundled seed" in content
    assert "feed.modelmeld.ai" in content
    assert "subscription" in content.lower()


def test_open_core_boundary_explains_moat_protection_via_data() -> None:
    """The strategic note about moat-via-data must be visible."""
    content = (_DOCS / "open-core-boundary.md").read_text(encoding="utf-8")
    assert "Moat" in content or "moat" in content
    assert "curated" in content.lower()


# ---------------------------------------------------------------------------
# README explains the licensing split in plain language
# ---------------------------------------------------------------------------

def test_readme_has_licensing_section() -> None:
    content = (_CORE_DIR / "README.md").read_text(encoding="utf-8")
    assert "AGPL-3.0-or-later" in content
    assert "CC-BY-4.0" in content
    assert "Subscription" in content or "subscription" in content


def test_readme_says_everything_works_without_subscription() -> None:
    """The README has to make clear that the OSS path is genuinely usable
    standalone — otherwise the adoption flywheel doesn't fire."""
    content = (_CORE_DIR / "README.md").read_text(encoding="utf-8")
    # Some phrasing of "free path is functional" must be present
    assert any(
        phrase in content.lower()
        for phrase in (
            "everything works",
            "fully functional",
            "standalone",
            "works standalone",
        )
    )


# ---------------------------------------------------------------------------
# registry-feed.md is the user-facing operator doc
# ---------------------------------------------------------------------------

def test_registry_feed_doc_covers_both_modes() -> None:
    content = (_DOCS / "registry-feed.md").read_text(encoding="utf-8")
    assert "Bundled seed" in content or "bundled seed" in content
    assert "Live curated feed" in content or "live curated feed" in content.lower()


def test_registry_feed_doc_documents_configuration() -> None:
    """The settings table must be present + the env vars accurate."""
    content = (_DOCS / "registry-feed.md").read_text(encoding="utf-8")
    for env_var in (
        "MODELMELD_REGISTRY_FEED_URL",
        "MODELMELD_REGISTRY_FEED_LICENSE_KEY",
        "MODELMELD_REGISTRY_FEED_PUBLIC_KEY_PEM",
    ):
        assert env_var in content, f"registry-feed.md missing env var {env_var}"


def test_registry_feed_doc_explains_failure_semantics() -> None:
    content = (_DOCS / "registry-feed.md").read_text(encoding="utf-8")
    # The three FeedFetchResult sources should be documented
    for source in ("feed", "cached", "seed"):
        assert f'"{source}"' in content, (
            f"registry-feed.md doesn't document the {source!r} result source"
        )
