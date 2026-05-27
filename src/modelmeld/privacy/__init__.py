# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""PII / secret scrubbing for cloud egress.

Public surface:
    Scrubber          — abstract base class
    RegexScrubber     — pattern-based default implementation
    Redaction         — counted match record
    build_scrubber(s) — factory keyed on GatewaySettings.pii_scrub_cloud
"""

from __future__ import annotations

from modelmeld.privacy.scrubber import Redaction, RegexScrubber, Scrubber


def build_scrubber(settings: object) -> Scrubber | None:
    """Construct a Scrubber based on settings, or None if scrubbing is disabled."""
    from modelmeld.config import GatewaySettings

    if not isinstance(settings, GatewaySettings):
        raise TypeError(
            f"build_scrubber expects GatewaySettings, got {type(settings).__name__}"
        )
    if not settings.pii_scrub_cloud:
        return None
    return RegexScrubber()


__all__ = ["Redaction", "RegexScrubber", "Scrubber", "build_scrubber"]
