# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression: the gateway must default to the MULTI-provider registry.

Defaulting `app.state.model_registry` to the base registry left OSS models on
their unreachable `vllm` rows only, so capability routing saw no eligible OSS
provider and fell back to frontier (-auto -> Haiku) or 400'd (-saver). The
overlay that makes Fireworks/Together/OpenRouter routable was shipped but never
loaded by the running app. This locks the wiring in place.
"""
from __future__ import annotations

from modelmeld.api.server import build_app


def test_default_app_loads_multi_provider_overlay() -> None:
    app = build_app()
    reg = app.state.model_registry

    entries_fn = getattr(reg, "all_entries_multi", None)
    assert entries_fn is not None, "default app must use a MultiProviderModelRegistry"

    hosted = {"fireworks", "together", "openrouter"}
    hosted_oss = {(e.model_id, e.provider) for e in entries_fn() if e.provider in hosted}

    # The hosted OSS rows from the overlay must be present and routable.
    assert ("deepseek-v4-pro", "openrouter") in hosted_oss
    assert len(hosted_oss) >= 5, f"expected hosted OSS rows, got {sorted(hosted_oss)}"
