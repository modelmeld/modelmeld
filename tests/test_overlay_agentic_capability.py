# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""The overlay carries a MEASURED `agentic_coding` capability on the leader rows.

Derived from the RO-3 multi-provider-correct rate (how often a model's agentic
patch threads the chosen provider's slug vs re-deriving it — the failure class the
gates were historically blind to). This corrects the hand-estimated base scores,
which ranked deepseek-v4-pro *above* glm-5 on coding while the measured data is
the reverse (glm-5 88% multi-provider-correct vs the chronic shortcut-takers).
"""
from __future__ import annotations

from modelmeld.scout.multi_provider_registry import default_multi_provider_registry


def _agentic(reg, model_id: str) -> float | None:
    for e in reg.all_entries_multi():
        if e.model_id == model_id and "agentic_coding" in e.task_scores:
            return e.task_scores["agentic_coding"]
    return None


def test_measured_agentic_coding_present_on_leaders() -> None:
    reg = default_multi_provider_registry()
    for m in ("glm-5", "kimi-k2.6", "minimax-m3", "deepseek-v4-flash",
              "deepseek-v4-pro", "hy3-preview", "mimo-v2.5", "qwen3.7-plus"):
        assert _agentic(reg, m) is not None, f"missing agentic_coding for {m}"


def test_agentic_coding_corrects_the_manual_inversion() -> None:
    reg = default_multi_provider_registry()
    glm = _agentic(reg, "glm-5")
    # The reliable OSS coders outrank the chronic shortcut-takers — the opposite
    # of what the manual base coding estimates implied.
    assert glm > _agentic(reg, "qwen3.7-plus")
    assert glm > _agentic(reg, "mimo-v2.5")
    assert _agentic(reg, "kimi-k2.6") > _agentic(reg, "hy3-preview")
    # measured values are in [0, 1]
    for m in ("glm-5", "qwen3.7-plus"):
        assert 0.0 <= _agentic(reg, m) <= 1.0
