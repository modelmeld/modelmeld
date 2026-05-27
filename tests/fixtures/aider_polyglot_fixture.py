"""Synthetic Aider Polyglot YAML fixture."""

from __future__ import annotations

# Approximates the shape of Aider's `edit_leaderboard.yml`. Real entries have
# more fields; we include the ones we actually parse plus a few realistic
# extras to make sure the parser doesn't break on them.
AIDER_YAML_BASELINE = """
- dirname: 2026-04-12--claude-opus-4-7-test
  model: anthropic/claude-opus-4-7
  edit_format: diff
  commit_hash: abc123
  pass_rate_1: 75.2
  pass_rate_2: 81.0
  percent_cases_well_formed: 99.6
  error_outputs: 1
  num_user_asks: 0
  total_cost: 186.50
  released: 2026-04-12
- dirname: 2026-04-15--gpt-5-mini
  model: openai/gpt-5-mini
  edit_format: diff
  pass_rate_2: 62.4
  percent_cases_well_formed: 95.1
  total_cost: 12.30
  released: 2026-04-15
- dirname: 2026-05-01--qwen3-coder-next
  model: alibaba/qwen3-coder-next
  edit_format: diff
  pass_rate_2: 71.0
  percent_cases_well_formed: 91.3
  total_cost: 4.20
  released: 2026-05-01
"""

# Edge cases: missing model, missing pass_rate, malformed values
AIDER_YAML_MALFORMED = """
- dirname: 2026-01-01--no-model
  pass_rate_2: 70.0
- dirname: 2026-01-01--no-pass-rate
  model: openai/gpt-4o
- dirname: 2026-01-01--bad-pass-rate
  model: openai/o1
  pass_rate_2: "not-a-number"
- dirname: 2026-01-01--out-of-range
  model: openai/o2
  pass_rate_2: 150
- dirname: 2026-01-01--no-slash
  model: bare-model-name
  pass_rate_2: 50.0
"""
