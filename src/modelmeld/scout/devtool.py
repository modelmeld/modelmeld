# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Dev-tool fingerprinter.

Identifies which client tool likely originated a request — Cursor, Claude Code,
Aider, Cline, or generic OpenAI/Anthropic SDK usage. Detection is best-effort
regex matching against message text; false positives and negatives are expected.

Used for:
    - Per-tool savings benchmarks (deferred work)
    - Per-tool heuristic tuning in the Scout (future)
    - Audit-log enrichment

Patterns live behind a `PatternProvider` ABC so operators can add custom
detection for internal dev tools without forking the OSS package.
`DefaultPatternProvider` ships the full current catalog; the
*methodology* for deriving these patterns (observation harness over real
gateway traffic) lives in `modelmeld_enterprise.routing_tuning`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    SystemMessage,
    TextPart,
    UserMessage,
)


class DevTool(str, Enum):
    """Originating developer tool detected from request shape."""

    CURSOR = "cursor"
    CLAUDE_CODE = "claude_code"
    AIDER = "aider"
    CLINE = "cline"
    OPENCODE = "opencode"
    GITHUB_COPILOT = "github_copilot"
    OPENAI_SDK = "openai_sdk"
    ANTHROPIC_SDK = "anthropic_sdk"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Fingerprint:
    """Output of `Fingerprinter.identify()`."""

    tool: DevTool
    confidence: float
    matched_signals: list[str] = field(default_factory=list)


class PatternProvider(ABC):
    """Source of regex patterns per dev tool (injection seam).

    `Fingerprinter` consults this to decide which tool sent a request.
    Override to add custom internal-dev-tool patterns or tune detection
    sensitivity per deployment.
    """

    @abstractmethod
    def patterns_for(self, tool: DevTool) -> list[re.Pattern[str]]:
        """Compiled patterns whose match against request text indicates `tool`.

        Returning an empty list means the provider has no opinion about
        this tool — the fingerprinter will simply never identify it.
        """

    @abstractmethod
    def supported_tools(self) -> list[DevTool]:
        """All DevTool values this provider knows how to detect."""


# Each tool gets a list of regex patterns. Hits across messages are summed; the
# tool with the most hits (and ≥1) wins. Patterns are case-insensitive.
_DEFAULT_PATTERNS: dict[DevTool, list[re.Pattern[str]]] = {
    DevTool.CURSOR: [
        re.compile(r"\bCursor\b(?!\s+(University|College))", re.IGNORECASE),
        re.compile(r"<custom_instructions>", re.IGNORECASE),
        re.compile(r"powerful agentic AI coding assistant", re.IGNORECASE),
        re.compile(r"You are a coding assistant in Cursor", re.IGNORECASE),
        re.compile(r"<user_info>|</user_info>", re.IGNORECASE),
    ],
    DevTool.CLAUDE_CODE: [
        re.compile(r"You are Claude Code", re.IGNORECASE),
        re.compile(r"Anthropic'?s official CLI", re.IGNORECASE),
        re.compile(r"<system-reminder>", re.IGNORECASE),
        re.compile(r"\bClaudeCode\b", re.IGNORECASE),
        re.compile(r"<function_calls>", re.IGNORECASE),
    ],
    DevTool.AIDER: [
        re.compile(r"SEARCH/REPLACE block", re.IGNORECASE),
        re.compile(r"<<<<<<< SEARCH", re.IGNORECASE),
        re.compile(r"^>>>>>>> REPLACE", re.IGNORECASE | re.MULTILINE),
        re.compile(r"=======\n", re.IGNORECASE),
        re.compile(r"\baider\b(?:'s| chat| repo)?", re.IGNORECASE),
    ],
    DevTool.CLINE: [
        re.compile(r"\bI am Cline\b", re.IGNORECASE),
        re.compile(r"<read_file>|<execute_command>|<write_to_file>", re.IGNORECASE),
        re.compile(r"You are Cline\b", re.IGNORECASE),
    ],
    # opencode (github.com/sst/opencode) — SST's terminal coding agent.
    # Known limitation: opencode actively spoofs the upstream provider's
    # official-CLI prompt when routing to that provider's API. For Anthropic
    # routing, opencode injects "You are Claude Code, Anthropic's official
    # CLI for Claude" verbatim. That means opencode → Anthropic traffic
    # will fingerprint as CLAUDE_CODE here (the spoof is intentional on
    # opencode's side, so opencode requests look indistinguishable from
    # real Claude Code at the wire level). opencode → OpenAI/Gemini traffic
    # preserves opencode's own identity strings, which these patterns catch.
    DevTool.OPENCODE: [
        re.compile(r"\bI'?m opencode\b", re.IGNORECASE),
        re.compile(r"\bYou are opencode\b", re.IGNORECASE),
        re.compile(r"\bopencode\b[^.\n]{0,40}\bAI coding (?:assistant|agent)\b", re.IGNORECASE),
    ],
    DevTool.GITHUB_COPILOT: [
        re.compile(r"You are GitHub Copilot", re.IGNORECASE),
        re.compile(r"#GitHubCopilot", re.IGNORECASE),
    ],
}


class DefaultPatternProvider(PatternProvider):
    """Ships the production-tuned catalog as OSS default.

    Subclass to add custom patterns:

        class MyProvider(DefaultPatternProvider):
            def patterns_for(self, tool):
                if tool is DevTool.CURSOR:
                    return super().patterns_for(tool) + [
                        re.compile(r"<my-internal-fork-marker>", re.IGNORECASE),
                    ]
                return super().patterns_for(tool)
    """

    def patterns_for(self, tool: DevTool) -> list[re.Pattern[str]]:
        return _DEFAULT_PATTERNS.get(tool, [])

    def supported_tools(self) -> list[DevTool]:
        return list(_DEFAULT_PATTERNS.keys())


class Fingerprinter:
    """Detect which dev tool sent a request by inspecting message text."""

    def __init__(self, patterns: PatternProvider | None = None) -> None:
        self.patterns = patterns or DefaultPatternProvider()

    def identify(self, request: ChatCompletionRequest) -> Fingerprint:
        text = _gather_text(request)
        scores: dict[DevTool, list[str]] = {}
        for tool in self.patterns.supported_tools():
            hits = [
                p.pattern for p in self.patterns.patterns_for(tool)
                if p.search(text)
            ]
            if hits:
                scores[tool] = hits

        if not scores:
            return Fingerprint(tool=DevTool.UNKNOWN, confidence=0.0)

        # Tool with the most hits wins; ties go to the first-seen tool (dict order).
        winner = max(scores, key=lambda t: len(scores[t]))
        hits = scores[winner]
        # Confidence: capped at 1.0; first hit gives 0.6, each subsequent adds 0.2.
        confidence = min(1.0, 0.6 + 0.2 * (len(hits) - 1))
        return Fingerprint(tool=winner, confidence=confidence, matched_signals=hits)


def _gather_text(request: ChatCompletionRequest) -> str:
    pieces: list[str] = []
    for msg in request.messages:
        if isinstance(msg, (SystemMessage, UserMessage)):
            content = msg.content
            if isinstance(content, str):
                pieces.append(content)
            else:
                pieces.extend(p.text for p in content if isinstance(p, TextPart))
        # Assistant + Tool messages are prior context; tool-tag signals usually
        # appear in system or current-turn user content.
    return "\n".join(pieces)
