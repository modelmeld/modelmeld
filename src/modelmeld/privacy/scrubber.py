# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""PII / secret scrubbing for cloud egress.

`RegexScrubber` runs a fixed set of patterns over every text field in a
`ChatCompletionRequest`. Image, audio, and tool-definition fields are left
intact; tool-call arguments ARE scrubbed because they often carry user data.

Tradeoff: regex-based detection has known false-positive and false-negative
rates. This is a defense-in-depth layer, not a guarantee. Named-entity
detection is planned as an optional upgrade.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    FunctionCall,
    Message,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# Maximum length (in characters) of any single text field we scrub. Inputs
# longer than this are truncated for the purpose of scrubbing only — the
# original request is forwarded as-is past the size-limit middleware (which
# enforces the body-size hard cap). This bound prevents ReDoS-adjacent
# pathological-input slowdowns on the patterns below, several of which use
# unbounded `{40,}`-style quantifiers.
_MAX_SCRUB_LENGTH = 256 * 1024  # 256 KB per text field

# Patterns chosen to be high-precision; conservative is the right bias.
# Each entry maps a label to a compiled regex.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(
        r"\b(?:\d{4}[ -]?){3}\d{4}\b"
    ),
    "AWS_ACCESS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Anthropic SHOULD come first so its more specific pattern wins over OPENAI.
    "ANTHROPIC_API_KEY": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "OPENAI_API_KEY": re.compile(r"\bsk-[A-Za-z0-9_\-]{40,}\b"),
    "GITHUB_PAT": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "GITHUB_FINE_GRAINED_PAT": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # US phone — restricted to 10-digit forms to keep false positives down.
    # Leading boundary is the `(` itself for the parenthesized form; `\b` for bare digits.
    "PHONE_US": re.compile(
        r"(?:\(\d{3}\)\s?|\b\d{3}[ .-])\d{3}[ .-]\d{4}\b"
    ),
}


@dataclass(frozen=True)
class Redaction:
    """One PII match — type + count. Used for audit / metrics."""

    label: str
    count: int


class Scrubber(ABC):
    """Strip PII / secrets from outgoing chat requests."""

    name: str

    @abstractmethod
    def scrub_text(self, text: str) -> str:
        """Scrub a single string. Pure function."""

    @abstractmethod
    def scrub_request(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, list[Redaction]]:
        """Return (scrubbed request, list of redactions applied)."""


class RegexScrubber(Scrubber):
    name = "regex"

    def __init__(self, patterns: dict[str, re.Pattern[str]] | None = None) -> None:
        self.patterns = patterns if patterns is not None else _PATTERNS

    def _redact_label(self, label: str) -> str:
        return f"<REDACTED:{label}>"

    def scrub_text(self, text: str) -> str:
        # Cap pathological inputs. Scrubbing a 5 MB string against ten
        # unbounded patterns can take many seconds; if a text field is
        # that large it's already crossed the body-size middleware OR
        # is internal (memory store, audit hash input). Bound regex work
        # at 256 KB per field — beyond that we still scrub the head but
        # leave the tail untouched. The body-size middleware is the
        # outer guarantee; this is defense in depth on the regex engine.
        if len(text) > _MAX_SCRUB_LENGTH:
            head = text[:_MAX_SCRUB_LENGTH]
            tail = text[_MAX_SCRUB_LENGTH:]
            for label, pattern in self.patterns.items():
                head = pattern.sub(self._redact_label(label), head)
            return head + tail
        for label, pattern in self.patterns.items():
            text = pattern.sub(self._redact_label(label), text)
        return text

    def _count(self, text: str) -> list[Redaction]:
        counts: list[Redaction] = []
        # Same bound as scrub_text — keep findall() work bounded.
        if len(text) > _MAX_SCRUB_LENGTH:
            text = text[:_MAX_SCRUB_LENGTH]
        for label, pattern in self.patterns.items():
            n = len(pattern.findall(text))
            if n:
                counts.append(Redaction(label=label, count=n))
        return counts

    def _scrub_str_or_parts(
        self, content: str | list[TextPart] | None
    ) -> str | list[TextPart] | None:
        if content is None:
            return None
        if isinstance(content, str):
            return self.scrub_text(content)
        return [self._scrub_part(p) for p in content]  # pyright: ignore[reportReturnType]

    def _scrub_part(self, part: object) -> object:
        if isinstance(part, TextPart):
            return TextPart(type="text", text=self.scrub_text(part.text))
        return part  # image, audio, etc. pass through untouched

    def _scrub_message(self, msg: Message) -> Message:
        if isinstance(msg, SystemMessage):
            return SystemMessage(
                role="system",
                content=self._scrub_str_or_parts(msg.content),  # type: ignore[arg-type]
                name=msg.name,
            )
        if isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                new_content: object = self.scrub_text(content)
            else:
                new_content = [self._scrub_part(p) for p in content]
            return UserMessage(
                role="user",
                content=new_content,  # type: ignore[arg-type]
                name=msg.name,
            )
        if isinstance(msg, AssistantMessage):
            return AssistantMessage(
                role="assistant",
                content=self._scrub_str_or_parts(msg.content),
                name=msg.name,
                tool_calls=[self._scrub_tool_call(tc) for tc in msg.tool_calls]
                if msg.tool_calls
                else None,
                refusal=self.scrub_text(msg.refusal) if msg.refusal else None,
            )
        if isinstance(msg, ToolMessage):
            return ToolMessage(
                role="tool",
                content=self._scrub_str_or_parts(msg.content),  # type: ignore[arg-type]
                tool_call_id=msg.tool_call_id,
            )
        return msg

    def _scrub_tool_call(self, call: ToolCall) -> ToolCall:
        return ToolCall(
            id=call.id,
            type=call.type,
            function=FunctionCall(
                name=call.function.name,
                arguments=self.scrub_text(call.function.arguments),
            ),
        )

    def scrub_request(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, list[Redaction]]:
        # Count redactions across all message text before scrubbing.
        redactions: list[Redaction] = []
        all_text = self._collect_text(request)
        for redaction in self._count(all_text):
            redactions.append(redaction)

        new_messages = [self._scrub_message(m) for m in request.messages]
        new_request = request.model_copy(update={"messages": new_messages})
        return new_request, redactions

    def _collect_text(self, request: ChatCompletionRequest) -> str:
        pieces: list[str] = []
        for msg in request.messages:
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    pieces.append(msg.content)
                elif isinstance(msg.content, list):
                    pieces.extend(p.text for p in msg.content)
                if msg.refusal:
                    pieces.append(msg.refusal)
                if msg.tool_calls:
                    pieces.extend(tc.function.arguments for tc in msg.tool_calls)
                continue
            content = msg.content
            if isinstance(content, str):
                pieces.append(content)
            else:
                for part in content:
                    if isinstance(part, TextPart):
                        pieces.append(part.text)
        return "\n".join(pieces)
