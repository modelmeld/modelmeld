"""ProviderAdapter ABC contract tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
)


def test_cannot_instantiate_abstract_class() -> None:
    with pytest.raises(TypeError):
        ProviderAdapter()  # type: ignore[abstract]


def test_missing_method_blocks_instantiation() -> None:
    class Partial(ProviderAdapter):
        name = "partial"

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            raise NotImplementedError

        # missing stream_chat, health

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_concrete_implementation_works() -> None:
    class Concrete(ProviderAdapter):
        name = "concrete"

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            return ChatCompletion(
                model=request.model,
                choices=[Choice(index=0, message=ResponseMessage(content="ok"))],
            )

        async def stream_chat(
            self, request: ChatCompletionRequest
        ) -> AsyncIterator[ChatCompletionChunk]:
            if False:  # pragma: no cover
                yield  # async generator marker

        async def health(self) -> bool:
            return True

    adapter = Concrete()
    assert adapter.name == "concrete"


async def test_close_has_default_no_op() -> None:
    class C(ProviderAdapter):
        name = "c"

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            raise NotImplementedError

        async def stream_chat(
            self, request: ChatCompletionRequest
        ) -> AsyncIterator[ChatCompletionChunk]:
            if False:  # pragma: no cover
                yield

        async def health(self) -> bool:
            return True

    await C().close()  # should not raise


def test_adapter_error_is_exception() -> None:
    err = AdapterError("upstream timeout")
    assert isinstance(err, Exception)
    assert str(err) == "upstream timeout"
