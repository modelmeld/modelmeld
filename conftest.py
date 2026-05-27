"""Shared pytest config for modelmeld. Adds --gpu flag for live-vLLM tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption(
            "--gpu",
            action="store_true",
            default=False,
            help="run tests marked @pytest.mark.requires_gpu (needs RUNPOD_API_KEY + provisioned endpoint)",
        )
    except ValueError:
        pass  # already registered by a sibling conftest (combined run)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu flag and live vLLM endpoint")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip)
