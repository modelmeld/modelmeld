# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Run the gateway server: `python -m modelmeld`."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from modelmeld.config import GatewaySettings

    settings = GatewaySettings()
    uvicorn.run(
        "modelmeld.api.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
