"""Boundary-contract proof: modelmeld alone touches no enterprise infrastructure.

Two complementary checks:
1. Import-time isolation — a fresh Python subprocess imports modelmeld and its
   submodules, then asserts no SQLAlchemy / Alembic / asyncpg / Redis / Qdrant
   modules ended up in sys.modules. Runs in a subprocess so the surrounding
   pytest session (which DOES pull in SQLAlchemy via enterprise tests) doesn't
   pollute the result.
2. Runtime network — a socket-level spy installed for the duration of a chat
   request. Any TCP connect to a known DB / cache port (5432 / 6379 / 6333) is
   recorded; the test asserts the recording stays empty.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

from modelmeld.api.server import build_app


_FORBIDDEN_MODULE_PREFIXES = (
    "sqlalchemy",
    "alembic",
    "asyncpg",
    "aiosqlite",
    "psycopg2",
    "psycopg",
    "redis",
    "aioredis",
    "qdrant_client",
    "modelmeld_enterprise",
)

_DB_AND_CACHE_PORTS = {
    5432,   # Postgres
    5433,   # Postgres alt
    6379,   # Redis
    6333,   # Qdrant HTTP
    6334,   # Qdrant gRPC
}


def test_core_engine_imports_no_database_or_enterprise_libraries() -> None:
    """A fresh Python imports only modelmeld; no SQLAlchemy/Postgres/etc. leak in."""
    probe = (
        "import sys\n"
        "import modelmeld\n"
        "from modelmeld import adapters, api, config, hooks, privacy, router, scout\n"
        "from modelmeld.api import server, schemas\n"
        "from modelmeld.api.server import build_app\n"
        "from modelmeld.api.routes import chat, healthz, models\n"
        "build_app()\n"
        f"forbidden = {_FORBIDDEN_MODULE_PREFIXES!r}\n"
        "leaked = sorted({m for m in sys.modules if m.startswith(forbidden)})\n"
        "if leaked:\n"
        "    print('LEAKED:', leaked)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"modelmeld import leaked forbidden modules:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_default_app_makes_no_database_connection_on_request() -> None:
    """Runtime: socket spy records any TCP connect to known DB / cache ports."""
    original_connect = socket.socket.connect
    attempts: list[tuple[str, int]] = []

    def spy(self: socket.socket, address: object) -> None:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if isinstance(port, int) and port in _DB_AND_CACHE_PORTS:
                attempts.append((str(host), port))
        return original_connect(self, address)

    with patch.object(socket.socket, "connect", spy):
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
    assert response.status_code == 200
    assert attempts == [], (
        f"core-engine reached out to {attempts}; it must not contact any DB/cache."
    )


def test_default_app_state_has_no_db_engine_or_session() -> None:
    """app.state is the contract surface; verify no DB handle leaks via it."""
    app = build_app()
    for forbidden in ("enterprise_engine", "enterprise_session_maker", "audit_logger"):
        assert not hasattr(app.state, forbidden), (
            f"core-engine app.state should not expose {forbidden}"
        )
