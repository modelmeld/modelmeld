"""GET /version — deploy verification endpoint.

Returns a JSON snapshot of what's running on this instance: the
modelmeld OSS version, the modelmeld-enterprise version if installed,
the deployed commit SHA, and the size of the live model registry.
Mirrors `/healthz` in placement (root path, no auth, no prefix) so
operators can verify a deployment with a single HTTP request.

Field semantics:

- `modelmeld` — version of this OSS package per installed metadata.
- `modelmeld_enterprise` — version of the enterprise package if it
  imports cleanly, otherwise `null`. OSS-only deployments see `null`
  here; that's the expected state.
- `deployed_commit` — git SHA of the build. Reads (in order)
  `MODELMELD_DEPLOYED_COMMIT` (set explicitly), then `RENDER_GIT_COMMIT`
  (auto-injected by Render on native-Python services). Returns
  `null` if neither is set.
- `registry_size` — number of models in the live `ModelRegistry`.
  The default OSS registry and a richer multi-provider registry have
  very different sizes, so this is a quick way to confirm the
  expected registry was wired up.
"""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from fastapi import APIRouter, Request

router = APIRouter()


def _package_version(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def _deployed_commit() -> str | None:
    return (
        os.environ.get("MODELMELD_DEPLOYED_COMMIT")
        or os.environ.get("RENDER_GIT_COMMIT")
        or None
    )


@router.get("/version")
async def get_version(request: Request) -> dict[str, str | int | None]:
    registry = getattr(request.app.state, "model_registry", None)
    registry_size = len(registry) if registry is not None else 0

    return {
        "modelmeld": _package_version("modelmeld"),
        "modelmeld_enterprise": _package_version("modelmeld-enterprise"),
        "deployed_commit": _deployed_commit(),
        "registry_size": registry_size,
    }
